#include "./lut_max_cell_error.hpp"

#include <ceres/jet.h>
#include <omp.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "./cameramodels.hpp"
#include "./pybind_utils.hpp"

namespace lensboy {

namespace {

constexpr int kInterpNearest = 0;
constexpr int kInterpBilinear = 1;
constexpr int kInterpBicubic = 2;

// Penalty multiplier on cell-membership violation. Chosen large enough that
// any boundary crossing dominates any interior gain in the objective, but
// finite so that gradient ascent still makes monotonic progress.
constexpr double kPenaltyLambda = 1.0e30;

constexpr int kMaxLineSearchSteps = 30;
constexpr double kStepGrowth = 1.4;
constexpr double kStepShrink = 0.5;
constexpr double kMinStepRatio = 1.0e-12;

// Approximate pixel spacing between gradient-ascent seeds inside a cell.
// One seed per ~32 px of cell extent in each direction, with a one-seed
// floor for tiny cells. Distortion functions are smooth, so the objective
// is single-modal in every reasonable cell — coarse seeding suffices.
constexpr double kSeedStridePixels = 32.0;

using Jet2 = ceres::Jet<double, 2>;

template <typename T>
inline T relu(
    const T& x
) {
    return x > T(0.0) ? x : T(0.0);
}

template <typename T>
inline T penalty_outside_cell(
    const T& px,
    const T& py,
    double x0,
    double x1,
    double y0,
    double y1
) {
    return relu(T(x0) - px) + relu(px - T(x1)) + relu(T(y0) - py) +
           relu(py - T(y1));
}

// Catmull-Rom weights matching the Python implementation.
template <typename T>
inline void catmull_rom_weights_T(
    const T& t,
    T w[4]
) {
    const T t2 = t * t;
    const T t3 = t2 * t;
    w[0] = T(-0.5) * t + t2 + T(-0.5) * t3;
    w[1] = T(1.0) + T(-2.5) * t2 + T(1.5) * t3;
    w[2] = T(0.5) * t + T(2.0) * t2 + T(-1.5) * t3;
    w[3] = T(-0.5) * t2 + T(0.5) * t3;
}

template <typename T>
inline void interp_lut_nearest(
    const double* lut_xy_grid,
    int Wgrid,
    int Hgrid,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& px,
    const T& py,
    T& out_x,
    T& out_y
) {
    const double px_a = scalar_value(px);
    const double py_a = scalar_value(py);
    const double gx = (px_a - grid_x_min) * grid_scale_x;
    const double gy = (py_a - grid_y_min) * grid_scale_y;
    // Match numpy's np.rint (round-half-to-even) so the C++ optimiser and
    // the Python LUT query agree at exact half-integer grid coordinates.
    // std::lround rounds half-away-from-zero and disagreed at .5 values.
    int ix = static_cast<int>(std::nearbyint(gx));
    int iy = static_cast<int>(std::nearbyint(gy));
    ix = std::clamp(ix, 0, Wgrid - 1);
    iy = std::clamp(iy, 0, Hgrid - 1);
    const double* node = lut_xy_grid + (iy * Wgrid + ix) * 2;
    out_x = T(node[0]);
    out_y = T(node[1]);
}

template <typename T>
inline void interp_lut_bilinear(
    const double* lut_xy_grid,
    int Wgrid,
    int Hgrid,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& px,
    const T& py,
    T& out_x,
    T& out_y
) {
    const T gx = (px - T(grid_x_min)) * T(grid_scale_x);
    const T gy = (py - T(grid_y_min)) * T(grid_scale_y);

    int ix0 = 0;
    int iy0 = 0;
    T tx(0.0);
    T ty(0.0);

    if (Wgrid > 1) {
        const T gx_clip =
            clamp_T(gx, T(0.0), T(static_cast<double>(Wgrid - 1)));
        ix0 = static_cast<int>(std::floor(scalar_value(gx_clip)));
        ix0 = std::clamp(ix0, 0, Wgrid - 2);
        tx = gx_clip - T(static_cast<double>(ix0));
    }
    if (Hgrid > 1) {
        const T gy_clip =
            clamp_T(gy, T(0.0), T(static_cast<double>(Hgrid - 1)));
        iy0 = static_cast<int>(std::floor(scalar_value(gy_clip)));
        iy0 = std::clamp(iy0, 0, Hgrid - 2);
        ty = gy_clip - T(static_cast<double>(iy0));
    }

    const int ix1 = std::min(ix0 + 1, Wgrid - 1);
    const int iy1 = std::min(iy0 + 1, Hgrid - 1);

    const double* v00 = lut_xy_grid + (iy0 * Wgrid + ix0) * 2;
    const double* v10 = lut_xy_grid + (iy0 * Wgrid + ix1) * 2;
    const double* v01 = lut_xy_grid + (iy1 * Wgrid + ix0) * 2;
    const double* v11 = lut_xy_grid + (iy1 * Wgrid + ix1) * 2;

    const T one_minus_tx = T(1.0) - tx;
    const T one_minus_ty = T(1.0) - ty;
    const T top_x = one_minus_tx * v00[0] + tx * v10[0];
    const T top_y = one_minus_tx * v00[1] + tx * v10[1];
    const T bot_x = one_minus_tx * v01[0] + tx * v11[0];
    const T bot_y = one_minus_tx * v01[1] + tx * v11[1];
    out_x = one_minus_ty * top_x + ty * bot_x;
    out_y = one_minus_ty * top_y + ty * bot_y;
}

template <typename T>
inline void interp_lut_bicubic(
    const double* lut_xy_grid,
    int Wgrid,
    int Hgrid,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& px,
    const T& py,
    T& out_x,
    T& out_y
) {
    if (Wgrid < 4 || Hgrid < 4) {
        interp_lut_bilinear(
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            px,
            py,
            out_x,
            out_y
        );
        return;
    }

    const T gx = (px - T(grid_x_min)) * T(grid_scale_x);
    const T gy = (py - T(grid_y_min)) * T(grid_scale_y);
    const T gx_clip =
        clamp_T(gx, T(0.0), T(static_cast<double>(Wgrid - 1)));
    const T gy_clip =
        clamp_T(gy, T(0.0), T(static_cast<double>(Hgrid - 1)));

    const int ix1 = static_cast<int>(std::floor(scalar_value(gx_clip)));
    const int iy1 = static_cast<int>(std::floor(scalar_value(gy_clip)));

    const bool has_full_support =
        (ix1 >= 1) && (ix1 <= Wgrid - 3) && (iy1 >= 1) && (iy1 <= Hgrid - 3);
    if (!has_full_support) {
        interp_lut_bilinear(
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            px,
            py,
            out_x,
            out_y
        );
        return;
    }

    const T tx = gx_clip - T(static_cast<double>(ix1));
    const T ty = gy_clip - T(static_cast<double>(iy1));

    T wx[4], wy[4];
    catmull_rom_weights_T(tx, wx);
    catmull_rom_weights_T(ty, wy);

    out_x = T(0.0);
    out_y = T(0.0);
    for (int j = 0; j < 4; ++j) {
        const int sample_y = iy1 + j - 1;
        for (int i = 0; i < 4; ++i) {
            const int sample_x = ix1 + i - 1;
            const double* node = lut_xy_grid + (sample_y * Wgrid + sample_x) * 2;
            const T w = wy[j] * wx[i];
            out_x = out_x + w * node[0];
            out_y = out_y + w * node[1];
        }
    }
}

template <typename T>
inline void interp_lut_dispatch(
    int interpolation_mode,
    const double* lut_xy_grid,
    int Wgrid,
    int Hgrid,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& px,
    const T& py,
    T& out_x,
    T& out_y
) {
    if (interpolation_mode == kInterpNearest) {
        interp_lut_nearest(
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            px,
            py,
            out_x,
            out_y
        );
    } else if (interpolation_mode == kInterpBilinear) {
        interp_lut_bilinear(
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            px,
            py,
            out_x,
            out_y
        );
    } else {
        interp_lut_bicubic(
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            px,
            py,
            out_x,
            out_y
        );
    }
}

// Cubic B-spline tensor-product evaluation with a double-valued grid and
// templated coordinates. Mirrors eval_bspline2d_uniform_cubic_clamped from
// cameramodels.hpp, but avoids materialising a Jet copy of the knot grid.
template <typename T>
T eval_bspline2d_double_grid(
    const double* grid,
    int Nx,
    int Ny,
    const T& x_spline,
    const T& y_spline
) {
    constexpr double eps = 1e-12;
    T gx = clamp_T(x_spline, T(0.0), T(static_cast<double>(Nx - 1) - eps));
    T gy = clamp_T(y_spline, T(0.0), T(static_cast<double>(Ny - 1) - eps));

    const int ix = static_cast<int>(std::floor(scalar_value(gx)));
    const int iy = static_cast<int>(std::floor(scalar_value(gy)));

    const T u = gx - T(static_cast<double>(ix));
    const T v = gy - T(static_cast<double>(iy));

    T wx[4], wy[4];
    cubic_bspline_basis_uniform(u, wx);
    cubic_bspline_basis_uniform(v, wy);

    const int xs[4] = {
        clamp_int(ix - 1, 0, Nx - 1),
        clamp_int(ix + 0, 0, Nx - 1),
        clamp_int(ix + 1, 0, Nx - 1),
        clamp_int(ix + 2, 0, Nx - 1)
    };
    const int ys[4] = {
        clamp_int(iy - 1, 0, Ny - 1),
        clamp_int(iy + 0, 0, Ny - 1),
        clamp_int(iy + 1, 0, Ny - 1),
        clamp_int(iy + 2, 0, Ny - 1)
    };

    T acc(0.0);
    for (int b = 0; b < 4; ++b) {
        const int row0 = ys[b] * Nx;
        const T wyb = wy[b];
        for (int a = 0; a < 4; ++a) {
            acc = acc + (wyb * wx[a]) * grid[row0 + xs[a]];
        }
    }
    return acc;
}

// Project a normalized point (n_x, n_y, 1) through PinholeSplined with double
// constants. Templated on T for autodiff over the input point.
template <typename T>
inline void project_pinhole_splined_n(
    const SplineMap& map,
    const double* pinhole_params,  // fx, fy, cx, cy
    const double* dx_grid,
    const double* dy_grid,
    const T& nx,
    const T& ny,
    T& px,
    T& py
) {
    T x_spline, y_spline;
    map.normalized_to_grid_coords(nx, ny, x_spline, y_spline);

    const T dx =
        eval_bspline2d_double_grid(dx_grid, map.Nx, map.Ny, x_spline, y_spline);
    const T dy =
        eval_bspline2d_double_grid(dy_grid, map.Nx, map.Ny, x_spline, y_spline);

    px = T(pinhole_params[0]) * (nx + dx) + T(pinhole_params[2]);
    py = T(pinhole_params[1]) * (ny + dy) + T(pinhole_params[3]);
}

// Project a normalized point through OpenCV with double constants. Wraps
// the 18 intrinsics into a stack-local T buffer once and reuses the existing
// templated project_opencv to avoid re-implementing the distortion math.
template <typename T>
inline void project_opencv_n(
    const double* intrinsics,
    const T& nx,
    const T& ny,
    T& px,
    T& py
) {
    T wrapped[18];
    for (int i = 0; i < 18; ++i) {
        wrapped[i] = T(intrinsics[i]);
    }
    Vec3<T> point_in_cam(nx, ny, T(1.0));
    Vec2<T> result;
    project_opencv<T>(wrapped, point_in_cam, result);
    px = result[0];
    py = result[1];
}

// Angular error in degrees between rays (exact_x, exact_y, 1) and
// (approx_x, approx_y, 1). Uses atan2(‖cross‖, dot) instead of acos(dot/norms)
// because acos near 1 catastrophically loses precision for tiny angles
// (1 - cos(angle) is below double-precision resolution well before angles
// stop mattering).
inline double angular_error_deg_from_xy(
    double exact_x,
    double exact_y,
    double approx_x,
    double approx_y
) {
    // cross((ex, ey, 1), (ax, ay, 1)) = (ey - ay, ax - ex, ex*ay - ey*ax)
    const double cx = exact_y - approx_y;
    const double cy = approx_x - exact_x;
    const double cz = exact_x * approx_y - exact_y * approx_x;
    const double cross_norm = std::sqrt(cx * cx + cy * cy + cz * cz);
    const double dot = exact_x * approx_x + exact_y * approx_y + 1.0;
    return std::atan2(cross_norm, dot) * (180.0 / M_PI);
}

// Per-cell gradient ascent on f(n) = sin²(angle between rays
// (approx_xy(project(n)), 1) and (n, 1)) minus a ReLU penalty pushing
// project(n) inside the pixel cell.
//
// We optimise sin² of the angle (monotone in the angular error we report)
// rather than ‖approx_xy − n‖²: for off-axis cells (|n| ≫ 0) the squared
// xy distance and the angular error have *different* argmaxes, since the
// angular formula divides by ‖r1‖·‖r2‖ which varies with position.
//
// The Project callable is invoked as:
//   project(nx, ny, px, py)        for T=double
//   project(jet_nx, jet_ny, ...)   for T=Jet2
//
// Returns (max_angular_error_deg, peak_pixel_x, peak_pixel_y).
template <typename Project>
struct CellMaximizer {
    Project project;
    int interpolation_mode;
    const double* lut_xy_grid;
    int Wgrid;
    int Hgrid;
    double grid_x_min;
    double grid_y_min;
    double grid_scale_x;
    double grid_scale_y;
    int max_iters;
    double grad_tol;

    template <typename T>
    inline T eval(
        const T& nx,
        const T& ny,
        double cell_x0,
        double cell_x1,
        double cell_y0,
        double cell_y1
    ) const {
        T px, py;
        project(nx, ny, px, py);
        T approx_x, approx_y;
        interp_lut_dispatch(
            interpolation_mode,
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            px,
            py,
            approx_x,
            approx_y
        );
        // sin²(angle) between rays (nx, ny, 1) and (approx_x, approx_y, 1):
        //   sin²(θ) = ‖cross‖² / (‖r1‖² · ‖r2‖²)
        // cross((nx, ny, 1), (approx_x, approx_y, 1)) =
        //   (ny - approx_y, approx_x - nx, nx*approx_y - ny*approx_x).
        const T cross_x = ny - approx_y;
        const T cross_y = approx_x - nx;
        const T cross_z = nx * approx_y - ny * approx_x;
        const T cross_norm_sq =
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
        const T n_norm_sq = nx * nx + ny * ny + T(1.0);
        const T approx_norm_sq =
            approx_x * approx_x + approx_y * approx_y + T(1.0);
        const T sin_sq = cross_norm_sq / (n_norm_sq * approx_norm_sq);
        const T pen =
            penalty_outside_cell(px, py, cell_x0, cell_x1, cell_y0, cell_y1);
        return sin_sq - T(kPenaltyLambda) * pen;
    }

    // Gradient ascent from a single initial point in normalised space.
    // Returns the final (nx, ny) and objective value via out parameters.
    // No bbox clamp on the search — the corner-bbox is an approximation of
    // the cell's image in normalised space and can be too tight for highly
    // nonlinear models. The ReLU penalty inside `eval` enforces the true
    // constraint (project(n) ∈ pixel cell).
    void optimize_from(
        double cell_x0,
        double cell_x1,
        double cell_y0,
        double cell_y1,
        double n_span,
        double init_nx,
        double init_ny,
        double& out_nx,
        double& out_ny,
        double& out_f
    ) const {
        double nx = init_nx;
        double ny = init_ny;
        double step_size = std::max(n_span * 0.25, 1.0e-12);
        const double min_step = std::max(n_span * kMinStepRatio, 1.0e-30);

        double current_f =
            eval<double>(nx, ny, cell_x0, cell_x1, cell_y0, cell_y1);

        for (int iter = 0; iter < max_iters; ++iter) {
            Jet2 jnx(nx, 0);
            Jet2 jny(ny, 1);
            const Jet2 f =
                eval<Jet2>(jnx, jny, cell_x0, cell_x1, cell_y0, cell_y1);

            const double gx = f.v[0];
            const double gy = f.v[1];
            const double grad_norm = std::sqrt(gx * gx + gy * gy);
            if (grad_norm < grad_tol) {
                current_f = f.a;
                break;
            }

            const double inv_grad_norm = 1.0 / grad_norm;
            const double dir_x = gx * inv_grad_norm;
            const double dir_y = gy * inv_grad_norm;

            double step = step_size;
            bool accepted = false;
            for (int ls = 0; ls < kMaxLineSearchSteps; ++ls) {
                const double trial_nx = nx + step * dir_x;
                const double trial_ny = ny + step * dir_y;
                const double trial_f = eval<double>(
                    trial_nx,
                    trial_ny,
                    cell_x0,
                    cell_x1,
                    cell_y0,
                    cell_y1
                );
                if (trial_f > current_f) {
                    nx = trial_nx;
                    ny = trial_ny;
                    current_f = trial_f;
                    step_size = step * kStepGrowth;
                    accepted = true;
                    break;
                }
                step *= kStepShrink;
                if (step < min_step) {
                    break;
                }
            }
            if (!accepted) {
                break;
            }
        }

        out_nx = nx;
        out_ny = ny;
        out_f = current_f;
    }

    // Multistart wrapper. Lays down a seed grid inside the cell at roughly
    // one seed per `kSeedStridePixels` of cell extent (in pixel space),
    // bilinearly interpolates the 4 corner normals to get an initial
    // normalised coordinate per seed, runs gradient ascent from each, and
    // keeps the best. Defensive against multimodal cells where a single
    // warm start would land in a sub-optimal basin.
    void run(
        double cell_x0,
        double cell_x1,
        double cell_y0,
        double cell_y1,
        const double* n00,
        const double* n10,
        const double* n01,
        const double* n11,
        double& out_max_angular_deg,
        double& out_peak_pixel_x,
        double& out_peak_pixel_y,
        double& out_delta_x,
        double& out_delta_y
    ) const {
        const double n_lo_x = std::min({n00[0], n10[0], n01[0], n11[0]});
        const double n_hi_x = std::max({n00[0], n10[0], n01[0], n11[0]});
        const double n_lo_y = std::min({n00[1], n10[1], n01[1], n11[1]});
        const double n_hi_y = std::max({n00[1], n10[1], n01[1], n11[1]});
        const double n_span_full = std::max(n_hi_x - n_lo_x, n_hi_y - n_lo_y);

        const double cell_width = cell_x1 - cell_x0;
        const double cell_height = cell_y1 - cell_y0;
        const int n_seeds_x = std::max(
            1,
            static_cast<int>(std::ceil(cell_width / kSeedStridePixels))
        );
        const int n_seeds_y = std::max(
            1,
            static_cast<int>(std::ceil(cell_height / kSeedStridePixels))
        );

        // Scale the gradient-ascent step to the seed sub-cell so an
        // optimisation from one seed doesn't immediately overshoot into a
        // neighbouring seed's basin. Line search adapts up if the basin is
        // wider.
        const int max_seeds_per_dim = std::max(n_seeds_x, n_seeds_y);
        const double per_seed_n_span = n_span_full / max_seeds_per_dim;

        double best_nx = 0.5 * (n_lo_x + n_hi_x);
        double best_ny = 0.5 * (n_lo_y + n_hi_y);
        double best_f = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_seeds_y; ++j) {
            const double t = (static_cast<double>(j) + 0.5) /
                             static_cast<double>(n_seeds_y);
            for (int i = 0; i < n_seeds_x; ++i) {
                const double s = (static_cast<double>(i) + 0.5) /
                                 static_cast<double>(n_seeds_x);
                const double w00 = (1.0 - s) * (1.0 - t);
                const double w10 = s * (1.0 - t);
                const double w01 = (1.0 - s) * t;
                const double w11 = s * t;
                const double init_nx = w00 * n00[0] + w10 * n10[0] +
                                       w01 * n01[0] + w11 * n11[0];
                const double init_ny = w00 * n00[1] + w10 * n10[1] +
                                       w01 * n01[1] + w11 * n11[1];

                double trial_nx = 0.0;
                double trial_ny = 0.0;
                double trial_f = 0.0;
                optimize_from(
                    cell_x0,
                    cell_x1,
                    cell_y0,
                    cell_y1,
                    per_seed_n_span,
                    init_nx,
                    init_ny,
                    trial_nx,
                    trial_ny,
                    trial_f
                );
                if (trial_f > best_f) {
                    best_f = trial_f;
                    best_nx = trial_nx;
                    best_ny = trial_ny;
                }
            }
        }

        double peak_px, peak_py;
        project(best_nx, best_ny, peak_px, peak_py);
        double approx_x, approx_y;
        interp_lut_dispatch<double>(
            interpolation_mode,
            lut_xy_grid,
            Wgrid,
            Hgrid,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            peak_px,
            peak_py,
            approx_x,
            approx_y
        );
        out_max_angular_deg =
            angular_error_deg_from_xy(best_nx, best_ny, approx_x, approx_y);
        out_peak_pixel_x = peak_px;
        out_peak_pixel_y = peak_py;
        out_delta_x = approx_x - best_nx;
        out_delta_y = approx_y - best_ny;
    }
};

template <typename Project>
py::array_t<double> run_max_cell_errors(
    const Project& project,
    const double* lut_xy_grid,
    int Wgrid,
    int Hgrid,
    double grid_x_min,
    double grid_x_max,
    double grid_y_min,
    double grid_y_max,
    int interpolation_mode,
    int max_iters,
    double grad_tol
) {
    require(Wgrid >= 2 && Hgrid >= 2, "LUT grid dimensions must be at least 2");
    require(
        interpolation_mode >= 0 && interpolation_mode <= 2,
        "interpolation_mode must be 0 (nearest), 1 (bilinear), or 2 (bicubic)"
    );
    require(max_iters > 0, "max_iters must be positive");

    const double grid_scale_x =
        Wgrid > 1 ? static_cast<double>(Wgrid - 1) / (grid_x_max - grid_x_min)
                  : 0.0;
    const double grid_scale_y =
        Hgrid > 1 ? static_cast<double>(Hgrid - 1) / (grid_y_max - grid_y_min)
                  : 0.0;

    const int num_cells_x = Wgrid - 1;
    const int num_cells_y = Hgrid - 1;
    const ssize_t num_cells =
        static_cast<ssize_t>(num_cells_x) * static_cast<ssize_t>(num_cells_y);

    py::array_t<double> out({num_cells, static_cast<ssize_t>(5)});
    auto ob = out.request();
    double* O = static_cast<double*>(ob.ptr);

    CellMaximizer<Project> maximizer{
        project,
        interpolation_mode,
        lut_xy_grid,
        Wgrid,
        Hgrid,
        grid_x_min,
        grid_y_min,
        grid_scale_x,
        grid_scale_y,
        max_iters,
        grad_tol
    };

    const double pixel_span_x =
        Wgrid > 1 ? (grid_x_max - grid_x_min) / static_cast<double>(Wgrid - 1)
                  : 0.0;
    const double pixel_span_y =
        Hgrid > 1 ? (grid_y_max - grid_y_min) / static_cast<double>(Hgrid - 1)
                  : 0.0;

#pragma omp parallel for schedule(static) collapse(2)
    for (int cy = 0; cy < num_cells_y; ++cy) {
        for (int cx = 0; cx < num_cells_x; ++cx) {
            const ssize_t cell_idx =
                static_cast<ssize_t>(cy) * num_cells_x + cx;

            const double cell_x0 = grid_x_min + cx * pixel_span_x;
            const double cell_x1 = grid_x_min + (cx + 1) * pixel_span_x;
            const double cell_y0 = grid_y_min + cy * pixel_span_y;
            const double cell_y1 = grid_y_min + (cy + 1) * pixel_span_y;

            const double* n00 = lut_xy_grid + (cy * Wgrid + cx) * 2;
            const double* n10 = lut_xy_grid + (cy * Wgrid + (cx + 1)) * 2;
            const double* n01 = lut_xy_grid + ((cy + 1) * Wgrid + cx) * 2;
            const double* n11 =
                lut_xy_grid + ((cy + 1) * Wgrid + (cx + 1)) * 2;

            double max_err_deg = 0.0;
            double peak_px = 0.5 * (cell_x0 + cell_x1);
            double peak_py = 0.5 * (cell_y0 + cell_y1);
            double delta_x = 0.0;
            double delta_y = 0.0;
            maximizer.run(
                cell_x0,
                cell_x1,
                cell_y0,
                cell_y1,
                n00,
                n10,
                n01,
                n11,
                max_err_deg,
                peak_px,
                peak_py,
                delta_x,
                delta_y
            );

            double* row = O + cell_idx * 5;
            row[0] = max_err_deg;
            row[1] = peak_px;
            row[2] = peak_py;
            row[3] = delta_x;
            row[4] = delta_y;
        }
    }

    return out;
}

}  // namespace

py::array_t<double> max_cell_errors_pinhole_splined(
    PinholeSplinedConfig& config,
    PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> lut_xy_grid,
    double grid_x_min,
    double grid_x_max,
    double grid_y_min,
    double grid_y_max,
    int interpolation_mode,
    int max_iters,
    double grad_tol
) {
    auto pinhole_buf = intrinsics.pinhole_parameters.request();
    require(
        pinhole_buf.ndim == 1 && pinhole_buf.shape[0] == 4,
        "pinhole_parameters must have shape (4,)"
    );
    auto dxb = intrinsics.dx_grid.request();
    auto dyb = intrinsics.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == config.num_knots_y &&
            (uint32_t)dxb.shape[1] == config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dyb.shape[0] == config.num_knots_y &&
            (uint32_t)dyb.shape[1] == config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    auto lutb = lut_xy_grid.request();
    require(
        lutb.ndim == 3 && lutb.shape[2] == 2,
        "lut_xy_grid must have shape (Hgrid, Wgrid, 2)"
    );
    const int Hgrid = static_cast<int>(lutb.shape[0]);
    const int Wgrid = static_cast<int>(lutb.shape[1]);

    const double* pinhole_params =
        static_cast<const double*>(pinhole_buf.ptr);
    const double* dx_grid_ptr = static_cast<const double*>(dxb.ptr);
    const double* dy_grid_ptr = static_cast<const double*>(dyb.ptr);
    const double* lut_ptr = static_cast<const double*>(lutb.ptr);

    const SplineMap map(config);

    auto project = [&](auto&& nx, auto&& ny, auto& px, auto& py) {
        project_pinhole_splined_n(
            map,
            pinhole_params,
            dx_grid_ptr,
            dy_grid_ptr,
            nx,
            ny,
            px,
            py
        );
    };

    return run_max_cell_errors(
        project,
        lut_ptr,
        Wgrid,
        Hgrid,
        grid_x_min,
        grid_x_max,
        grid_y_min,
        grid_y_max,
        interpolation_mode,
        max_iters,
        grad_tol
    );
}

py::array_t<double> max_cell_errors_opencv(
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> lut_xy_grid,
    double grid_x_min,
    double grid_x_max,
    double grid_y_min,
    double grid_y_max,
    int interpolation_mode,
    int max_iters,
    double grad_tol
) {
    auto intr_buf = intrinsics.request();
    require(
        intr_buf.ndim == 1 && intr_buf.shape[0] == 18,
        "intrinsics must have shape (18,) — fx, fy, cx, cy + 14 distortion"
    );

    auto lutb = lut_xy_grid.request();
    require(
        lutb.ndim == 3 && lutb.shape[2] == 2,
        "lut_xy_grid must have shape (Hgrid, Wgrid, 2)"
    );
    const int Hgrid = static_cast<int>(lutb.shape[0]);
    const int Wgrid = static_cast<int>(lutb.shape[1]);

    const double* intr_ptr = static_cast<const double*>(intr_buf.ptr);
    const double* lut_ptr = static_cast<const double*>(lutb.ptr);

    auto project = [&](auto&& nx, auto&& ny, auto& px, auto& py) {
        project_opencv_n(intr_ptr, nx, ny, px, py);
    };

    return run_max_cell_errors(
        project,
        lut_ptr,
        Wgrid,
        Hgrid,
        grid_x_min,
        grid_x_max,
        grid_y_min,
        grid_y_max,
        interpolation_mode,
        max_iters,
        grad_tol
    );
}

}  // namespace lensboy
