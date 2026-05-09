#include <ceres/jet.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "./cameramodels.hpp"
#include "./pybind_utils.hpp"

namespace lensboy {
namespace py = pybind11;

// ---------------------------------------------------------------------------
// Hash grid for O(1) nearest-neighbor lookup over seed pixels.
//
// Cell size = minimum adjacent seed spacing.  Each cell holds a short list
// of seed indices.  Query checks the cell + 8 neighbors.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Flat-array spatial index for O(1) nearest-neighbor lookup.
//
// Cell size = min adjacent seed spacing.  Flat 2D array of short vectors.
// Query checks the cell + 8 neighbors.
// ---------------------------------------------------------------------------

class SeedGrid {
   public:
    SeedGrid(
        const double* pixel_xy,
        int n,
        int seed_w,
        int seed_h
    ) {
        // Find min adjacent edge length in pixel space
        double min_edge_sq = std::numeric_limits<double>::max();
        for (int j = 0; j < seed_h; j++) {
            for (int i = 0; i < seed_w; i++) {
                int idx = j * seed_w + i;
                double x0 = pixel_xy[idx * 2], y0 = pixel_xy[idx * 2 + 1];
                if (!std::isfinite(x0)) {
                    continue;
                }
                if (i + 1 < seed_w) {
                    int idx1 = j * seed_w + (i + 1);
                    double x1 = pixel_xy[idx1 * 2], y1 = pixel_xy[idx1 * 2 + 1];
                    if (std::isfinite(x1)) {
                        double d =
                            (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
                        if (d > 0) {
                            min_edge_sq = std::min(min_edge_sq, d);
                        }
                    }
                }
                if (j + 1 < seed_h) {
                    int idx1 = (j + 1) * seed_w + i;
                    double x1 = pixel_xy[idx1 * 2], y1 = pixel_xy[idx1 * 2 + 1];
                    if (std::isfinite(x1)) {
                        double d =
                            (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
                        if (d > 0) {
                            min_edge_sq = std::min(min_edge_sq, d);
                        }
                    }
                }
            }
        }
        cell_size_ = std::sqrt(min_edge_sq);
        if (cell_size_ < 1e-6) {
            cell_size_ = 1.0;
        }
        inv_cell_ = 1.0 / cell_size_;

        // Bounding box
        double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30;
        for (int i = 0; i < n; i++) {
            double x = pixel_xy[i * 2], y = pixel_xy[i * 2 + 1];
            if (!std::isfinite(x)) {
                continue;
            }
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
            ymin = std::min(ymin, y);
            ymax = std::max(ymax, y);
        }
        x_min_ = xmin;
        y_min_ = ymin;
        gw_ = (int)std::ceil((xmax - xmin) * inv_cell_) + 1;
        gh_ = (int)std::ceil((ymax - ymin) * inv_cell_) + 1;

        // Flat array, each cell stores up to 4 indices (enough for ~1 point
        // per cell).  Use -1 as sentinel.
        constexpr int CELL_CAP = 4;
        cells_.assign((size_t)gw_ * gh_ * CELL_CAP, -1);

        for (int i = 0; i < n; i++) {
            double x = pixel_xy[i * 2], y = pixel_xy[i * 2 + 1];
            if (!std::isfinite(x) || !std::isfinite(y)) {
                continue;
            }
            int cx = to_cx(x), cy = to_cy(y);
            int base = (cy * gw_ + cx) * CELL_CAP;
            for (int s = 0; s < CELL_CAP; s++) {
                if (cells_[base + s] < 0) {
                    cells_[base + s] = i;
                    break;
                }
            }
        }
    }

    int nearest(
        double qx,
        double qy,
        const double* pixel_xy
    ) const {
        constexpr int CELL_CAP = 4;
        int cx = to_cx(qx), cy = to_cy(qy);
        int best = -1;
        double best_sq = std::numeric_limits<double>::max();
        for (int dy = -1; dy <= 1; dy++) {
            int ry = cy + dy;
            if (ry < 0 || ry >= gh_) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int rx = cx + dx;
                if (rx < 0 || rx >= gw_) {
                    continue;
                }
                int base = (ry * gw_ + rx) * CELL_CAP;
                for (int s = 0; s < CELL_CAP; s++) {
                    int idx = cells_[base + s];
                    if (idx < 0) {
                        break;
                    }
                    double ex = qx - pixel_xy[idx * 2];
                    double ey = qy - pixel_xy[idx * 2 + 1];
                    double d = ex * ex + ey * ey;
                    if (d < best_sq) {
                        best_sq = d;
                        best = idx;
                    }
                }
            }
        }
        return best;
    }

   private:
    double cell_size_, inv_cell_, x_min_, y_min_;
    int gw_, gh_;
    std::vector<int> cells_;

    int to_cx(
        double x
    ) const {
        return std::max(
            0,
            std::min(gw_ - 1, (int)std::floor((x - x_min_) * inv_cell_))
        );
    }
    int to_cy(
        double y
    ) const {
        return std::max(
            0,
            std::min(gh_ - 1, (int)std::floor((y - y_min_) * inv_cell_))
        );
    }
};

// ---------------------------------------------------------------------------
// Inverse-distance weighted interpolation from the 4 quad corners.
//
// The NN tells us grid node (ni, nj). The query pixel's position relative
// to that node picks one of the 4 adjacent quads.  Then IDW over the 4
// corners of that quad gives the initial guess -- no iteration needed.
// ---------------------------------------------------------------------------

static bool idw_interp_from_nn(
    double pixel_x,
    double pixel_y,
    int nearest_idx,
    int seed_w,
    int seed_h,
    const double* seed_pixels,
    const double* seed_normals,
    double& normalized_x_out,
    double& normalized_y_out
) {
    int ni = nearest_idx % seed_w;
    int nj = nearest_idx / seed_w;
    double nearest_pixel_x = seed_pixels[nearest_idx * 2];
    double nearest_pixel_y = seed_pixels[nearest_idx * 2 + 1];

    // Pick the quad based on which side of the NN the query falls
    int i0 = (pixel_x >= nearest_pixel_x) ? ni : ni - 1;
    int j0 = (pixel_y >= nearest_pixel_y) ? nj : nj - 1;
    // Clamp to valid quad range
    i0 = std::max(0, std::min(seed_w - 2, i0));
    j0 = std::max(0, std::min(seed_h - 2, j0));

    int idx[4] = {
        j0 * seed_w + i0,
        j0 * seed_w + (i0 + 1),
        (j0 + 1) * seed_w + i0,
        (j0 + 1) * seed_w + (i0 + 1),
    };

    // Check all 4 corners are valid
    for (int c = 0; c < 4; c++) {
        if (!std::isfinite(seed_pixels[idx[c] * 2])) {
            return false;
        }
    }

    // Approximate bilinear (u,v) by projecting onto the quad edges.
    const double* p00 = &seed_pixels[idx[0] * 2];
    const double* p10 = &seed_pixels[idx[1] * 2];
    const double* p01 = &seed_pixels[idx[2] * 2];
    double ex = p10[0] - p00[0], ey = p10[1] - p00[1];
    double fx = p01[0] - p00[0], fy = p01[1] - p00[1];
    double qx = pixel_x - p00[0], qy = pixel_y - p00[1];

    double det = ex * fy - fx * ey;
    if (std::abs(det) < 1e-30) {
        return false;
    }
    double inv = 1.0 / det;
    double u = std::max(0.0, std::min(1.0, (qx * fy - fx * qy) * inv));
    double v = std::max(0.0, std::min(1.0, (ex * qy - qx * ey) * inv));

    double mu = 1.0 - u, mv = 1.0 - v;
    const double* n00 = &seed_normals[idx[0] * 2];
    const double* n10 = &seed_normals[idx[1] * 2];
    const double* n01 = &seed_normals[idx[2] * 2];
    const double* n11 = &seed_normals[idx[3] * 2];
    normalized_x_out =
        mu * mv * n00[0] + u * mv * n10[0] + mu * v * n01[0] + u * v * n11[0];
    normalized_y_out =
        mu * mv * n00[1] + u * mv * n10[1] + mu * v * n01[1] + u * v * n11[1];
    return true;
}

// ---------------------------------------------------------------------------
// Newton refinement: generic 2D solver using Ceres Jet autodiff.
// ---------------------------------------------------------------------------

// Forward-project (normalized_x, normalized_y) -> (pixel_x, pixel_y) for the OpenCV model.
template <typename T>
static inline void forward_opencv(
    const T& normalized_x,
    const T& normalized_y,
    const T* intrinsics,  // fx, fy, cx, cy, dist[14]
    T& pixel_x,
    T& pixel_y
) {
    Vec3<T> point(normalized_x, normalized_y, T(1));
    Vec2<T> result;
    project_opencv(intrinsics, point, result);
    pixel_x = result[0];
    pixel_y = result[1];
}

// Forward-project (normalized_x, normalized_y) -> (pixel_x, pixel_y) for the pinhole-splined model.
template <typename T>
static inline void forward_splined(
    const T& normalized_x,
    const T& normalized_y,
    PinholeSplinedConfig* config,
    const T* pinhole_params,
    const T* dx_grid,
    const T* dy_grid,
    T& pixel_x,
    T& pixel_y
) {
    Vec3<T> point(normalized_x, normalized_y, T(1));
    Vec2<T> result;
    project_pinhole_splined(
        config,
        pinhole_params,
        dx_grid,
        dy_grid,
        point,
        result
    );
    pixel_x = result[0];
    pixel_y = result[1];
}

// Newton refinement for OpenCV model, starting from initial guess (normalized_x, normalized_y).
static void refine_opencv(
    double target_u,
    double target_v,
    double& normalized_x,
    double& normalized_y,
    const double* intrinsics  // fx, fy, cx, cy, dist[14]
) {
    using Jet = ceres::Jet<double, 2>;
    constexpr int max_iter = 20;
    constexpr double tol_sq = 1e-14;

    // Convert intrinsics to Jet constants (only normalized_x, normalized_y are variables)
    std::array<Jet, 18> jintrinsics;
    for (int i = 0; i < 18; i++) {
        jintrinsics[i] = Jet(intrinsics[i]);
    }

    for (int iter = 0; iter < max_iter; iter++) {
        Jet jet_normalized_x(normalized_x, 0);
        Jet jet_normalized_y(normalized_y, 1);
        Jet jet_pixel_x, jet_pixel_y;
        forward_opencv(jet_normalized_x, jet_normalized_y, jintrinsics.data(), jet_pixel_x, jet_pixel_y);

        double r0 = jet_pixel_x.a - target_u;
        double r1 = jet_pixel_y.a - target_v;
        if (r0 * r0 + r1 * r1 < tol_sq) {
            break;
        }

        double J00 = jet_pixel_x.v[0], J01 = jet_pixel_x.v[1];
        double J10 = jet_pixel_y.v[0], J11 = jet_pixel_y.v[1];
        double det = J00 * J11 - J01 * J10;
        if (std::abs(det) < 1e-30) {
            break;
        }
        double inv = 1.0 / det;
        normalized_x -= inv * (J11 * r0 - J01 * r1);
        normalized_y -= inv * (-J10 * r0 + J00 * r1);
    }
}

struct SplineConstants {
    int Nx, Ny;
    double half_x, half_y, x_scale, y_scale;
    double fx, fy, cx, cy;

    explicit SplineConstants(
        PinholeSplinedConfig* config,
        const double* pinhole_params
    )
        : Nx((int)config->num_knots_x),
          Ny((int)config->num_knots_y),
          fx(pinhole_params[0]),
          fy(pinhole_params[1]),
          cx(pinhole_params[2]),
          cy(pinhole_params[3]) {
        const double fov_rad_x = config->fov_deg_x * M_PI / 180.0;
        const double fov_rad_y = config->fov_deg_y * M_PI / 180.0;
        half_x = stereo_half_range(fov_rad_x);
        half_y = stereo_half_range(fov_rad_y);
        x_scale = (Nx - 3) / (2.0 * half_x);
        y_scale = (Ny - 3) / (2.0 * half_y);
    }
};

// Newton refinement for pinhole-splined model.
static void refine_splined(
    double target_u,
    double target_v,
    double& normalized_x,
    double& normalized_y,
    const SplineConstants& sc,
    const double* dx_grid,
    const double* dy_grid,
    int* out_rebuilds = nullptr,
    int* out_iters = nullptr
) {
    using Jet = ceres::Jet<double, 2>;
    constexpr int max_newton = 15;
    constexpr int max_rebuilds = 5;
    constexpr double tol_sq = 1e-20;
    constexpr double eps = 1e-12;

    const int Nx = sc.Nx;
    const int Ny = sc.Ny;
    const double half_x = sc.half_x;
    const double half_y = sc.half_y;
    const double x_scale = sc.x_scale;
    const double y_scale = sc.y_scale;

    int rebuild_count = 0, iter_count = 0;
    for (int rebuild = 0; rebuild < max_rebuilds; rebuild++) {
        rebuild_count++;
        double sx, sy;
        normalized_to_stereographic(normalized_x, normalized_y, sx, sy);
        double gx = std::max(
            0.0,
            std::min(1.0 + (sx + half_x) * x_scale, Nx - 1.0 - eps)
        );
        double gy = std::max(
            0.0,
            std::min(1.0 + (sy + half_y) * y_scale, Ny - 1.0 - eps)
        );
        const int ix0 = (int)std::floor(gx);
        const int iy0 = (int)std::floor(gy);

        double local_dx[16], local_dy[16];
        int kidx = 0;
        for (int b = 0; b < 4; b++) {
            const int yy = clamp_int(iy0 + b - 1, 0, Ny - 1);
            for (int a = 0; a < 4; a++) {
                const int xx = clamp_int(ix0 + a - 1, 0, Nx - 1);
                local_dx[kidx] = dx_grid[yy * Nx + xx];
                local_dy[kidx] = dy_grid[yy * Nx + xx];
                kidx++;
            }
        }

        for (int iter = 0; iter < max_newton; iter++) {
            iter_count++;
            Jet jet_normalized_x(normalized_x, 0);
            Jet jet_normalized_y(normalized_y, 1);
            Jet jsx, jsy;
            normalized_to_stereographic(jet_normalized_x, jet_normalized_y, jsx, jsy);
            Jet jgx = clamp_T(
                Jet(1.0) + (jsx + Jet(half_x)) * Jet(x_scale),
                Jet(0.0),
                Jet(Nx - 1.0 - eps)
            );
            Jet jgy = clamp_T(
                Jet(1.0) + (jsy + Jet(half_y)) * Jet(y_scale),
                Jet(0.0),
                Jet(Ny - 1.0 - eps)
            );
            Jet ju = jgx - Jet((double)ix0);
            Jet jv = jgy - Jet((double)iy0);
            Jet wx[4], wy[4];
            cubic_bspline_basis_uniform(ju, wx);
            cubic_bspline_basis_uniform(jv, wy);
            Jet dx_val(0.0), dy_val(0.0);
            int ki = 0;
            for (int b = 0; b < 4; b++) {
                for (int a = 0; a < 4; a++) {
                    Jet w = wy[b] * wx[a];
                    dx_val += Jet(local_dx[ki]) * w;
                    dy_val += Jet(local_dy[ki]) * w;
                    ki++;
                }
            }
            Jet r0 = Jet(sc.fx) * (jet_normalized_x + dx_val) + Jet(sc.cx) - Jet(target_u);
            Jet r1 = Jet(sc.fy) * (jet_normalized_y + dy_val) + Jet(sc.cy) - Jet(target_v);
            double res0 = r0.a, res1 = r1.a;
            if (res0 * res0 + res1 * res1 < tol_sq) {
                break;
            }
            double J00 = r0.v[0], J01 = r0.v[1];
            double J10 = r1.v[0], J11 = r1.v[1];
            double det = J00 * J11 - J01 * J10;
            if (std::abs(det) < 1e-30) {
                break;
            }
            double inv = 1.0 / det;
            normalized_x -= inv * (J11 * res0 - J01 * res1);
            normalized_y -= inv * (-J10 * res0 + J00 * res1);
        }

        normalized_to_stereographic(normalized_x, normalized_y, sx, sy);
        gx = std::max(
            0.0,
            std::min(1.0 + (sx + half_x) * x_scale, Nx - 1.0 - eps)
        );
        gy = std::max(
            0.0,
            std::min(1.0 + (sy + half_y) * y_scale, Ny - 1.0 - eps)
        );
        if ((int)std::floor(gx) == ix0 && (int)std::floor(gy) == iy0) {
            break;
        }
    }
    if (out_rebuilds) {
        *out_rebuilds = rebuild_count;
    }
    if (out_iters) {
        *out_iters = iter_count;
    }
}

// ---------------------------------------------------------------------------
// Shared kernel: validation, NN+IDW initial guess, parallel Newton refinement.
// Per-model parts (intrinsics parsing, residual+jacobian) live in the entry
// points and the refine_* functions above.
// ---------------------------------------------------------------------------

namespace {

void validate_seeded_inputs(
    const py::buffer_info& sp,
    const py::buffer_info& sn,
    const py::buffer_info& qp,
    int seed_w,
    int seed_h
) {
    require(sp.ndim == 2 && sp.shape[1] == 2, "seed_pixels must be (M, 2)");
    require(sn.ndim == 2 && sn.shape[1] == 2, "seed_normals must be (M, 2)");
    require(
        sp.shape[0] == sn.shape[0],
        "seed_pixels and seed_normals must have same length"
    );
    require(
        sp.shape[0] == (ssize_t)(seed_w * seed_h),
        "seed length must equal seed_w * seed_h"
    );
    require(qp.ndim == 2 && qp.shape[1] == 2, "query_pixels must be (N, 2)");
}

template <typename Refiner>
py::array_t<double> run_seeded_normalize(
    const double* SP,
    const double* SN,
    int M,
    int seed_w,
    int seed_h,
    const double* QP,
    ssize_t N,
    double fx,
    double fy,
    double cx,
    double cy,
    Refiner refine_one
) {
    py::array_t<double> out({N, (ssize_t)3});
    double* O = static_cast<double*>(out.request().ptr);

    SeedGrid grid(SP, M, seed_w, seed_h);

    py::gil_scoped_release release;

    // clang-format off
    #pragma omp parallel for schedule(static)
    // clang-format on
    for (ssize_t i = 0; i < N; i++) {
        double pixel_x = QP[i * 2];
        double pixel_y_val = QP[i * 2 + 1];

        double normalized_x = (pixel_x - cx) / fx;
        double normalized_y = (pixel_y_val - cy) / fy;

        int nearest = grid.nearest(pixel_x, pixel_y_val, SP);
        if (nearest >= 0) {
            double interp_normalized_x, interp_normalized_y;
            if (idw_interp_from_nn(
                    pixel_x,
                    pixel_y_val,
                    nearest,
                    seed_w,
                    seed_h,
                    SP,
                    SN,
                    interp_normalized_x,
                    interp_normalized_y
                )) {
                normalized_x = interp_normalized_x;
                normalized_y = interp_normalized_y;
            } else {
                normalized_x = SN[nearest * 2];
                normalized_y = SN[nearest * 2 + 1];
            }
        }

        refine_one(pixel_x, pixel_y_val, normalized_x, normalized_y);

        O[i * 3 + 0] = normalized_x;
        O[i * 3 + 1] = normalized_y;
        O[i * 3 + 2] = 1.0;
    }

    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Python-facing entry points.
// ---------------------------------------------------------------------------

py::array_t<double> seeded_normalize_opencv(
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_normals,
    int seed_w,
    int seed_h,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics
) {
    auto sp = seed_pixels.request();
    auto sn = seed_normals.request();
    auto qp = query_pixels.request();
    auto ip = intrinsics.request();

    validate_seeded_inputs(sp, sn, qp, seed_w, seed_h);
    require(
        ip.ndim == 1 && ip.shape[0] == 18,
        "intrinsics must be (18,): fx, fy, cx, cy, dist[14]"
    );

    const int M = (int)sp.shape[0];
    const ssize_t N = qp.shape[0];
    const double* SP = static_cast<const double*>(sp.ptr);
    const double* SN = static_cast<const double*>(sn.ptr);
    const double* QP = static_cast<const double*>(qp.ptr);
    const double* IP = static_cast<const double*>(ip.ptr);

    return run_seeded_normalize(
        SP,
        SN,
        M,
        seed_w,
        seed_h,
        QP,
        N,
        IP[0],
        IP[1],
        IP[2],
        IP[3],
        [IP](double target_u, double target_v, double& normalized_x, double& normalized_y) {
            refine_opencv(target_u, target_v, normalized_x, normalized_y, IP);
        }
    );
}

py::array_t<double> seeded_normalize_splined(
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_normals,
    int seed_w,
    int seed_h,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_pixels,
    PinholeSplinedConfig& config,
    PinholeSplinedIntrinsicsParameters& params
) {
    auto sp = seed_pixels.request();
    auto sn = seed_normals.request();
    auto qp = query_pixels.request();

    validate_seeded_inputs(sp, sn, qp, seed_w, seed_h);

    auto dxb = params.dx_grid.request();
    auto dyb = params.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == config.num_knots_y &&
            (uint32_t)dxb.shape[1] == config.num_knots_x,
        "dx_grid shape mismatch"
    );
    require(
        (uint32_t)dyb.shape[0] == config.num_knots_y &&
            (uint32_t)dyb.shape[1] == config.num_knots_x,
        "dy_grid shape mismatch"
    );

    auto ppb = params.pinhole_parameters.request();
    require(
        ppb.ndim == 1 && ppb.shape[0] == 4,
        "pinhole_parameters must be (4,)"
    );
    const double* pinhole_params = static_cast<const double*>(ppb.ptr);
    const double fx = pinhole_params[0], fy = pinhole_params[1],
                 cx = pinhole_params[2], cy = pinhole_params[3];
    require(fx != 0.0 && fy != 0.0, "fx/fy must be non-zero");

    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);

    const int M = (int)sp.shape[0];
    const ssize_t N = qp.shape[0];
    const double* SP = static_cast<const double*>(sp.ptr);
    const double* SN = static_cast<const double*>(sn.ptr);
    const double* QP = static_cast<const double*>(qp.ptr);

    SplineConstants sc(&config, pinhole_params);

    return run_seeded_normalize(
        SP,
        SN,
        M,
        seed_w,
        seed_h,
        QP,
        N,
        fx,
        fy,
        cx,
        cy,
        [&sc, dxp, dyp](double target_u, double target_v, double& normalized_x, double& normalized_y) {
            refine_splined(target_u, target_v, normalized_x, normalized_y, sc, dxp, dyp);
        }
    );
}

}  // namespace lensboy
