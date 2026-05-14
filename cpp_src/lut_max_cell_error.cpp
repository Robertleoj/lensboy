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

constexpr int INTERP_NEAREST = 0;
constexpr int INTERP_BILINEAR = 1;
constexpr int INTERP_BICUBIC = 2;

// Penalty multiplier on cell-membership violation. Chosen large enough that
// any boundary crossing dominates any interior gain in the objective, but
// finite so that gradient ascent still makes monotonic progress.
constexpr double PENALTY_LAMBDA = 1.0e30;

constexpr int MAX_LINE_SEARCH_STEPS = 30;
constexpr double STEP_GROWTH = 1.4;
constexpr double STEP_SHRINK = 0.5;
constexpr double MIN_STEP_RATIO = 1.0e-12;

// Approximate pixel spacing between gradient-ascent seeds inside a cell.
// One seed per ~32 pixel_x of cell extent in each direction, with a one-seed
// floor for tiny cells. Distortion functions are smooth, so the objective
// is single-modal in every reasonable cell — coarse seeding suffices.
constexpr double SEED_STRIDE_PIXELS = 32.0;

using Jet2 = ceres::Jet<double, 2>;

template <typename T>
inline T relu(
    const T& x
) {
    return x > T(0.0) ? x : T(0.0);
}

template <typename T>
inline T penalty_outside_cell(
    const T& pixel_x,
    const T& pixel_y,
    double cell_x0,
    double cell_x1,
    double cell_y0,
    double cell_y1
) {
    return relu(T(cell_x0) - pixel_x) + relu(pixel_x - T(cell_x1)) +
           relu(T(cell_y0) - pixel_y) + relu(pixel_y - T(cell_y1));
}

// Catmull-Rom weights matching the Python implementation.
template <typename T>
inline void catmull_rom_weights_T(
    const T& fractional,
    T weights[4]
) {
    const T fractional_squared = fractional * fractional;
    const T fractional_cubed = fractional_squared * fractional;
    weights[0] =
        T(-0.5) * fractional + fractional_squared + T(-0.5) * fractional_cubed;
    weights[1] =
        T(1.0) + T(-2.5) * fractional_squared + T(1.5) * fractional_cubed;
    weights[2] = T(0.5) * fractional + T(2.0) * fractional_squared +
                 T(-1.5) * fractional_cubed;
    weights[3] = T(-0.5) * fractional_squared + T(0.5) * fractional_cubed;
}

template <typename T>
inline void interp_lut_nearest(
    const double* lut_xy_grid,
    int grid_width,
    int grid_height,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& pixel_x,
    const T& pixel_y,
    T& out_x,
    T& out_y
) {
    const double pixel_x_scalar = scalar_value(pixel_x);
    const double pixel_y_scalar = scalar_value(pixel_y);
    const double grid_x = (pixel_x_scalar - grid_x_min) * grid_scale_x;
    const double grid_y = (pixel_y_scalar - grid_y_min) * grid_scale_y;
    // Match numpy's np.rint (round-half-to-even) so the C++ optimiser and
    // the Python LUT query agree at exact half-integer grid coordinates.
    // std::lround rounds half-away-from-zero and disagreed at .5 values.
    int nearest_index_x = static_cast<int>(std::nearbyint(grid_x));
    int nearest_index_y = static_cast<int>(std::nearbyint(grid_y));
    nearest_index_x = std::clamp(nearest_index_x, 0, grid_width - 1);
    nearest_index_y = std::clamp(nearest_index_y, 0, grid_height - 1);
    const double* node =
        lut_xy_grid + (nearest_index_y * grid_width + nearest_index_x) * 2;
    out_x = T(node[0]);
    out_y = T(node[1]);
}

template <typename T>
inline void interp_lut_bilinear(
    const double* lut_xy_grid,
    int grid_width,
    int grid_height,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& pixel_x,
    const T& pixel_y,
    T& out_x,
    T& out_y
) {
    const T grid_x = (pixel_x - T(grid_x_min)) * T(grid_scale_x);
    const T grid_y = (pixel_y - T(grid_y_min)) * T(grid_scale_y);

    const T grid_x_clamped =
        clamp_T(grid_x, T(0.0), T(static_cast<double>(grid_width - 1)));
    int corner_index_x0 =
        static_cast<int>(std::floor(scalar_value(grid_x_clamped)));
    corner_index_x0 = std::clamp(corner_index_x0, 0, grid_width - 2);
    const T fractional_x =
        grid_x_clamped - T(static_cast<double>(corner_index_x0));

    const T grid_y_clamped =
        clamp_T(grid_y, T(0.0), T(static_cast<double>(grid_height - 1)));
    int corner_index_y0 =
        static_cast<int>(std::floor(scalar_value(grid_y_clamped)));
    corner_index_y0 = std::clamp(corner_index_y0, 0, grid_height - 2);
    const T fractional_y =
        grid_y_clamped - T(static_cast<double>(corner_index_y0));

    const int corner_index_x1 = corner_index_x0 + 1;
    const int corner_index_y1 = corner_index_y0 + 1;

    const double* corner_x0_y0 =
        lut_xy_grid + (corner_index_y0 * grid_width + corner_index_x0) * 2;
    const double* corner_x1_y0 =
        lut_xy_grid + (corner_index_y0 * grid_width + corner_index_x1) * 2;
    const double* corner_x0_y1 =
        lut_xy_grid + (corner_index_y1 * grid_width + corner_index_x0) * 2;
    const double* corner_x1_y1 =
        lut_xy_grid + (corner_index_y1 * grid_width + corner_index_x1) * 2;

    const T one_minus_fractional_x = T(1.0) - fractional_x;
    const T one_minus_fractional_y = T(1.0) - fractional_y;
    const T top_x = one_minus_fractional_x * corner_x0_y0[0] +
                    fractional_x * corner_x1_y0[0];
    const T top_y = one_minus_fractional_x * corner_x0_y0[1] +
                    fractional_x * corner_x1_y0[1];
    const T bottom_x = one_minus_fractional_x * corner_x0_y1[0] +
                       fractional_x * corner_x1_y1[0];
    const T bottom_y = one_minus_fractional_x * corner_x0_y1[1] +
                       fractional_x * corner_x1_y1[1];
    out_x = one_minus_fractional_y * top_x + fractional_y * bottom_x;
    out_y = one_minus_fractional_y * top_y + fractional_y * bottom_y;
}

template <typename T>
inline void interp_lut_bicubic(
    const double* lut_xy_grid,
    int grid_width,
    int grid_height,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& pixel_x,
    const T& pixel_y,
    T& out_x,
    T& out_y
) {
    if (grid_width < 4 || grid_height < 4) {
        interp_lut_bilinear(
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            pixel_x,
            pixel_y,
            out_x,
            out_y
        );
        return;
    }

    const T grid_x = (pixel_x - T(grid_x_min)) * T(grid_scale_x);
    const T grid_y = (pixel_y - T(grid_y_min)) * T(grid_scale_y);
    const T grid_x_clamped =
        clamp_T(grid_x, T(0.0), T(static_cast<double>(grid_width - 1)));
    const T grid_y_clamped =
        clamp_T(grid_y, T(0.0), T(static_cast<double>(grid_height - 1)));

    const int base_index_x =
        static_cast<int>(std::floor(scalar_value(grid_x_clamped)));
    const int base_index_y =
        static_cast<int>(std::floor(scalar_value(grid_y_clamped)));

    const bool has_full_support =
        (base_index_x >= 1) && (base_index_x <= grid_width - 3) &&
        (base_index_y >= 1) && (base_index_y <= grid_height - 3);
    if (!has_full_support) {
        interp_lut_bilinear(
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            pixel_x,
            pixel_y,
            out_x,
            out_y
        );
        return;
    }

    const T fractional_x =
        grid_x_clamped - T(static_cast<double>(base_index_x));
    const T fractional_y =
        grid_y_clamped - T(static_cast<double>(base_index_y));

    T weights_x[4], weights_y[4];
    catmull_rom_weights_T(fractional_x, weights_x);
    catmull_rom_weights_T(fractional_y, weights_y);

    out_x = T(0.0);
    out_y = T(0.0);
    for (int row = 0; row < 4; ++row) {
        const int sample_y = base_index_y + row - 1;
        for (int col = 0; col < 4; ++col) {
            const int sample_x = base_index_x + col - 1;
            const double* node =
                lut_xy_grid + (sample_y * grid_width + sample_x) * 2;
            const T weight = weights_y[row] * weights_x[col];
            out_x = out_x + weight * node[0];
            out_y = out_y + weight * node[1];
        }
    }
}

template <typename T>
inline void interp_lut_dispatch(
    int interpolation_mode,
    const double* lut_xy_grid,
    int grid_width,
    int grid_height,
    double grid_x_min,
    double grid_y_min,
    double grid_scale_x,
    double grid_scale_y,
    const T& pixel_x,
    const T& pixel_y,
    T& out_x,
    T& out_y
) {
    if (interpolation_mode == INTERP_NEAREST) {
        interp_lut_nearest(
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            pixel_x,
            pixel_y,
            out_x,
            out_y
        );
    } else if (interpolation_mode == INTERP_BILINEAR) {
        interp_lut_bilinear(
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            pixel_x,
            pixel_y,
            out_x,
            out_y
        );
    } else {
        interp_lut_bicubic(
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            pixel_x,
            pixel_y,
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
    int grid_width,
    int grid_height,
    const T& x_spline,
    const T& y_spline
) {
    constexpr double epsilon = 1e-12;
    T spline_x_clamped = clamp_T(
        x_spline,
        T(0.0),
        T(static_cast<double>(grid_width - 1) - epsilon)
    );
    T spline_y_clamped = clamp_T(
        y_spline,
        T(0.0),
        T(static_cast<double>(grid_height - 1) - epsilon)
    );

    const int cell_index_x =
        static_cast<int>(std::floor(scalar_value(spline_x_clamped)));
    const int cell_index_y =
        static_cast<int>(std::floor(scalar_value(spline_y_clamped)));

    const T fractional_x =
        spline_x_clamped - T(static_cast<double>(cell_index_x));
    const T fractional_y =
        spline_y_clamped - T(static_cast<double>(cell_index_y));

    T weights_x[4], weights_y[4];
    cubic_bspline_basis_uniform(fractional_x, weights_x);
    cubic_bspline_basis_uniform(fractional_y, weights_y);

    const int clamped_x_indices[4] = {
        clamp_int(cell_index_x - 1, 0, grid_width - 1),
        clamp_int(cell_index_x + 0, 0, grid_width - 1),
        clamp_int(cell_index_x + 1, 0, grid_width - 1),
        clamp_int(cell_index_x + 2, 0, grid_width - 1)
    };
    const int clamped_y_indices[4] = {
        clamp_int(cell_index_y - 1, 0, grid_height - 1),
        clamp_int(cell_index_y + 0, 0, grid_height - 1),
        clamp_int(cell_index_y + 1, 0, grid_height - 1),
        clamp_int(cell_index_y + 2, 0, grid_height - 1)
    };

    T accumulator(0.0);
    for (int row = 0; row < 4; ++row) {
        const int row_offset = clamped_y_indices[row] * grid_width;
        const T weight_y = weights_y[row];
        for (int col = 0; col < 4; ++col) {
            accumulator =
                accumulator + (weight_y * weights_x[col]) *
                                  grid[row_offset + clamped_x_indices[col]];
        }
    }
    return accumulator;
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
    const double cross_x = exact_y - approx_y;
    const double cross_y = approx_x - exact_x;
    const double cross_z = exact_x * approx_y - exact_y * approx_x;
    const double cross_norm =
        std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
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
//   project(normalized_x, normalized_y, pixel_x, pixel_y)        for T=double
//   project(jet_normalized_x, jet_normalized_y, ...)   for T=Jet2
//
// Returns (max_angular_error_deg, peak_pixel_x, peak_pixel_y).
template <typename Project>
struct CellMaximizer {
    Project project;
    int interpolation_mode;
    const double* lut_xy_grid;
    int grid_width;
    int grid_height;
    double grid_x_min;
    double grid_y_min;
    double grid_scale_x;
    double grid_scale_y;
    int max_iterations;
    double gradient_tolerance;

    template <typename T>
    inline T eval(
        const T& normalized_x,
        const T& normalized_y,
        double cell_x0,
        double cell_x1,
        double cell_y0,
        double cell_y1
    ) const {
        T pixel_x, pixel_y;
        project(normalized_x, normalized_y, pixel_x, pixel_y);
        T approx_x, approx_y;
        interp_lut_dispatch(
            interpolation_mode,
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            pixel_x,
            pixel_y,
            approx_x,
            approx_y
        );
        // sin²(angle) between rays (normalized_x, normalized_y, 1) and
        // (approx_x, approx_y, 1):
        //   sin²(θ) = ‖cross‖² / (‖r1‖² · ‖r2‖²)
        // cross((normalized_x, normalized_y, 1), (approx_x, approx_y, 1)) =
        //   (normalized_y - approx_y, approx_x - normalized_x,
        //   normalized_x*approx_y - normalized_y*approx_x).
        const T cross_x = normalized_y - approx_y;
        const T cross_y = approx_x - normalized_x;
        const T cross_z = normalized_x * approx_y - normalized_y * approx_x;
        const T cross_norm_squared =
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
        const T normalized_norm_squared =
            normalized_x * normalized_x + normalized_y * normalized_y + T(1.0);
        const T approx_norm_squared =
            approx_x * approx_x + approx_y * approx_y + T(1.0);
        const T sin_squared = cross_norm_squared /
                              (normalized_norm_squared * approx_norm_squared);
        const T penalty = penalty_outside_cell(
            pixel_x,
            pixel_y,
            cell_x0,
            cell_x1,
            cell_y0,
            cell_y1
        );
        return sin_squared - T(PENALTY_LAMBDA) * penalty;
    }

    // Gradient ascent from a single initial point in normalised space.
    // Returns the final (normalized_x, normalized_y) and objective value via
    // out parameters. No bbox clamp on the search — the corner-bbox is an
    // approximation of the cell's image in normalised space and can be too
    // tight for highly nonlinear models. The ReLU penalty inside `eval`
    // enforces the true constraint (project(n) ∈ pixel cell).
    void optimize_from(
        double cell_x0,
        double cell_x1,
        double cell_y0,
        double cell_y1,
        double normal_span,
        double init_normalized_x,
        double init_normalized_y,
        double& out_normalized_x,
        double& out_normalized_y,
        double& out_objective
    ) const {
        double normalized_x = init_normalized_x;
        double normalized_y = init_normalized_y;
        double step_size = std::max(normal_span * 0.25, 1.0e-12);
        const double min_step = std::max(normal_span * MIN_STEP_RATIO, 1.0e-30);

        double current_objective = eval<double>(
            normalized_x,
            normalized_y,
            cell_x0,
            cell_x1,
            cell_y0,
            cell_y1
        );

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            Jet2 jet_normalized_x(normalized_x, 0);
            Jet2 jet_normalized_y(normalized_y, 1);
            const Jet2 objective_jet = eval<Jet2>(
                jet_normalized_x,
                jet_normalized_y,
                cell_x0,
                cell_x1,
                cell_y0,
                cell_y1
            );

            const double gradient_x = objective_jet.v[0];
            const double gradient_y = objective_jet.v[1];
            const double gradient_norm =
                std::sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
            if (gradient_norm < gradient_tolerance) {
                current_objective = objective_jet.a;
                break;
            }

            const double inverse_gradient_norm = 1.0 / gradient_norm;
            const double direction_x = gradient_x * inverse_gradient_norm;
            const double direction_y = gradient_y * inverse_gradient_norm;

            double step = step_size;
            bool accepted = false;
            for (int line_search_step = 0;
                 line_search_step < MAX_LINE_SEARCH_STEPS;
                 ++line_search_step) {
                const double trial_normalized_x =
                    normalized_x + step * direction_x;
                const double trial_normalized_y =
                    normalized_y + step * direction_y;
                const double trial_objective = eval<double>(
                    trial_normalized_x,
                    trial_normalized_y,
                    cell_x0,
                    cell_x1,
                    cell_y0,
                    cell_y1
                );
                if (trial_objective > current_objective) {
                    normalized_x = trial_normalized_x;
                    normalized_y = trial_normalized_y;
                    current_objective = trial_objective;
                    step_size = step * STEP_GROWTH;
                    accepted = true;
                    break;
                }
                step *= STEP_SHRINK;
                if (step < min_step) {
                    break;
                }
            }
            if (!accepted) {
                break;
            }
        }

        out_normalized_x = normalized_x;
        out_normalized_y = normalized_y;
        out_objective = current_objective;
    }

    // Multistart wrapper. Lays down a seed grid inside the cell at roughly
    // one seed per `SEED_STRIDE_PIXELS` of cell extent (in pixel space),
    // bilinearly interpolates the 4 corner normals to get an initial
    // normalised coordinate per seed, runs gradient ascent from each, and
    // keeps the best. Defensive against multimodal cells where a single
    // warm start would land in a sub-optimal basin.
    void run(
        double cell_x0,
        double cell_x1,
        double cell_y0,
        double cell_y1,
        const double* normal_x0_y0,
        const double* normal_x1_y0,
        const double* normal_x0_y1,
        const double* normal_x1_y1,
        double& out_peak_pixel_x,
        double& out_peak_pixel_y,
        double& out_exact_x,
        double& out_exact_y,
        double& out_approx_x,
        double& out_approx_y
    ) const {
        const double normal_low_x = std::min(
            {normal_x0_y0[0], normal_x1_y0[0], normal_x0_y1[0], normal_x1_y1[0]}
        );
        const double normal_high_x = std::max(
            {normal_x0_y0[0], normal_x1_y0[0], normal_x0_y1[0], normal_x1_y1[0]}
        );
        const double normal_low_y = std::min(
            {normal_x0_y0[1], normal_x1_y0[1], normal_x0_y1[1], normal_x1_y1[1]}
        );
        const double normal_high_y = std::max(
            {normal_x0_y0[1], normal_x1_y0[1], normal_x0_y1[1], normal_x1_y1[1]}
        );
        const double normal_span_full = std::max(
            normal_high_x - normal_low_x,
            normal_high_y - normal_low_y
        );

        const double cell_width = cell_x1 - cell_x0;
        const double cell_height = cell_y1 - cell_y0;
        const int num_seeds_x = std::max(
            1,
            static_cast<int>(std::ceil(cell_width / SEED_STRIDE_PIXELS))
        );
        const int num_seeds_y = std::max(
            1,
            static_cast<int>(std::ceil(cell_height / SEED_STRIDE_PIXELS))
        );

        // Scale the gradient-ascent step to the seed sub-cell so an
        // optimisation from one seed doesn't immediately overshoot into a
        // neighbouring seed's basin. Line search adapts up if the basin is
        // wider.
        const int max_seeds_per_dim = std::max(num_seeds_x, num_seeds_y);
        const double per_seed_normal_span =
            normal_span_full / max_seeds_per_dim;

        double best_normalized_x = 0.5 * (normal_low_x + normal_high_x);
        double best_normalized_y = 0.5 * (normal_low_y + normal_high_y);
        double best_objective = -std::numeric_limits<double>::infinity();
        for (int seed_row = 0; seed_row < num_seeds_y; ++seed_row) {
            const double seed_fraction_y =
                (static_cast<double>(seed_row) + 0.5) /
                static_cast<double>(num_seeds_y);
            for (int seed_col = 0; seed_col < num_seeds_x; ++seed_col) {
                const double seed_fraction_x =
                    (static_cast<double>(seed_col) + 0.5) /
                    static_cast<double>(num_seeds_x);
                const double weight_x0_y0 =
                    (1.0 - seed_fraction_x) * (1.0 - seed_fraction_y);
                const double weight_x1_y0 =
                    seed_fraction_x * (1.0 - seed_fraction_y);
                const double weight_x0_y1 =
                    (1.0 - seed_fraction_x) * seed_fraction_y;
                const double weight_x1_y1 = seed_fraction_x * seed_fraction_y;
                const double init_normalized_x =
                    weight_x0_y0 * normal_x0_y0[0] +
                    weight_x1_y0 * normal_x1_y0[0] +
                    weight_x0_y1 * normal_x0_y1[0] +
                    weight_x1_y1 * normal_x1_y1[0];
                const double init_normalized_y =
                    weight_x0_y0 * normal_x0_y0[1] +
                    weight_x1_y0 * normal_x1_y0[1] +
                    weight_x0_y1 * normal_x0_y1[1] +
                    weight_x1_y1 * normal_x1_y1[1];

                double trial_normalized_x = 0.0;
                double trial_normalized_y = 0.0;
                double trial_objective = 0.0;
                optimize_from(
                    cell_x0,
                    cell_x1,
                    cell_y0,
                    cell_y1,
                    per_seed_normal_span,
                    init_normalized_x,
                    init_normalized_y,
                    trial_normalized_x,
                    trial_normalized_y,
                    trial_objective
                );
                if (trial_objective > best_objective) {
                    best_objective = trial_objective;
                    best_normalized_x = trial_normalized_x;
                    best_normalized_y = trial_normalized_y;
                }
            }
        }

        double peak_pixel_x, peak_pixel_y;
        project(
            best_normalized_x,
            best_normalized_y,
            peak_pixel_x,
            peak_pixel_y
        );
        double approx_x, approx_y;
        interp_lut_dispatch<double>(
            interpolation_mode,
            lut_xy_grid,
            grid_width,
            grid_height,
            grid_x_min,
            grid_y_min,
            grid_scale_x,
            grid_scale_y,
            peak_pixel_x,
            peak_pixel_y,
            approx_x,
            approx_y
        );
        out_peak_pixel_x = peak_pixel_x;
        out_peak_pixel_y = peak_pixel_y;
        out_exact_x = best_normalized_x;
        out_exact_y = best_normalized_y;
        out_approx_x = approx_x;
        out_approx_y = approx_y;
    }
};

template <typename Project>
py::array_t<double> run_max_cell_errors(
    const Project& project,
    const double* lut_xy_grid,
    int grid_width,
    int grid_height,
    double grid_x_min,
    double grid_x_max,
    double grid_y_min,
    double grid_y_max,
    int interpolation_mode,
    int max_iterations,
    double gradient_tolerance
) {
    require(
        grid_width >= 2 && grid_height >= 2,
        "LUT grid dimensions must be at least 2"
    );
    require(
        interpolation_mode >= 0 && interpolation_mode <= 2,
        "interpolation_mode must be 0 (nearest), 1 (bilinear), or 2 (bicubic)"
    );
    require(max_iterations > 0, "max_iterations must be positive");

    const double grid_scale_x =
        static_cast<double>(grid_width - 1) / (grid_x_max - grid_x_min);
    const double grid_scale_y =
        static_cast<double>(grid_height - 1) / (grid_y_max - grid_y_min);

    const int num_cells_x = grid_width - 1;
    const int num_cells_y = grid_height - 1;
    const ssize_t num_cells =
        static_cast<ssize_t>(num_cells_x) * static_cast<ssize_t>(num_cells_y);

    py::array_t<double> out({num_cells, static_cast<ssize_t>(6)});
    auto output_buffer = out.request();
    double* output_data = static_cast<double*>(output_buffer.ptr);

    CellMaximizer<Project> maximizer{
        project,
        interpolation_mode,
        lut_xy_grid,
        grid_width,
        grid_height,
        grid_x_min,
        grid_y_min,
        grid_scale_x,
        grid_scale_y,
        max_iterations,
        gradient_tolerance
    };

    const double pixel_span_x = 1.0 / grid_scale_x;
    const double pixel_span_y = 1.0 / grid_scale_y;

    // clang-format off
    #pragma omp parallel for schedule(static) collapse(2)
    // clang-format on
    for (int cell_index_y = 0; cell_index_y < num_cells_y; ++cell_index_y) {
        for (int cell_index_x = 0; cell_index_x < num_cells_x; ++cell_index_x) {
            const ssize_t cell_index =
                static_cast<ssize_t>(cell_index_y) * num_cells_x + cell_index_x;

            const double cell_x0 = grid_x_min + cell_index_x * pixel_span_x;
            const double cell_x1 =
                grid_x_min + (cell_index_x + 1) * pixel_span_x;
            const double cell_y0 = grid_y_min + cell_index_y * pixel_span_y;
            const double cell_y1 =
                grid_y_min + (cell_index_y + 1) * pixel_span_y;

            const double* normal_x0_y0 =
                lut_xy_grid + (cell_index_y * grid_width + cell_index_x) * 2;
            const double* normal_x1_y0 =
                lut_xy_grid +
                (cell_index_y * grid_width + (cell_index_x + 1)) * 2;
            const double* normal_x0_y1 =
                lut_xy_grid +
                ((cell_index_y + 1) * grid_width + cell_index_x) * 2;
            const double* normal_x1_y1 =
                lut_xy_grid +
                ((cell_index_y + 1) * grid_width + (cell_index_x + 1)) * 2;

            double peak_pixel_x = 0.5 * (cell_x0 + cell_x1);
            double peak_pixel_y = 0.5 * (cell_y0 + cell_y1);
            double exact_x = 0.0;
            double exact_y = 0.0;
            double approx_x = 0.0;
            double approx_y = 0.0;
            maximizer.run(
                cell_x0,
                cell_x1,
                cell_y0,
                cell_y1,
                normal_x0_y0,
                normal_x1_y0,
                normal_x0_y1,
                normal_x1_y1,
                peak_pixel_x,
                peak_pixel_y,
                exact_x,
                exact_y,
                approx_x,
                approx_y
            );

            double* row = output_data + cell_index * 6;
            row[0] = peak_pixel_x;
            row[1] = peak_pixel_y;
            row[2] = exact_x;
            row[3] = exact_y;
            row[4] = approx_x;
            row[5] = approx_y;
        }
    }

    return out;
}

}  // namespace

// Project a normalized point (normalized_x, normalized_y, 1) through
// PinholeSplined with double constants. Templated on T for autodiff over the
// input point.
template <typename T>
inline void project_pinhole_splined_n(
    const SplineMap& map,
    const double* pinhole_params,  // fx, fy, cx, cy
    const double* dx_grid,
    const double* dy_grid,
    const T& normalized_x,
    const T& normalized_y,
    T& pixel_x,
    T& pixel_y
) {
    T x_spline, y_spline;
    map.normalized_to_grid_coords(
        normalized_x,
        normalized_y,
        x_spline,
        y_spline
    );

    const T distortion_x =
        eval_bspline2d_double_grid(dx_grid, map.Nx, map.Ny, x_spline, y_spline);
    const T distortion_y =
        eval_bspline2d_double_grid(dy_grid, map.Nx, map.Ny, x_spline, y_spline);

    pixel_x = T(pinhole_params[0]) * (normalized_x + distortion_x) +
              T(pinhole_params[2]);
    pixel_y = T(pinhole_params[1]) * (normalized_y + distortion_y) +
              T(pinhole_params[3]);
}

py::array_t<double> max_cell_errors_pinhole_splined(
    PinholeSplinedConfig& config,
    PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> lut_xy_grid,
    double grid_x_min,
    double grid_x_max,
    double grid_y_min,
    double grid_y_max,
    int interpolation_mode,
    int max_iterations,
    double gradient_tolerance
) {
    auto pinhole_buffer = intrinsics.pinhole_parameters.request();
    require(
        pinhole_buffer.ndim == 1 && pinhole_buffer.shape[0] == 4,
        "pinhole_parameters must have shape (4,)"
    );
    auto dx_grid_buffer = intrinsics.dx_grid.request();
    auto dy_grid_buffer = intrinsics.dy_grid.request();
    require(
        (uint32_t)dx_grid_buffer.shape[0] == config.num_knots_y &&
            (uint32_t)dx_grid_buffer.shape[1] == config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dy_grid_buffer.shape[0] == config.num_knots_y &&
            (uint32_t)dy_grid_buffer.shape[1] == config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    auto lut_buffer = lut_xy_grid.request();
    require(
        lut_buffer.ndim == 3 && lut_buffer.shape[2] == 2,
        "lut_xy_grid must have shape (grid_height, grid_width, 2)"
    );
    const int grid_height = static_cast<int>(lut_buffer.shape[0]);
    const int grid_width = static_cast<int>(lut_buffer.shape[1]);

    const double* pinhole_params =
        static_cast<const double*>(pinhole_buffer.ptr);
    const double* dx_grid_data = static_cast<const double*>(dx_grid_buffer.ptr);
    const double* dy_grid_data = static_cast<const double*>(dy_grid_buffer.ptr);
    const double* lut_data = static_cast<const double*>(lut_buffer.ptr);

    const SplineMap map(config);

    auto project = [&](auto&& normalized_x,
                       auto&& normalized_y,
                       auto& pixel_x,
                       auto& pixel_y) {
        project_pinhole_splined_n(
            map,
            pinhole_params,
            dx_grid_data,
            dy_grid_data,
            normalized_x,
            normalized_y,
            pixel_x,
            pixel_y
        );
    };

    return run_max_cell_errors(
        project,
        lut_data,
        grid_width,
        grid_height,
        grid_x_min,
        grid_x_max,
        grid_y_min,
        grid_y_max,
        interpolation_mode,
        max_iterations,
        gradient_tolerance
    );
}

// Project a normalized point through OpenCV with double constants. Wraps the
// 18 intrinsics into a stack-local T buffer once and reuses the existing
// templated project_opencv to avoid re-implementing the distortion math.
template <typename T>
inline void project_opencv_n(
    const double* intrinsics,
    const T& normalized_x,
    const T& normalized_y,
    T& pixel_x,
    T& pixel_y
) {
    T wrapped_intrinsics[18];
    for (int i = 0; i < 18; ++i) {
        wrapped_intrinsics[i] = T(intrinsics[i]);
    }
    Vec3<T> point_in_camera(normalized_x, normalized_y, T(1.0));
    Vec2<T> result;
    project_opencv<T>(wrapped_intrinsics, point_in_camera, result);
    pixel_x = result[0];
    pixel_y = result[1];
}

py::array_t<double> max_cell_errors_opencv(
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> lut_xy_grid,
    double grid_x_min,
    double grid_x_max,
    double grid_y_min,
    double grid_y_max,
    int interpolation_mode,
    int max_iterations,
    double gradient_tolerance
) {
    auto intrinsics_buffer = intrinsics.request();
    require(
        intrinsics_buffer.ndim == 1 && intrinsics_buffer.shape[0] == 18,
        "intrinsics must have shape (18,) — fx, fy, cx, cy + 14 distortion"
    );

    auto lut_buffer = lut_xy_grid.request();
    require(
        lut_buffer.ndim == 3 && lut_buffer.shape[2] == 2,
        "lut_xy_grid must have shape (grid_height, grid_width, 2)"
    );
    const int grid_height = static_cast<int>(lut_buffer.shape[0]);
    const int grid_width = static_cast<int>(lut_buffer.shape[1]);

    const double* intrinsics_data =
        static_cast<const double*>(intrinsics_buffer.ptr);
    const double* lut_data = static_cast<const double*>(lut_buffer.ptr);

    auto project = [&](auto&& normalized_x,
                       auto&& normalized_y,
                       auto& pixel_x,
                       auto& pixel_y) {
        project_opencv_n(
            intrinsics_data,
            normalized_x,
            normalized_y,
            pixel_x,
            pixel_y
        );
    };

    return run_max_cell_errors(
        project,
        lut_data,
        grid_width,
        grid_height,
        grid_x_min,
        grid_x_max,
        grid_y_min,
        grid_y_max,
        interpolation_mode,
        max_iterations,
        gradient_tolerance
    );
}

}  // namespace lensboy
