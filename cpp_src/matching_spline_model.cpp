#include <ceres/ceres.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <vector>
#include "./cameramodels.hpp"
#include "./utils.hpp"

namespace lensboy {

struct DistortionError {
    DistortionError(
        const double opencv_distorted_x,
        const double opencv_distorted_y,
        const double x_normalized,
        const double y_normalized,
        const double spline_u,
        const double spline_v
    )
        : opencv_distorted_x(opencv_distorted_x),
          opencv_distorted_y(opencv_distorted_y),
          x_normalized(x_normalized),
          y_normalized(y_normalized),
          spline_u(spline_u),
          spline_v(spline_v) {}

    template <typename T>
    bool operator()(
        // dx control points: y-major naming + ordering (y1 row, then y2, ...)
        const T* const dx_y1x1,
        const T* const dx_y1x2,
        const T* const dx_y1x3,
        const T* const dx_y1x4,
        const T* const dx_y2x1,
        const T* const dx_y2x2,
        const T* const dx_y2x3,
        const T* const dx_y2x4,
        const T* const dx_y3x1,
        const T* const dx_y3x2,
        const T* const dx_y3x3,
        const T* const dx_y3x4,
        const T* const dx_y4x1,
        const T* const dx_y4x2,
        const T* const dx_y4x3,
        const T* const dx_y4x4,

        // dy control points: y-major naming + ordering
        const T* const dy_y1x1,
        const T* const dy_y1x2,
        const T* const dy_y1x3,
        const T* const dy_y1x4,
        const T* const dy_y2x1,
        const T* const dy_y2x2,
        const T* const dy_y2x3,
        const T* const dy_y2x4,
        const T* const dy_y3x1,
        const T* const dy_y3x2,
        const T* const dy_y3x3,
        const T* const dy_y3x4,
        const T* const dy_y4x1,
        const T* const dy_y4x2,
        const T* const dy_y4x3,
        const T* const dy_y4x4,

        T* residuals
    ) const {
        T wx[4], wy[4];
        cubic_bspline_basis_uniform(T(spline_u), wx);
        cubic_bspline_basis_uniform(T(spline_v), wy);

        // local grids are indexed [y][x]
        T grid_x[4][4] = {
            {dx_y1x1[0], dx_y1x2[0], dx_y1x3[0], dx_y1x4[0]},
            {dx_y2x1[0], dx_y2x2[0], dx_y2x3[0], dx_y2x4[0]},
            {dx_y3x1[0], dx_y3x2[0], dx_y3x3[0], dx_y3x4[0]},
            {dx_y4x1[0], dx_y4x2[0], dx_y4x3[0], dx_y4x4[0]}
        };

        T grid_y[4][4] = {
            {dy_y1x1[0], dy_y1x2[0], dy_y1x3[0], dy_y1x4[0]},
            {dy_y2x1[0], dy_y2x2[0], dy_y2x3[0], dy_y2x4[0]},
            {dy_y3x1[0], dy_y3x2[0], dy_y3x3[0], dy_y3x4[0]},
            {dy_y4x1[0], dy_y4x2[0], dy_y4x3[0], dy_y4x4[0]}
        };

        T dx = T(0);
        T dy = T(0);

        // y-major accumulation: b = y, a = x
        for (int b = 0; b < 4; ++b) {
            const T wyb = wy[b];
            for (int a = 0; a < 4; ++a) {
                const T w = wyb * wx[a];
                dx += grid_x[b][a] * w;
                dy += grid_y[b][a] * w;
            }
        }

        const T spline_distorted_x = T(x_normalized) + dx;
        const T spline_distorted_y = T(y_normalized) + dy;

        residuals[0] = spline_distorted_x - T(opencv_distorted_x);
        residuals[1] = spline_distorted_y - T(opencv_distorted_y);
        return true;
    }

    static ceres::CostFunction* Create(
        const double opencv_distorted_x,
        const double opencv_distorted_y,
        const double x_normalized,
        const double y_normalized,
        const double spline_u,
        const double spline_v
    ) {
        // clang-format off
        using Cost = ceres::AutoDiffCostFunction<
            DistortionError,
            2,  // residuals
            1,1,1,1, 1,1,1,1,
            1,1,1,1, 1,1,1,1,
            1,1,1,1, 1,1,1,1,
            1,1,1,1, 1,1,1,1
        >;
        // clang-format on

        return new Cost(new DistortionError(
            opencv_distorted_x,
            opencv_distorted_y,
            x_normalized,
            y_normalized,
            spline_u,
            spline_v
        ));
    }

    double spline_u;
    double spline_v;
    double x_normalized;
    double y_normalized;
    double opencv_distorted_x;
    double opencv_distorted_y;
};

struct StereographicOpenCVError {
    StereographicOpenCVError(
        const double stereographic_x,
        const double stereographic_y,
        const double target_pixel_x,
        const double target_pixel_y
    )
        : stereographic_x(stereographic_x),
          stereographic_y(stereographic_y),
          target_pixel_x(target_pixel_x),
          target_pixel_y(target_pixel_y) {}

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        T* residuals
    ) const {
        const T sx = T(stereographic_x);
        const T sy = T(stereographic_y);
        const T r_s = sqrt(sx * sx + sy * sy + T(1e-30));
        const T theta = T(2) * atan(r_s / T(2));
        const T r_n = tan(theta);
        const T scale = r_n / r_s;

        const Vec3<T> point_in_camera(sx * scale, sy * scale, T(1));
        Vec2<T> image_point;
        project_opencv(intrinsics, point_in_camera, image_point);

        residuals[0] = image_point[0] - T(target_pixel_x);
        residuals[1] = image_point[1] - T(target_pixel_y);
        return true;
    }

    static ceres::CostFunction* Create(
        const double stereographic_x,
        const double stereographic_y,
        const double target_pixel_x,
        const double target_pixel_y
    ) {
        return new ceres::AutoDiffCostFunction<
            StereographicOpenCVError,
            2,
            18>(
            new StereographicOpenCVError(
                stereographic_x,
                stereographic_y,
                target_pixel_x,
                target_pixel_y
            )
        );
    }

    double stereographic_x;
    double stereographic_y;
    double target_pixel_x;
    double target_pixel_y;
};

py::dict get_matching_spline_distortion_model(
    std::vector<double>& opencv_distortion_params,
    PinholeSplinedOptimizationConfig& model_config,
    double image_bound_x,
    double image_bound_y
) {
    const SplineMap map(model_config);

    const double fov_rad_x = model_config.fov_deg_x * M_PI / 180.0;
    const double fov_rad_y = model_config.fov_deg_y * M_PI / 180.0;

    // Sampling range in normalized (pinhole) space
    const double half_x_pinhole = std::tan(fov_rad_x / 2.0);
    const double half_y_pinhole = std::tan(fov_rad_y / 2.0);

    const uint32_t num_samples_x = model_config.num_knots_x;
    const uint32_t num_samples_y = model_config.num_knots_y;

    // y-major storage: knots[y][x]
    auto x_knots = vector_mat<double>(
        model_config.num_knots_y,
        model_config.num_knots_x,
        0
    );
    auto y_knots = vector_mat<double>(
        model_config.num_knots_y,
        model_config.num_knots_x,
        0
    );

    ceres::Problem problem;

    // add all control points (y-major)
    for (size_t y = 0; y < model_config.num_knots_y; ++y) {
        for (size_t x = 0; x < model_config.num_knots_x; ++x) {
            problem.AddParameterBlock(&x_knots[y][x], 1);
            problem.AddParameterBlock(&y_knots[y][x], 1);
        }
    }

    // Track which cells have in-image distortion residuals
    std::vector<bool> cell_has_data(map.Nx * map.Ny, false);

    for (size_t y_sample_idx = 0; y_sample_idx < num_samples_y;
         ++y_sample_idx) {
        for (size_t x_sample_idx = 0; x_sample_idx < num_samples_x;
             ++x_sample_idx) {
            const double x_proportion =
                static_cast<double>(x_sample_idx) /
                (static_cast<double>(num_samples_x) - 1);
            const double y_proportion =
                static_cast<double>(y_sample_idx) /
                (static_cast<double>(num_samples_y) - 1);

            // Sample in normalized (pinhole) space
            const double x_normalized =
                -half_x_pinhole + 2.0 * half_x_pinhole * x_proportion;
            const double y_normalized =
                -half_y_pinhole + 2.0 * half_y_pinhole * y_proportion;

            double x_spline, y_spline;
            map.normalized_to_grid_coords(
                x_normalized,
                y_normalized,
                x_spline,
                y_spline
            );

            const int ix = static_cast<int>(x_spline);
            const int iy = static_cast<int>(y_spline);

            if (!map.is_inside_fov(ix, iy)) {
                continue;
            }

            // Skip distortion matching outside the image bounds
            if (std::abs(x_normalized) > image_bound_x ||
                std::abs(y_normalized) > image_bound_y) {
                continue;
            }

            cell_has_data[iy * map.Nx + ix] = true;

            Vec2<double> normalized_point(x_normalized, y_normalized);

            Vec2<double> opencv_distorted_point;
            distort_opencv(
                opencv_distortion_params.data(),
                normalized_point,
                opencv_distorted_point
            );

            const double u = x_spline - static_cast<double>(ix);
            const double v = y_spline - static_cast<double>(iy);

            ceres::CostFunction* cost = DistortionError::Create(
                opencv_distorted_point.x(),
                opencv_distorted_point.y(),
                x_normalized,
                y_normalized,
                u,
                v
            );

            // clang-format off
            problem.AddResidualBlock(
                cost,
                nullptr,

                // dx control points (y-major): y first, then x
                &x_knots[iy - 1][ix - 1], &x_knots[iy - 1][ix + 0],
                &x_knots[iy - 1][ix + 1], &x_knots[iy - 1][ix + 2],

                &x_knots[iy + 0][ix - 1], &x_knots[iy + 0][ix + 0],
                &x_knots[iy + 0][ix + 1], &x_knots[iy + 0][ix + 2],

                &x_knots[iy + 1][ix - 1], &x_knots[iy + 1][ix + 0],
                &x_knots[iy + 1][ix + 1], &x_knots[iy + 1][ix + 2],

                &x_knots[iy + 2][ix - 1], &x_knots[iy + 2][ix + 0],
                &x_knots[iy + 2][ix + 1], &x_knots[iy + 2][ix + 2],

                // dy control points (y-major): y first, then x
                &y_knots[iy - 1][ix - 1], &y_knots[iy - 1][ix + 0],
                &y_knots[iy - 1][ix + 1], &y_knots[iy - 1][ix + 2],

                &y_knots[iy + 0][ix - 1], &y_knots[iy + 0][ix + 0],
                &y_knots[iy + 0][ix + 1], &y_knots[iy + 0][ix + 2],

                &y_knots[iy + 1][ix - 1], &y_knots[iy + 1][ix + 0],
                &y_knots[iy + 1][ix + 1], &y_knots[iy + 1][ix + 2],

                &y_knots[iy + 2][ix - 1], &y_knots[iy + 2][ix + 0],
                &y_knots[iy + 2][ix + 1], &y_knots[iy + 2][ix + 2]
            );
            // clang-format on
        }
    }

    // Smoothness priors for cells without in-image data
    const double sqrt_lambda = std::sqrt(model_config.smoothness_lambda);
    for (int cy = 0; cy < map.Ny; cy++) {
        for (int cx = 0; cx < map.Nx; cx++) {
            if (cell_has_data[cy * map.Nx + cx]) {
                continue;
            }

            // Horizontal: 4-knot stencil along rows
            if (cx - 1 >= 0 && cx + 2 < map.Nx) {
                for (int row = cy; row <= cy + 1 && row < map.Ny; row++) {
                    auto* cost = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost,
                        nullptr,
                        &x_knots[row][cx - 1],
                        &x_knots[row][cx],
                        &x_knots[row][cx + 1],
                        &x_knots[row][cx + 2]
                    );
                    auto* cost_y = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost_y,
                        nullptr,
                        &y_knots[row][cx - 1],
                        &y_knots[row][cx],
                        &y_knots[row][cx + 1],
                        &y_knots[row][cx + 2]
                    );
                }
            }

            // Vertical: 4-knot stencil along columns
            if (cy - 1 >= 0 && cy + 2 < map.Ny) {
                for (int col = cx; col <= cx + 1 && col < map.Nx; col++) {
                    auto* cost = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost,
                        nullptr,
                        &x_knots[cy - 1][col],
                        &x_knots[cy][col],
                        &x_knots[cy + 1][col],
                        &x_knots[cy + 2][col]
                    );
                    auto* cost_y = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost_y,
                        nullptr,
                        &y_knots[cy - 1][col],
                        &y_knots[cy][col],
                        &y_knots[cy + 1][col],
                        &y_knots[cy + 2][col]
                    );
                }
            }
        }
    }

    spdlog::debug("Created problem");
    ceres::Solver::Options options;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = 10'000;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    spdlog::debug("Solve done");
    spdlog::debug(summary.BriefReport());

    // Return NumPy arrays in y-major shape (Y, X)
    py::array_t<double> x_array(
        {model_config.num_knots_y, model_config.num_knots_x}
    );
    py::array_t<double> y_array(
        {model_config.num_knots_y, model_config.num_knots_x}
    );

    auto x_buf = x_array.mutable_unchecked<2>();
    auto y_buf = y_array.mutable_unchecked<2>();

    for (size_t y = 0; y < model_config.num_knots_y; ++y) {
        for (size_t x = 0; x < model_config.num_knots_x; ++x) {
            x_buf(y, x) = x_knots[y][x];
            y_buf(y, x) = y_knots[y][x];
        }
    }

    py::dict out;
    out["x_knots"] = x_array;
    out["y_knots"] = y_array;

    out["x_range_start"] = -map.half_x;
    out["x_range_end"] = map.half_x;
    out["y_range_start"] = -map.half_y;
    out["y_range_end"] = map.half_y;

    return out;
}

py::dict get_matching_stereographic_opencv_model(
    uint32_t image_width,
    uint32_t image_height,
    double stereographic_focal_length,
    std::vector<bool>& distortion_param_optimize_mask
) {
    if (distortion_param_optimize_mask.size() != 14) {
        throw py::value_error(
            "distortion_param_optimize_mask must have length 14"
        );
    }

    const double cx = static_cast<double>(image_width) / 2.0;
    const double cy = static_cast<double>(image_height) / 2.0;

    std::vector<double> intrinsics(18, 0.0);
    intrinsics[0] = stereographic_focal_length;
    intrinsics[1] = stereographic_focal_length;
    intrinsics[2] = cx;
    intrinsics[3] = cy;

    ceres::Problem problem;
    problem.AddParameterBlock(intrinsics.data(), intrinsics.size());

    std::vector<int> fixed_intrinsics_param_indices = {2, 3};
    for (size_t param_idx = 0; param_idx < distortion_param_optimize_mask.size();
         ++param_idx) {
        if (distortion_param_optimize_mask[param_idx]) {
            continue;
        }
        fixed_intrinsics_param_indices.push_back(static_cast<int>(4 + param_idx));
    }

    auto* manifold =
        new ceres::SubsetManifold(intrinsics.size(), fixed_intrinsics_param_indices);
    problem.SetManifold(intrinsics.data(), manifold);

    const uint32_t num_samples_x = std::max<uint32_t>(32, image_width / 16);
    const uint32_t num_samples_y = std::max<uint32_t>(24, image_height / 16);

    for (uint32_t y_sample_idx = 0; y_sample_idx < num_samples_y;
         ++y_sample_idx) {
        for (uint32_t x_sample_idx = 0; x_sample_idx < num_samples_x;
             ++x_sample_idx) {
            const double x_proportion =
                static_cast<double>(x_sample_idx) /
                (static_cast<double>(num_samples_x) - 1.0);
            const double y_proportion =
                static_cast<double>(y_sample_idx) /
                (static_cast<double>(num_samples_y) - 1.0);

            const double pixel_x =
                static_cast<double>(image_width - 1) * x_proportion;
            const double pixel_y =
                static_cast<double>(image_height - 1) * y_proportion;
            const double stereographic_x =
                (pixel_x - cx) / stereographic_focal_length;
            const double stereographic_y =
                (pixel_y - cy) / stereographic_focal_length;

            problem.AddResidualBlock(
                StereographicOpenCVError::Create(
                    stereographic_x,
                    stereographic_y,
                    pixel_x,
                    pixel_y
                ),
                nullptr,
                intrinsics.data()
            );
        }
    }

    ceres::Solver::Options options;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = 10'000;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    py::dict out;
    out["intrinsics"] = intrinsics;
    out["final_cost"] = summary.final_cost;
    return out;
}

}  // namespace lensboy
