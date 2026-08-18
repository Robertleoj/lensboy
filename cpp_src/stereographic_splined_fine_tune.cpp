#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "./calibrate.hpp"
#include "./cameramodels.hpp"
#include "./ceres_geometry.hpp"
#include "./pybind_utils.hpp"
#include "./type_defs.hpp"

namespace lensboy {

struct ReprojectionErrorStereographicSplined {
    const StereographicSplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    double obs_x, obs_y;
    bool has_warp;
    WarpCoordinates warp_coords;
    int n_knots;

    template <typename T>
    bool operator()(
        T const* const* parameters,
        T* residuals
    ) const {
        const T* const cam = parameters[0];
        const T* const warp_coeffs = parameters[1];
        const T* const dx_grid = parameters[2];
        const T* const dy_grid = parameters[3];

        Vec3<T> pw_warped;
        if (has_warp) {
            pw_warped = apply_warp_to_target_point(
                Vec3<T>(pw.cast<T>()),
                warp_coords,
                warp_coeffs
            );
        } else {
            pw_warped = pw.cast<T>();
        }

        T pw_t[3] = {pw_warped[0], pw_warped[1], pw_warped[2]};
        T pc[3];
        ceres::AngleAxisRotatePoint(cam, pw_t, pc);
        pc[0] += cam[3];
        pc[1] += cam[4];
        pc[2] += cam[5];

        const Vec3<T> ray = normalize_to_unit_ray(Vec3<T>(pc[0], pc[1], pc[2]));
        T x_stereo, y_stereo;
        unit_ray_to_stereographic(ray, x_stereo, y_stereo);

        T gx, gy;
        map.stereo_to_grid_coords(x_stereo, y_stereo, gx, gy);

        const T dx = eval_bspline2d_uniform_cubic_clamped(
            dx_grid,
            map.Nx,
            map.Ny,
            gx,
            gy
        );
        const T dy = eval_bspline2d_uniform_cubic_clamped(
            dy_grid,
            map.Nx,
            map.Ny,
            gx,
            gy
        );

        residuals[0] = T(fx) * (x_stereo + dx) + T(cx) - T(obs_x);
        residuals[1] = T(fy) * (y_stereo + dy) + T(cy) - T(obs_y);
        return true;
    }
};

struct GridSmoothnessResidual {
    int k0, k1, k2, k3;
    double weight;

    template <typename T>
    bool operator()(
        T const* const* parameters,
        T* residuals
    ) const {
        const T* const grid = parameters[0];
        residuals[0] = T(weight) *
                       (-grid[k0] + T(3.0) * grid[k1] -
                        T(3.0) * grid[k2] + grid[k3]);
        return true;
    }
};

static void add_grid_smoothness(
    ceres::Problem& problem,
    double* grid,
    int nx,
    int ny,
    double weight
) {
    const int n_knots = nx * ny;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x + 3 < nx; x++) {
            auto* cost =
                new ceres::DynamicAutoDiffCostFunction<GridSmoothnessResidual>(
                    new GridSmoothnessResidual{
                        y * nx + x,
                        y * nx + x + 1,
                        y * nx + x + 2,
                        y * nx + x + 3,
                        weight
                    }
                );
            cost->AddParameterBlock(n_knots);
            cost->SetNumResiduals(1);
            problem.AddResidualBlock(cost, nullptr, grid);
        }
    }
    for (int y = 0; y + 3 < ny; y++) {
        for (int x = 0; x < nx; x++) {
            auto* cost =
                new ceres::DynamicAutoDiffCostFunction<GridSmoothnessResidual>(
                    new GridSmoothnessResidual{
                        y * nx + x,
                        (y + 1) * nx + x,
                        (y + 2) * nx + x,
                        (y + 3) * nx + x,
                        weight
                    }
                );
            cost->AddParameterBlock(n_knots);
            cost->SetNumResiduals(1);
            problem.AddResidualBlock(cost, nullptr, grid);
        }
    }
}

py::dict fine_tune_stereographic_splined(
    StereographicSplinedOptimizationConfig& model_config,
    StereographicSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates,
    std::array<double, 5> warp_coeffs_initial
) {
    auto dxb = intrinsics_parameters.dx_grid.request();
    auto dyb = intrinsics_parameters.dy_grid.request();
    require(
        static_cast<uint32_t>(dxb.shape[0]) == model_config.num_knots_y &&
            static_cast<uint32_t>(dxb.shape[1]) == model_config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        static_cast<uint32_t>(dyb.shape[0]) == model_config.num_knots_y &&
            static_cast<uint32_t>(dyb.shape[1]) == model_config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    auto params_buf = intrinsics_parameters.stereographic_parameters.request();
    require(
        params_buf.ndim == 1 && params_buf.shape[0] == 4,
        "stereographic_parameters must have shape (4,)"
    );
    const double* params = static_cast<const double*>(params_buf.ptr);
    double* dxp = static_cast<double*>(dxb.ptr);
    double* dyp = static_cast<double*>(dyb.ptr);

    double warp_coeffs[5] = {
        warp_coeffs_initial[0],
        warp_coeffs_initial[1],
        warp_coeffs_initial[2],
        warp_coeffs_initial[3],
        warp_coeffs_initial[4]
    };

    const bool has_warp = warp_coordinates.has_value();
    WarpCoordinates warp_coords;
    if (has_warp) {
        warp_coords = *warp_coordinates;
    }
    const int nx = static_cast<int>(model_config.num_knots_x);
    const int ny = static_cast<int>(model_config.num_knots_y);
    const int n_knots = nx * ny;
    const StereographicSplineMap map(model_config);

    ceres::Problem problem;
    problem.AddParameterBlock(warp_coeffs, 5);
    if (!has_warp) {
        problem.SetParameterBlockConstant(warp_coeffs);
    }
    problem.AddParameterBlock(dxp, n_knots);
    problem.AddParameterBlock(dyp, n_knots);

    for (auto& cam : cameras_from_target) {
        problem.AddParameterBlock(const_cast<double*>(cam.data()), 6);
    }
    for (auto& pt : target_points) {
        problem.AddParameterBlock(const_cast<double*>(pt.data()), 3);
        problem.SetParameterBlockConstant(const_cast<double*>(pt.data()));
    }

    for (size_t cam_idx = 0; cam_idx < frames.size(); cam_idx++) {
        auto& ids = std::get<0>(frames[cam_idx]);
        auto& observations = std::get<1>(frames[cam_idx]);
        auto& cam6 = cameras_from_target[cam_idx];
        for (size_t obs_idx = 0; obs_idx < observations.size(); obs_idx++) {
            const int pt_idx = ids[obs_idx];
            const auto& obs = observations[obs_idx];
            auto* cost = new ceres::DynamicAutoDiffCostFunction<
                ReprojectionErrorStereographicSplined>(
                new ReprojectionErrorStereographicSplined{
                    map,
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    target_points[pt_idx],
                    obs(0, 0),
                    obs(1, 0),
                    has_warp,
                    warp_coords,
                    n_knots
                }
            );
            cost->AddParameterBlock(6);
            cost->AddParameterBlock(5);
            cost->AddParameterBlock(n_knots);
            cost->AddParameterBlock(n_knots);
            cost->SetNumResiduals(2);
            problem.AddResidualBlock(
                cost,
                nullptr,
                const_cast<double*>(cam6.data()),
                warp_coeffs,
                dxp,
                dyp
            );
        }
    }

    const double smoothness_weight = std::sqrt(model_config.smoothness_lambda);
    if (smoothness_weight > 0.0) {
        add_grid_smoothness(problem, dxp, nx, ny, smoothness_weight);
        add_grid_smoothness(problem, dyp, nx, ny, smoothness_weight);
    }

    ceres::Solver::Options options;
    options.num_threads =
        std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    spdlog::debug(summary.FullReport());

    py::dict out;
    out["dx_grid"] = intrinsics_parameters.dx_grid;
    out["dy_grid"] = intrinsics_parameters.dy_grid;
    out["warp_coeffs"] = py::array_t<double>(5, warp_coeffs);

    std::vector<std::vector<double>> poses_out;
    poses_out.reserve(cameras_from_target.size());
    for (auto& cam : cameras_from_target) {
        poses_out.push_back(
            std::vector<double>(cam.data(), cam.data() + cam.size())
        );
    }
    out["cameras_from_target"] = poses_out;
    return out;
}

}  // namespace lensboy
