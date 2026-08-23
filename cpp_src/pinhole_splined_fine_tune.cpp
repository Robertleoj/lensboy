#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <vector>

#include "./calibrate.hpp"
#include "./cameramodels.hpp"
#include "./pinhole_splined_problem.hpp"
#include "./pybind_utils.hpp"
#include "./type_defs.hpp"

namespace lensboy {

static bool configure_sparse_schur_if_available(
    ceres::Solver::Options& options
) {
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        return true;
    }

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        return true;
    }

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::ACCELERATE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::ACCELERATE_SPARSE;
        return true;
    }

    return false;
}

py::dict fine_tune_pinhole_splined(
    lensboy::PinholeSplinedOptimizationConfig& model_config,
    lensboy::PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
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

    double* pinhole_params = static_cast<double*>(
        intrinsics_parameters.pinhole_parameters.request().ptr
    );
    double* dxp = static_cast<double*>(dxb.ptr);
    double* dyp = static_cast<double*>(dyb.ptr);
    const int normalized_x = static_cast<int>(model_config.num_knots_x);
    const int normalized_y = static_cast<int>(model_config.num_knots_y);
    const int n_knots = normalized_x * normalized_y;

    std::vector<Vec2<double>> knot_params(n_knots);
    for (int i = 0; i < n_knots; i++) {
        knot_params[i] = Vec2<double>(dxp[i], dyp[i]);
    }

    double warp_coeffs[5] = {
        warp_coeffs_initial[0],
        warp_coeffs_initial[1],
        warp_coeffs_initial[2],
        warp_coeffs_initial[3],
        warp_coeffs_initial[4]
    };

    SplineMap map(model_config);
    const double smoothness_weight = std::sqrt(model_config.smoothness_lambda);

    ceres::Solver::Options options;

    options.num_threads =
        std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
    if (!configure_sparse_schur_if_available(options)) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    options.minimizer_progress_to_stdout = false;

    constexpr int max_rebuilds = 1000;

    // declare these outside the loop so we don't reallocate on every rebuild
    std::vector<double*> knot_blocks;

    std::vector<ObservationRecord> obs_records;

    ceres::Solver::Summary last_summary;

    double prev_cost = std::numeric_limits<double>::max();
    int outer;
    for (outer = 0; outer < max_rebuilds; outer++) {
        ceres::Problem problem;

        build_pinhole_splined_problem(
            problem,
            model_config,
            map,
            pinhole_params,
            knot_params,
            warp_coeffs,
            warp_coordinates,
            cameras_from_target,
            target_points,
            frames,
            knot_blocks,
            obs_records,
            nullptr,
            smoothness_weight
        );

        spdlog::debug(
            "Solve pass {} (residuals wired for current cells)...",
            outer
        );

        ceres::Solve(options, &problem, &last_summary);

        if (!any_spline_cell_changed(
                map,
                cameras_from_target,
                target_points,
                obs_records
            )) {
            spdlog::debug(
                "No cell changes detected. Done after {} rebuild(s).",
                outer
            );
            break;
        }

        const double cost = last_summary.final_cost;
        const double rel_improvement = (prev_cost - cost) / (prev_cost + 1e-30);
        if (outer > 0 && rel_improvement < 1e-6) {
            spdlog::debug(
                "Cost converged (rel improvement {:.2e}). Done after {} "
                "rebuild(s).",
                rel_improvement,
                outer
            );
            break;
        }
        prev_cost = cost;

        spdlog::debug("Cell change detected -> rebuilding problem.");
    }
    spdlog::debug("Optimization done after {} rebuilds", outer);

    for (int i = 0; i < n_knots; i++) {
        dxp[i] = knot_params[i][0];
        dyp[i] = knot_params[i][1];
    }

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
