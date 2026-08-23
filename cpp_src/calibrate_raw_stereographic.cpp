#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <vector>
#include "./calibrate.hpp"
#include "./ceres_geometry.hpp"
#include "./pybind_utils.hpp"

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

template <typename T>
static void project_raw_stereographic(
    const T* const intrinsics,
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    using std::sqrt;

    const T norm = sqrt(
        point_in_camera[0] * point_in_camera[0] +
        point_in_camera[1] * point_in_camera[1] +
        point_in_camera[2] * point_in_camera[2]
    );
    const T denominator = norm + point_in_camera[2];
    const T sx = T(2.0) * point_in_camera[0] / denominator;
    const T sy = T(2.0) * point_in_camera[1] / denominator;

    result[0] = intrinsics[0] * sx + intrinsics[2];
    result[1] = intrinsics[1] * sy + intrinsics[3];
}

struct RawStereographicReprojectionError {
    Vec3<double> point_in_target;
    double observed_x;
    double observed_y;

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const camera_from_target,
        T* residuals
    ) const {
        Vec6<T> eigen_camera_from_target(camera_from_target);
        Vec3<T> eigen_point_in_target = point_in_target.cast<T>();
        Vec3<T> point_in_camera =
            transform_point(eigen_camera_from_target, eigen_point_in_target);

        Vec2<T> image_point;
        project_raw_stereographic(intrinsics, point_in_camera, image_point);

        residuals[0] = image_point[0] - T(observed_x);
        residuals[1] = image_point[1] - T(observed_y);
        return true;
    }
};

py::dict calibrate_raw_stereographic(
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames
) {
    require(
        intrinsics_initial_value.size() == 4,
        "Raw stereographic intrinsics_initial_value must have length 4"
    );
    require(
        intrinsics_param_optimize_mask.size() == 4,
        "Raw stereographic intrinsics_param_optimize_mask must have length 4"
    );
    require(
        cameras_from_target.size() == frames.size(),
        "cameras_from_target and frames must have the same length"
    );

    ceres::Problem problem;
    problem.AddParameterBlock(intrinsics_initial_value.data(), 4);

    std::vector<int> fixed_intrinsics_param_indices;
    for (size_t param_idx = 0; param_idx < intrinsics_initial_value.size();
         param_idx++) {
        if (!intrinsics_param_optimize_mask[param_idx]) {
            fixed_intrinsics_param_indices.push_back(static_cast<int>(param_idx));
        }
    }
    auto* manifold = new ceres::SubsetManifold(4, fixed_intrinsics_param_indices);
    problem.SetManifold(intrinsics_initial_value.data(), manifold);

    for (auto& camera_from_target : cameras_from_target) {
        problem.AddParameterBlock(camera_from_target.data(), 6);
    }

    for (size_t camera_idx = 0; camera_idx < frames.size(); camera_idx++) {
        auto& target_point_indices = std::get<0>(frames[camera_idx]);
        auto& observations = std::get<1>(frames[camera_idx]);
        auto& camera_from_target = cameras_from_target[camera_idx];

        for (size_t observation_idx = 0; observation_idx < observations.size();
             observation_idx++) {
            const int32_t target_idx = target_point_indices[observation_idx];
            const auto& observation = observations[observation_idx];
            auto* cost = new ceres::AutoDiffCostFunction<
                RawStereographicReprojectionError,
                2,
                4,
                6>(new RawStereographicReprojectionError{
                target_points[target_idx],
                observation(0, 0),
                observation(1, 0),
            });
            problem.AddResidualBlock(
                cost,
                nullptr,
                intrinsics_initial_value.data(),
                camera_from_target.data()
            );
        }
    }

    ceres::Solver::Options options;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (!configure_sparse_schur_if_available(options)) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
    }
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    spdlog::debug(summary.FullReport());

    std::vector<std::vector<double>> poses_out;
    poses_out.reserve(cameras_from_target.size());
    for (auto& camera_from_target : cameras_from_target) {
        poses_out.emplace_back(
            camera_from_target.data(),
            camera_from_target.data() + camera_from_target.size()
        );
    }

    py::dict result;
    result["intrinsics"] = intrinsics_initial_value;
    result["cameras_from_target"] = poses_out;
    return result;
}

}  // namespace lensboy
