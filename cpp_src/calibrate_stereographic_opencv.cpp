#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include "./calibrate.hpp"
#include "./cameramodels.hpp"
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

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::ACCELERATE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::ACCELERATE_SPARSE;
        return true;
    }

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        return true;
    }

    return false;
}

struct ReprojectionErrorStereographicOpenCVWarp {
    ReprojectionErrorStereographicOpenCVWarp(
        const Vec3<double> point_in_target,
        const double observed_x,
        const double observed_y,
        const WarpCoordinates warp_coords
    )
        : point_in_target(point_in_target),
          observed_x(observed_x),
          observed_y(observed_y),
          warp_coords(warp_coords) {}

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const camera_from_target,
        const T* const warp_coeffs,
        T* residuals
    ) const {
        Vec6<T> eigen_camera_from_target(camera_from_target);
        Vec3<T> eigen_point_in_target = point_in_target.cast<T>();
        Vec3<T> warped_point = apply_warp_to_target_point(
            eigen_point_in_target,
            warp_coords,
            warp_coeffs
        );

        Vec3<T> eigen_point_in_cam =
            transform_point(eigen_camera_from_target, warped_point);

        Vec2<T> image_point;
        project_stereographic_opencv(
            intrinsics,
            eigen_point_in_cam,
            image_point
        );

        residuals[0] = image_point[0] - observed_x;
        residuals[1] = image_point[1] - observed_y;

        return true;
    }

    static ceres::CostFunction* create(
        const Vec3<double>& point_in_target,
        const double observed_x,
        const double observed_y,
        const WarpCoordinates& warp_coords
    ) {
        return new ceres::AutoDiffCostFunction<
            ReprojectionErrorStereographicOpenCVWarp,
            2,
            18,
            6,
            5>(
            new ReprojectionErrorStereographicOpenCVWarp(
                point_in_target,
                observed_x,
                observed_y,
                warp_coords
            )
        );
    }

    Vec3<double> point_in_target;
    double observed_x;
    double observed_y;
    WarpCoordinates warp_coords;
};

struct ReprojectionErrorStereographicOpenCVNoWarp {
    ReprojectionErrorStereographicOpenCVNoWarp(
        const Vec3<double> point_in_target,
        const double observed_x,
        const double observed_y
    )
        : point_in_target(point_in_target),
          observed_x(observed_x),
          observed_y(observed_y) {}

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const camera_from_target,
        T* residuals
    ) const {
        Vec6<T> eigen_camera_from_target(camera_from_target);
        Vec3<T> eigen_point_in_target = point_in_target.cast<T>();

        Vec3<T> eigen_point_in_cam =
            transform_point(eigen_camera_from_target, eigen_point_in_target);

        Vec2<T> image_point;
        project_stereographic_opencv(
            intrinsics,
            eigen_point_in_cam,
            image_point
        );

        residuals[0] = image_point[0] - observed_x;
        residuals[1] = image_point[1] - observed_y;

        return true;
    }

    static ceres::CostFunction* create(
        const Vec3<double>& point_in_target,
        const double observed_x,
        const double observed_y
    ) {
        return new ceres::AutoDiffCostFunction<
            ReprojectionErrorStereographicOpenCVNoWarp,
            2,
            18,
            6>(
            new ReprojectionErrorStereographicOpenCVNoWarp(
                point_in_target,
                observed_x,
                observed_y
            )
        );
    }

    Vec3<double> point_in_target;
    double observed_x;
    double observed_y;
};

struct StereographicOpenCVOptimizationState {
    std::vector<double> intrinsics;
    std::vector<std::vector<double>> cameras_from_target;
    std::vector<std::vector<double>> target_points;

    static StereographicOpenCVOptimizationState from_input(
        std::vector<double>& intrinsics_initial_value,
        std::vector<Vec6<double>>& cameras_from_target,
        std::vector<Vec3<double>>& target_points
    ) {
        std::vector<std::vector<double>> camera_poses_out;
        for (auto& vec : cameras_from_target) {
            camera_poses_out.push_back(
                std::vector<double>(vec.data(), vec.data() + vec.size())
            );
        }

        std::vector<std::vector<double>> target_points_out;
        for (auto& vec : target_points) {
            target_points_out.push_back(
                std::vector<double>(vec.data(), vec.data() + vec.size())
            );
        }

        return StereographicOpenCVOptimizationState{
            intrinsics_initial_value,
            std::move(camera_poses_out),
            std::move(target_points_out),
        };
    }

    py::dict make_dict(
        const double* warp_coeffs
    ) {
        py::dict result;
        result["intrinsics"] = this->intrinsics;
        result["cameras_from_target"] = this->cameras_from_target;
        result["warp_coeffs"] = py::array_t<double>(5, warp_coeffs);
        return result;
    }
};

py::dict calibrate_stereographic_opencv(
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates,
    std::array<double, 5> warp_coeffs_initial
) {
    require(
        intrinsics_initial_value.size() == 18,
        "Stereographic OpenCV intrinsics_initial_value must have length 18"
    );
    require(
        intrinsics_param_optimize_mask.size() == 18,
        "Stereographic OpenCV intrinsics_param_optimize_mask must have length 18"
    );

    ceres::Problem problem;

    const bool has_warp = warp_coordinates.has_value();
    WarpCoordinates warp_coords;
    if (has_warp) {
        warp_coords = *warp_coordinates;
    }
    double warp_coeffs[5] = {
        warp_coeffs_initial[0],
        warp_coeffs_initial[1],
        warp_coeffs_initial[2],
        warp_coeffs_initial[3],
        warp_coeffs_initial[4]
    };

    auto state = StereographicOpenCVOptimizationState::from_input(
        intrinsics_initial_value,
        cameras_from_target,
        target_points
    );

    problem.AddParameterBlock(state.intrinsics.data(), state.intrinsics.size());
    std::vector<int> fixed_intrinsics_param_indices;
    for (size_t param_idx = 0; param_idx < state.intrinsics.size();
         param_idx++) {
        if (!intrinsics_param_optimize_mask[param_idx]) {
            fixed_intrinsics_param_indices.push_back(param_idx);
        }
    }
    auto* manifold = new ceres::SubsetManifold(
        state.intrinsics.size(),
        fixed_intrinsics_param_indices
    );
    problem.SetManifold(state.intrinsics.data(), manifold);

    if (has_warp) {
        problem.AddParameterBlock(warp_coeffs, 5);
    }

    for (auto& cam : state.cameras_from_target) {
        problem.AddParameterBlock(cam.data(), cam.size());
    }

    for (size_t camera_idx = 0; camera_idx < frames.size(); camera_idx++) {
        auto& target_point_indices = std::get<0>(frames[camera_idx]);
        auto& observations = std::get<1>(frames[camera_idx]);
        auto& camera_pose = state.cameras_from_target[camera_idx];

        for (size_t observation_idx = 0; observation_idx < observations.size();
             observation_idx++) {
            auto& observation = observations[observation_idx];
            const auto& target =
                state.target_points[target_point_indices[observation_idx]];

            if (has_warp) {
                problem.AddResidualBlock(
                    ReprojectionErrorStereographicOpenCVWarp::create(
                        Vec3<double>(target.data()),
                        observation(0, 0),
                        observation(1, 0),
                        warp_coords
                    ),
                    nullptr,
                    state.intrinsics.data(),
                    camera_pose.data(),
                    warp_coeffs
                );
            } else {
                problem.AddResidualBlock(
                    ReprojectionErrorStereographicOpenCVNoWarp::create(
                        Vec3<double>(target.data()),
                        observation(0, 0),
                        observation(1, 0)
                    ),
                    nullptr,
                    state.intrinsics.data(),
                    camera_pose.data()
                );
            }
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

    return state.make_dict(warp_coeffs);
}

}  // namespace lensboy
