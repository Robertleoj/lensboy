#pragma once

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <vector>
#include "./cameramodels.hpp"
#include "./type_defs.hpp"

namespace lensboy {

namespace py = pybind11;

py::dict calibrate_opencv(
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates = std::nullopt,
    std::array<double, 5> warp_coeffs_initial = {0.0, 0.0, 0.0, 0.0, 0.0}
);

py::dict calibrate_stereographic_opencv(
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates = std::nullopt,
    std::array<double, 5> warp_coeffs_initial = {0.0, 0.0, 0.0, 0.0, 0.0}
);

py::dict get_matching_spline_distortion_model(
    std::vector<double>& opencv_distortion_params,
    PinholeSplinedOptimizationConfig& model_config,
    double image_bound_x,
    double image_bound_y
);

py::dict get_matching_stereographic_opencv_model(
    uint32_t image_width,
    uint32_t image_height,
    double stereographic_focal_length,
    std::vector<bool>& distortion_param_optimize_mask
);

py::dict fine_tune_pinhole_splined(
    PinholeSplinedOptimizationConfig& model_config,
    PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates = std::nullopt,
    std::array<double, 5> warp_coeffs_initial = {0.0, 0.0, 0.0, 0.0, 0.0}
);

py::dict fine_tune_stereographic_splined(
    StereographicSplinedOptimizationConfig& model_config,
    StereographicSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates = std::nullopt,
    std::array<double, 5> warp_coeffs_initial = {0.0, 0.0, 0.0, 0.0, 0.0}
);

py::array_t<double> normalize_pinhole_splined_points(
    PinholeSplinedModelDefinition& config,
    PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> pixel_coords
);

}  // namespace lensboy
