#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "./cameramodels.hpp"

namespace lensboy {

namespace py = pybind11;

// Output shape is (num_cells, 5): per cell row =
//   [max_angular_error_deg,
//    peak_pixel_x, peak_pixel_y,
//    error_delta_x, error_delta_y].
// error_delta is (approx_xy(project(n*)) - n*) at the optimised peak.
// num_cells = (grid_width - 1) * (grid_height - 1), row-major over (cell_y, cell_x).
//
// interpolation_mode: 0=nearest, 1=bilinear, 2=bicubic.

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
);

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
);

}  // namespace lensboy
