#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "./cameramodels.hpp"

namespace lensboy {

namespace py = pybind11;

// Output shape is (num_cells, 6): per cell row =
//   [peak_pixel_x, peak_pixel_y,
//    exact_x, exact_y,
//    approx_x, approx_y].
// exact_xy is the optimiser's converged n (the normalised xy that was
// implicit in the maximised sin²); approx_xy is the LUT's interpolated
// value at the peak pixel. Angular error and residual delta are derivable
// from these two rays.
// num_cells = (grid_width - 1) * (grid_height - 1), row-major over (cell_y,
// cell_x).
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
