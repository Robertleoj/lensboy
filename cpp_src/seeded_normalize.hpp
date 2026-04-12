#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "./cameramodels.hpp"

namespace lensboy {
namespace py = pybind11;

py::array_t<double> seeded_normalize_opencv(
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_normals,
    int seed_w,
    int seed_h,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics
);

py::array_t<double> seeded_normalize_splined(
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_normals,
    int seed_w,
    int seed_h,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_pixels,
    PinholeSplinedConfig& config,
    PinholeSplinedIntrinsicsParameters& params
);

}  // namespace lensboy
