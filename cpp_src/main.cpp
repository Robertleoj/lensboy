#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include "./lut_max_cell_error.hpp"
#include "./python_camera_functions.hpp"
#include "./seeded_normalize.hpp"
#include "calibrate.hpp"
#include "cameramodels.hpp"

namespace py = pybind11;

int add(
    int a,
    int b
) {
    return a + b;
}

PYBIND11_MODULE(
    lensboy_bindings,
    m
) {
    m.doc() = "lensboy for camera calibration";
    spdlog::flush_on(spdlog::level::trace);
    py::class_<lensboy::PinholeSplinedConfig>(m, "PinholeSplinedConfig")
        .def(
            py::init<
                uint32_t,
                uint32_t,
                double,
                double,
                uint32_t,
                uint32_t,
                double>(),
            py::arg("image_width"),
            py::arg("image_height"),
            py::arg("fov_deg_x"),
            py::arg("fov_deg_y"),
            py::arg("num_knots_x"),
            py::arg("num_knots_y"),
            py::arg("smoothness_lambda")
        )
        .def_readwrite(
            "image_width",
            &lensboy::PinholeSplinedConfig::image_width
        )
        .def_readwrite(
            "image_height",
            &lensboy::PinholeSplinedConfig::image_height
        )
        .def_readwrite("fov_deg_x", &lensboy::PinholeSplinedConfig::fov_deg_x)
        .def_readwrite("fov_deg_y", &lensboy::PinholeSplinedConfig::fov_deg_y)

        .def_readwrite(
            "num_knots_x",
            &lensboy::PinholeSplinedConfig::num_knots_x
        )
        .def_readwrite(
            "num_knots_y",
            &lensboy::PinholeSplinedConfig::num_knots_y
        )
        .def_readwrite(
            "smoothness_lambda",
            &lensboy::PinholeSplinedConfig::smoothness_lambda
        )
        .def("__repr__", [](const lensboy::PinholeSplinedConfig& self) {
            std::ostringstream oss;
            oss << "PinholeSplinedConfig("
                << "image_width=" << self.image_width
                << ", image_height=" << self.image_height
                << ", fov_deg_x=" << self.fov_deg_x
                << ", fov_deg_y=" << self.fov_deg_y
                << ", num_knots_x=" << self.num_knots_x
                << ", num_knots_y=" << self.num_knots_y
                << ", smoothness_lambda=" << self.smoothness_lambda << ")";
            return oss.str();
        });

    py::class_<lensboy::PinholeSplinedIntrinsicsParameters>(
        m,
        "PinholeSplinedIntrinsicsParameters"
    )
        .def(
            py::init([](py::array_t<double> pinhole_parameters,
                        py::array_t<double> dx_grid,
                        py::array_t<double> dy_grid) {
                using A = py::
                    array_t<double, py::array::c_style | py::array::forcecast>;
                auto pinhole_params_ = A(pinhole_parameters);
                auto dx_ = A(dx_grid);
                auto dy_ = A(dy_grid);

                auto pinhole_params_buf = pinhole_params_.request();
                if (pinhole_params_buf.ndim != 1 ||
                    pinhole_params_buf.shape[0] != 4) {
                    throw py::value_error(
                        "pinhole_parameters must have shape (4,)"
                    );
                }

                auto dxb = dx_.request();
                auto dyb = dy_.request();
                if (dxb.ndim != 2) {
                    throw py::value_error("dx_grid must be a 2D array");
                }
                if (dyb.ndim != 2) {
                    throw py::value_error("dy_grid must be a 2D array");
                }

                return lensboy::PinholeSplinedIntrinsicsParameters{
                    pinhole_params_,
                    dx_,
                    dy_
                };
            }),
            py::arg("pinhole_parameters"),
            py::arg("dx_grid"),
            py::arg("dy_grid")
        )
        .def_readwrite(
            "pinhole_parameters",
            &lensboy::PinholeSplinedIntrinsicsParameters::pinhole_parameters
        )
        .def_readwrite(
            "dx_grid",
            &lensboy::PinholeSplinedIntrinsicsParameters::dx_grid
        )
        .def_readwrite(
            "dy_grid",
            &lensboy::PinholeSplinedIntrinsicsParameters::dy_grid
        )
        .def(
            "__repr__",
            [](const lensboy::PinholeSplinedIntrinsicsParameters& self) {
                auto dxb = self.dx_grid.request();
                std::ostringstream oss;
                oss << "PinholeSplinedIntrinsicsParameters("
                    << "dx_grid_shape=(" << dxb.shape[0] << ", " << dxb.shape[1]
                    << "))";
                return oss.str();
            }
        );

    py::class_<lensboy::WarpCoordinates>(m, "WarpCoordinates")
        .def(
            py::init<lensboy::Vec6<double>, double, double>(),
            py::arg("target_from_warp_frame"),
            py::arg("x_scale"),
            py::arg("y_scale")
        )
        .def_readwrite(
            "target_from_warp_frame",
            &lensboy::WarpCoordinates::target_from_warp_frame
        )
        .def_readwrite("x_scale", &lensboy::WarpCoordinates::x_scale)
        .def_readwrite("y_scale", &lensboy::WarpCoordinates::y_scale)
        .def("__repr__", [](const lensboy::WarpCoordinates& self) {
            std::ostringstream oss;
            oss << "WarpCoordinates("
                << "target_from_warp_frame=["
                << self.target_from_warp_frame.transpose() << "]"
                << ", x_scale=" << self.x_scale << ", y_scale=" << self.y_scale
                << ")";
            return oss.str();
        });

    m.def(
        "add",
        &add,
        "Add two integers together - test",
        py::arg("a"),
        py::arg("b")
    );

    m.def(
        "calibrate_opencv",
        &lensboy::calibrate_opencv,
        py::arg("intrinsics_initial_value"),
        py::arg("intrinsics_param_optimize_mask"),
        py::arg("cameras_from_target"),
        py::arg("target_points"),
        py::arg("frames"),
        py::arg("warp_coordinates") = py::none(),
        py::arg("warp_coeffs_initial") =
            std::array<double, 5>{0.0, 0.0, 0.0, 0.0, 0.0}
    );

    m.def(
        "get_matching_spline_distortion_model",
        &lensboy::get_matching_spline_distortion_model,
        py::arg("opencv_distortion_params"),
        py::arg("model_config"),
        py::arg("image_bound_x"),
        py::arg("image_bound_y")
    );

    m.def(
        "fine_tune_pinhole_splined",
        &lensboy::fine_tune_pinhole_splined,
        py::arg("model_config"),
        py::arg("intrinsics_parameters"),
        py::arg("cameras_from_target"),
        py::arg("target_points"),
        py::arg("frames"),
        py::arg("warp_coordinates") = py::none(),
        py::arg("warp_coeffs_initial") =
            std::array<double, 5>{0.0, 0.0, 0.0, 0.0, 0.0}
    );

    m.def(
        "project_pinhole_splined_points",
        &lensboy::project_pinhole_splined_pywrapper,
        py::arg("model_config"),
        py::arg("intrinsics"),
        py::arg("points_in_camera")
    );

    m.def(
        "normalize_pinhole_splined_points",
        &lensboy::normalize_pinhole_splined_points,
        py::arg("model_config"),
        py::arg("intrinsics"),
        py::arg("pixel_coords")
    );

    m.def(
        "warp_target_points",
        &lensboy::warp_target_points,
        py::arg("warp_coordinates"),
        py::arg("warp_coeffs"),
        py::arg("target_points")
    );

    m.def(
        "make_undistortion_maps_pinhole_splined",
        &lensboy::make_undistortion_maps_pinhole_splined,
        py::arg("model_config"),
        py::arg("intrinsics"),
        py::arg("pinhole_parameters"),
        py::arg("image_size_wh")
    );

    m.def(
        "seeded_normalize_opencv",
        &lensboy::seeded_normalize_opencv,
        py::arg("seed_pixels"),
        py::arg("seed_normals"),
        py::arg("seed_width"),
        py::arg("seed_height"),
        py::arg("query_pixels"),
        py::arg("intrinsics")
    );

    m.def(
        "seeded_normalize_splined",
        &lensboy::seeded_normalize_splined,
        py::arg("seed_pixels"),
        py::arg("seed_normals"),
        py::arg("seed_width"),
        py::arg("seed_height"),
        py::arg("query_pixels"),
        py::arg("config"),
        py::arg("intrinsics")
    );

    m.def(
        "max_cell_errors_pinhole_splined",
        &lensboy::max_cell_errors_pinhole_splined,
        py::arg("config"),
        py::arg("intrinsics"),
        py::arg("lut_xy_grid"),
        py::arg("grid_x_min"),
        py::arg("grid_x_max"),
        py::arg("grid_y_min"),
        py::arg("grid_y_max"),
        py::arg("interpolation_mode"),
        py::arg("max_iterations"),
        py::arg("gradient_tolerance")
    );

    m.def(
        "max_cell_errors_opencv",
        &lensboy::max_cell_errors_opencv,
        py::arg("intrinsics"),
        py::arg("lut_xy_grid"),
        py::arg("grid_x_min"),
        py::arg("grid_x_max"),
        py::arg("grid_y_min"),
        py::arg("grid_y_max"),
        py::arg("interpolation_mode"),
        py::arg("max_iterations"),
        py::arg("gradient_tolerance")
    );

    m.def(
        "set_log_level",
        [](const std::string& level) {
            if (level == "trace") {
                spdlog::set_level(spdlog::level::trace);
            } else if (level == "debug") {
                spdlog::set_level(spdlog::level::debug);
            } else if (level == "info") {
                spdlog::set_level(spdlog::level::info);
            } else if (level == "warn") {
                spdlog::set_level(spdlog::level::warn);
            } else if (level == "error") {
                spdlog::set_level(spdlog::level::err);
            } else if (level == "critical") {
                spdlog::set_level(spdlog::level::critical);
            } else if (level == "off") {
                spdlog::set_level(spdlog::level::off);
            } else {
                throw py::value_error(
                    "Unknown log level: '" + level +
                    "'. "
                    "Use trace/debug/info/warn/error/critical/off."
                );
            }
        },
        py::arg("level")
    );
}
