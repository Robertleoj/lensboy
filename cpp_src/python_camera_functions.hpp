#include <spdlog/spdlog.h>
#include "./pybind_utils.hpp"
#include "cameramodels.hpp"
#include "ceres_geometry.hpp"

#include <cstdint>

namespace lensboy {

static py::array_t<double> project_pinhole_splined_pywrapper(
    lensboy::PinholeSplinedModelDefinition& model_config,
    lensboy::PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        points_in_camera
) {
    // --- grids: must match model_config dimensions ---
    auto dxb = intrinsics.dx_grid.request();
    auto dyb = intrinsics.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dxb.shape[1] == model_config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dyb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dyb.shape[1] == model_config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    // --- points: shape (N, 3) ---
    auto pb = points_in_camera.request();
    require(pb.ndim == 2, "points_in_camera must be a 2D numpy array");
    require(pb.shape[1] == 3, "points_in_camera must have shape (N, 3)");

    const ssize_t N = pb.shape[0];

    const double* pinhole_params =
        static_cast<const double*>(intrinsics.pinhole_parameters.request().ptr);
    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);
    const double* P = static_cast<const double*>(pb.ptr);

    // Output: (N, 2), C contiguous
    py::array_t<double> out({N, (ssize_t)2});
    auto ob = out.request();
    auto* O = static_cast<double*>(ob.ptr);

    py::gil_scoped_release release;

    for (ssize_t i = 0; i < N; ++i) {
        Vec3<double> p;
        p[0] = P[i * 3 + 0];
        p[1] = P[i * 3 + 1];
        p[2] = P[i * 3 + 2];

        Vec2<double> r;
        project_pinhole_splined<double>(
            &model_config,
            pinhole_params,  // fx, fy, cx, cy
            dxp,             // row-major contiguous (C-order)
            dyp,             // row-major contiguous (C-order)
            p,
            r
        );

        O[i * 2 + 0] = r[0];
        O[i * 2 + 1] = r[1];
    }

    return out;
}

static py::tuple make_undistortion_maps_pinhole_splined(
    lensboy::PinholeSplinedModelDefinition& model_config,
    lensboy::PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        pinhole_parameters,
    std::pair<int, int> image_size_wh
) {
    // --- grids: must match model_config dimensions ---
    auto dxb = intrinsics.dx_grid.request();
    auto dyb = intrinsics.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dxb.shape[1] == model_config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dyb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dyb.shape[1] == model_config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    auto pinhole_params_in_buf = intrinsics.pinhole_parameters.request();
    require(
        pinhole_params_in_buf.ndim == 1 && pinhole_params_in_buf.shape[0] == 4,
        "intrinsics.pinhole_parameters must have shape (4,)"
    );
    const double* pinhole_params_in =
        static_cast<const double*>(pinhole_params_in_buf.ptr);

    require(
        pinhole_params_in[0] != 0.0 && pinhole_params_in[1] != 0.0,
        "intrinsics.pinhole_parameters fx/fy must be non-zero"
    );

    // Undistorted/output camera pinhole_parameters (controls output view)
    double pinhole_params_out_storage[4];

    auto pinhole_params_out_buf = pinhole_parameters.request();
    require(
        pinhole_params_out_buf.ndim == 1 &&
            pinhole_params_out_buf.shape[0] == 4,
        "pinhole_parameters must have shape (4,)"
    );
    const double* p = static_cast<const double*>(pinhole_params_out_buf.ptr);
    for (int i = 0; i < 4; ++i) {
        pinhole_params_out_storage[i] = p[i];
    }
    const double* pinhole_params_out = pinhole_params_out_storage;

    const double fx_out = pinhole_params_out[0];
    const double fy_out = pinhole_params_out[1];
    const double cx_out = pinhole_params_out[2];
    const double cy_out = pinhole_params_out[3];
    require(
        fx_out != 0.0 && fy_out != 0.0,
        "pinhole_parameters fx/fy must be non-zero"
    );

    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);

    int W = image_size_wh.first;
    int H = image_size_wh.second;

    require(W > 0 && H > 0, "Image width/height must be > 0");
    py::array_t<float> map_x({(ssize_t)H, (ssize_t)W});
    py::array_t<float> map_y({(ssize_t)H, (ssize_t)W});
    auto mx = map_x.request();
    auto my = map_y.request();
    float* MX = static_cast<float*>(mx.ptr);
    float* MY = static_cast<float*>(my.ptr);

    for (int y = 0; y < H; ++y) {
        const double y_norm = (double(y) - cy_out) / fy_out;
        for (int x = 0; x < W; ++x) {
            const double x_norm = (double(x) - cx_out) / fx_out;

            Vec3<double> p(x_norm, y_norm, 1.0);
            Vec2<double> r;
            project_pinhole_splined(
                &model_config,
                pinhole_params_in,
                dxp,
                dyp,
                p,
                r
            );

            const int idx = y * W + x;
            MX[idx] = (float)r[0];
            MY[idx] = (float)r[1];
        }
    }

    return py::make_tuple(map_x, map_y);
}

static py::array_t<double> warp_target_points(
    const lensboy::WarpCoordinates& warp,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_points
) {
    auto cb = coeffs_arr.request();
    require(cb.ndim == 1 && cb.shape[0] == 5, "coeffs must have shape (5,)");
    const double* coeffs = static_cast<const double*>(cb.ptr);

    auto pb = target_points.request();
    require(pb.ndim == 2, "target_points must be a 2D numpy array");
    require(pb.shape[1] == 3, "target_points must have shape (N, 3)");

    const ssize_t N = pb.shape[0];
    const double* P = static_cast<const double*>(pb.ptr);

    py::array_t<double> out({N, (ssize_t)3});
    auto ob = out.request();
    auto* O = static_cast<double*>(ob.ptr);

    for (ssize_t i = 0; i < N; ++i) {
        Vec3<double> p;
        p[0] = P[i * 3 + 0];
        p[1] = P[i * 3 + 1];
        p[2] = P[i * 3 + 2];

        Vec3<double> warped =
            apply_warp_to_target_point<double>(p, warp, coeffs);

        O[i * 3 + 0] = warped[0];
        O[i * 3 + 1] = warped[1];
        O[i * 3 + 2] = warped[2];
    }

    return out;
}

}  // namespace lensboy