#include <ceres/jet.h>
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

static py::array_t<double> project_stereographic_opencv_pywrapper(
    lensboy::StereographicOpenCVModelDefinition& model_config,
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        points_in_camera
) {
    auto ib = intrinsics.request();
    require(ib.ndim == 1, "intrinsics must be a 1D numpy array");
    require(ib.shape[0] == 18, "intrinsics must have shape (18,)");

    auto pb = points_in_camera.request();
    require(pb.ndim == 2, "points_in_camera must be a 2D numpy array");
    require(pb.shape[1] == 3, "points_in_camera must have shape (N, 3)");

    const ssize_t N = pb.shape[0];
    const double* params = static_cast<const double*>(ib.ptr);
    const double* P = static_cast<const double*>(pb.ptr);

    py::array_t<double> out({N, (ssize_t)2});
    auto ob = out.request();
    auto* O = static_cast<double*>(ob.ptr);

    py::gil_scoped_release release;
    for (ssize_t i = 0; i < N; ++i) {
        Vec3<double> p(P[i * 3 + 0], P[i * 3 + 1], P[i * 3 + 2]);
        Vec2<double> r;
        project_stereographic_opencv<double>(
            params,
            p,
            r
        );
        O[i * 2 + 0] = r[0];
        O[i * 2 + 1] = r[1];
    }

    return out;
}

static py::array_t<double> normalize_stereographic_opencv_points(
    lensboy::StereographicOpenCVModelDefinition& model_config,
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> pixel_coords
) {
    auto ib = intrinsics.request();
    require(ib.ndim == 1, "intrinsics must be a 1D numpy array");
    require(ib.shape[0] == 18, "intrinsics must have shape (18,)");
    const double* params = static_cast<const double*>(ib.ptr);
    const double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
    require(fx != 0.0 && fy != 0.0, "fx/fy must be non-zero");

    auto pb = pixel_coords.request();
    require(pb.ndim == 2, "pixel_coords must be a 2D numpy array");
    require(pb.shape[1] == 2, "pixel_coords must have shape (N, 2)");
    const ssize_t N = pb.shape[0];
    const double* P = static_cast<const double*>(pb.ptr);

    py::array_t<double> out({N, (ssize_t)3});
    auto ob = out.request();
    double* O = static_cast<double*>(ob.ptr);

    constexpr int max_newton = 50;
    constexpr double tol = 1e-14;
    for (ssize_t i = 0; i < N; i++) {
        const double target_x = (P[i * 2 + 0] - cx) / fx;
        const double target_y = (P[i * 2 + 1] - cy) / fy;
        double sx = target_x;
        double sy = target_y;

        for (int iter = 0; iter < max_newton; iter++) {
            using Jet = ceres::Jet<double, 2>;
            Jet jsx(sx, 0);
            Jet jsy(sy, 1);
            Jet coeffs[14];
            for (int coeff_idx = 0; coeff_idx < 14; coeff_idx++) {
                coeffs[coeff_idx] = Jet(params[4 + coeff_idx]);
            }
            Vec2<Jet> distorted;
            distort_opencv(coeffs, Vec2<Jet>(jsx, jsy), distorted);

            const double res0 = distorted[0].a - target_x;
            const double res1 = distorted[1].a - target_y;
            if (res0 * res0 + res1 * res1 < tol * tol) {
                break;
            }

            const double J00 = distorted[0].v[0], J01 = distorted[0].v[1];
            const double J10 = distorted[1].v[0], J11 = distorted[1].v[1];
            const double det = J00 * J11 - J01 * J10;
            if (std::abs(det) < 1e-30) {
                break;
            }
            const double inv_det = 1.0 / det;
            sx -= inv_det * (J11 * res0 - J01 * res1);
            sy -= inv_det * (-J10 * res0 + J00 * res1);
        }

        Vec3<double> ray = stereographic_to_unit_ray(sx, sy);
        O[i * 3 + 0] = ray[0];
        O[i * 3 + 1] = ray[1];
        O[i * 3 + 2] = ray[2];
    }

    return out;
}

static py::array_t<double> project_stereographic_splined_pywrapper(
    lensboy::StereographicSplinedModelDefinition& model_config,
    lensboy::StereographicSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        points_in_camera
) {
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

    auto pb = points_in_camera.request();
    require(pb.ndim == 2, "points_in_camera must be a 2D numpy array");
    require(pb.shape[1] == 3, "points_in_camera must have shape (N, 3)");

    const ssize_t N = pb.shape[0];
    const double* params = static_cast<const double*>(
        intrinsics.stereographic_parameters.request().ptr
    );
    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);
    const double* P = static_cast<const double*>(pb.ptr);

    py::array_t<double> out({N, (ssize_t)2});
    auto ob = out.request();
    auto* O = static_cast<double*>(ob.ptr);

    py::gil_scoped_release release;
    for (ssize_t i = 0; i < N; ++i) {
        Vec3<double> p(P[i * 3 + 0], P[i * 3 + 1], P[i * 3 + 2]);
        Vec2<double> r;
        project_stereographic_splined<double>(
            &model_config,
            params,
            dxp,
            dyp,
            p,
            r
        );
        O[i * 2 + 0] = r[0];
        O[i * 2 + 1] = r[1];
    }

    return out;
}

static py::array_t<double> normalize_stereographic_splined_points(
    lensboy::StereographicSplinedModelDefinition& model_config,
    lensboy::StereographicSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> pixel_coords
) {
    using Jet = ceres::Jet<double, 2>;

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

    auto params_buf = intrinsics.stereographic_parameters.request();
    require(
        params_buf.ndim == 1 && params_buf.shape[0] == 4,
        "stereographic_parameters must have shape (4,)"
    );
    const double* params = static_cast<const double*>(params_buf.ptr);
    const double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
    require(fx != 0.0 && fy != 0.0, "fx/fy must be non-zero");

    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);
    const StereographicSplineMap map(model_config);

    auto pb = pixel_coords.request();
    require(pb.ndim == 2, "pixel_coords must be a 2D numpy array");
    require(pb.shape[1] == 2, "pixel_coords must have shape (N, 2)");
    const ssize_t N = pb.shape[0];
    const double* P = static_cast<const double*>(pb.ptr);

    py::array_t<double> out({N, (ssize_t)3});
    auto ob = out.request();
    double* O = static_cast<double*>(ob.ptr);

    constexpr int max_rebuilds = 25;
    constexpr int max_newton = 50;
    constexpr double tol_sq = 1e-20;

    for (ssize_t i = 0; i < N; i++) {
        const double target_u = P[i * 2 + 0];
        const double target_v = P[i * 2 + 1];
        double sx = (target_u - cx) / fx;
        double sy = (target_v - cy) / fy;

        for (int rebuild = 0; rebuild < max_rebuilds; rebuild++) {
            double gx, gy;
            map.stereo_to_grid_coords(sx, sy, gx, gy);
            const int ix0 = static_cast<int>(std::floor(gx));
            const int iy0 = static_cast<int>(std::floor(gy));

            double local_dx[16], local_dy[16];
            int kidx = 0;
            for (int b = 0; b < 4; b++) {
                const int yy = clamp_int(iy0 + b - 1, 0, map.Ny - 1);
                for (int a = 0; a < 4; a++) {
                    const int xx = clamp_int(ix0 + a - 1, 0, map.Nx - 1);
                    local_dx[kidx] = dxp[yy * map.Nx + xx];
                    local_dy[kidx] = dyp[yy * map.Nx + xx];
                    kidx++;
                }
            }

            for (int iter = 0; iter < max_newton; iter++) {
                Jet jsx(sx, 0);
                Jet jsy(sy, 1);
                Jet jgx, jgy;
                map.stereo_to_grid_coords(jsx, jsy, jgx, jgy);
                Jet ju = jgx - Jet(static_cast<double>(ix0));
                Jet jv = jgy - Jet(static_cast<double>(iy0));

                Jet wx[4], wy[4];
                cubic_bspline_basis_uniform(ju, wx);
                cubic_bspline_basis_uniform(jv, wy);

                Jet dx_val(0.0), dy_val(0.0);
                int ki = 0;
                for (int b = 0; b < 4; b++) {
                    for (int a = 0; a < 4; a++) {
                        Jet w = wy[b] * wx[a];
                        dx_val += Jet(local_dx[ki]) * w;
                        dy_val += Jet(local_dy[ki]) * w;
                        ki++;
                    }
                }

                Jet r0 = Jet(fx) * (jsx + dx_val) + Jet(cx) - Jet(target_u);
                Jet r1 = Jet(fy) * (jsy + dy_val) + Jet(cy) - Jet(target_v);
                const double res0 = r0.a;
                const double res1 = r1.a;
                if (res0 * res0 + res1 * res1 < tol_sq) {
                    break;
                }

                const double J00 = r0.v[0], J01 = r0.v[1];
                const double J10 = r1.v[0], J11 = r1.v[1];
                const double det = J00 * J11 - J01 * J10;
                if (std::abs(det) < 1e-30) {
                    break;
                }
                const double inv_det = 1.0 / det;
                sx -= inv_det * (J11 * res0 - J01 * res1);
                sy -= inv_det * (-J10 * res0 + J00 * res1);
            }

            double new_gx, new_gy;
            map.stereo_to_grid_coords(sx, sy, new_gx, new_gy);
            if (static_cast<int>(std::floor(new_gx)) == ix0 &&
                static_cast<int>(std::floor(new_gy)) == iy0) {
                break;
            }
        }

        Vec3<double> ray = stereographic_to_unit_ray(sx, sy);
        O[i * 3 + 0] = ray[0];
        O[i * 3 + 1] = ray[1];
        O[i * 3 + 2] = ray[2];
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
