#include <ceres/jet.h>
#include <pybind11/numpy.h>
#include <cmath>

#include "./cameramodels.hpp"
#include "./pybind_utils.hpp"

namespace lensboy {

namespace py = pybind11;

static Vec2<double> normalize_single_point(
    double target_u,
    double target_v,
    double fx,
    double fy,
    double cx,
    double cy,
    const double* dxp,
    const double* dyp,
    int Nx,
    int Ny,
    double half_x,
    double half_y,
    double x_scale,
    double y_scale
) {
    using Jet = ceres::Jet<double, 2>;

    constexpr int max_newton = 50;
    constexpr int max_rebuilds = 25;
    constexpr double tol_sq = 1e-20;
    constexpr double eps = 1e-12;

    // Initial guess: inverse pinhole (ignore spline distortion)
    double normalized_x = (target_u - cx) / fx;
    double normalized_y = (target_v - cy) / fy;

    for (int rebuild = 0; rebuild < max_rebuilds; rebuild++) {
        // Convert to stereographic for spline lookup
        double sx, sy;
        normalized_to_stereographic(normalized_x, normalized_y, sx, sy);

        // Compute cell index for current (normalized_x, normalized_y) in
        // stereographic space
        double gx = std::max(
            0.0,
            std::min(1.0 + (sx + half_x) * x_scale, Nx - 1.0 - eps)
        );
        double gy = std::max(
            0.0,
            std::min(1.0 + (sy + half_y) * y_scale, Ny - 1.0 - eps)
        );
        const int ix0 = (int)std::floor(gx);
        const int iy0 = (int)std::floor(gy);

        // Extract 4x4 local knot patch
        double local_dx[16], local_dy[16];
        int kidx = 0;
        for (int b = 0; b < 4; b++) {
            const int yy = clamp_int(iy0 + b - 1, 0, Ny - 1);
            for (int a = 0; a < 4; a++) {
                const int xx = clamp_int(ix0 + a - 1, 0, Nx - 1);
                local_dx[kidx] = dxp[yy * Nx + xx];
                local_dy[kidx] = dyp[yy * Nx + xx];
                kidx++;
            }
        }

        // Newton iterations with autodiff via Ceres Jets
        for (int iter = 0; iter < max_newton; iter++) {
            Jet jet_normalized_x(normalized_x);
            Jet jet_normalized_y(normalized_y);
            jet_normalized_x.v[0] = 1.0;
            jet_normalized_y.v[1] = 1.0;

            // Normalized coords -> stereographic -> spline coords
            Jet jsx, jsy;
            normalized_to_stereographic(
                jet_normalized_x,
                jet_normalized_y,
                jsx,
                jsy
            );

            Jet jgx = clamp_T(
                Jet(1.0) + (jsx + Jet(half_x)) * Jet(x_scale),
                Jet(0.0),
                Jet(Nx - 1.0 - eps)
            );
            Jet jgy = clamp_T(
                Jet(1.0) + (jsy + Jet(half_y)) * Jet(y_scale),
                Jet(0.0),
                Jet(Ny - 1.0 - eps)
            );

            // Local spline parameter within cell
            Jet ju = jgx - Jet((double)ix0);
            Jet jv = jgy - Jet((double)iy0);

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

            // Projection residual
            Jet r0 =
                Jet(fx) * (jet_normalized_x + dx_val) + Jet(cx) - Jet(target_u);
            Jet r1 =
                Jet(fy) * (jet_normalized_y + dy_val) + Jet(cy) - Jet(target_v);

            const double res0 = r0.a;
            const double res1 = r1.a;

            if (res0 * res0 + res1 * res1 < tol_sq) {
                break;
            }

            // 2x2 Jacobian from Jet dual parts
            const double J00 = r0.v[0], J01 = r0.v[1];
            const double J10 = r1.v[0], J11 = r1.v[1];

            const double det = J00 * J11 - J01 * J10;
            if (std::abs(det) < 1e-30) {
                break;
            }
            const double inv_det = 1.0 / det;

            normalized_x -= inv_det * (J11 * res0 - J01 * res1);
            normalized_y -= inv_det * (-J10 * res0 + J00 * res1);
        }

        // Check if the solution moved to a different cell
        normalized_to_stereographic(normalized_x, normalized_y, sx, sy);
        gx = std::max(
            0.0,
            std::min(1.0 + (sx + half_x) * x_scale, Nx - 1.0 - eps)
        );
        gy = std::max(
            0.0,
            std::min(1.0 + (sy + half_y) * y_scale, Ny - 1.0 - eps)
        );
        const int new_ix = (int)std::floor(gx);
        const int new_iy = (int)std::floor(gy);

        if (new_ix == ix0 && new_iy == iy0) {
            break;
        }
    }

    return Vec2<double>(normalized_x, normalized_y);
}

py::array_t<double> normalize_pinhole_splined_points(
    PinholeSplinedModelDefinition& config,
    PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> pixel_coords
) {
    auto dxb = intrinsics.dx_grid.request();
    auto dyb = intrinsics.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == config.num_knots_y &&
            (uint32_t)dxb.shape[1] == config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dyb.shape[0] == config.num_knots_y &&
            (uint32_t)dyb.shape[1] == config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    auto pinhole_params_buf = intrinsics.pinhole_parameters.request();
    require(
        pinhole_params_buf.ndim == 1 && pinhole_params_buf.shape[0] == 4,
        "pinhole_parameters must have shape (4,)"
    );
    const double* pinhole_params =
        static_cast<const double*>(pinhole_params_buf.ptr);
    const double fx = pinhole_params[0], fy = pinhole_params[1],
                 cx = pinhole_params[2], cy = pinhole_params[3];
    require(fx != 0.0 && fy != 0.0, "fx/fy must be non-zero");

    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);

    auto pb = pixel_coords.request();
    require(pb.ndim == 2, "pixel_coords must be a 2D numpy array");
    require(pb.shape[1] == 2, "pixel_coords must have shape (N, 2)");
    const ssize_t N = pb.shape[0];
    const double* P = static_cast<const double*>(pb.ptr);

    const int Nx = (int)config.num_knots_x;
    const int Ny = (int)config.num_knots_y;

    const double fov_rad_x = config.fov_deg_x * M_PI / 180.0;
    const double fov_rad_y = config.fov_deg_y * M_PI / 180.0;
    const double half_x = stereo_half_range(fov_rad_x);
    const double half_y = stereo_half_range(fov_rad_y);
    const double x_scale = (Nx - 3) / (2.0 * half_x);
    const double y_scale = (Ny - 3) / (2.0 * half_y);

    py::array_t<double> out({N, (ssize_t)3});
    auto ob = out.request();
    double* O = static_cast<double*>(ob.ptr);

    for (ssize_t i = 0; i < N; i++) {
        Vec2<double> result = normalize_single_point(
            P[i * 2 + 0],
            P[i * 2 + 1],
            fx,
            fy,
            cx,
            cy,
            dxp,
            dyp,
            Nx,
            Ny,
            half_x,
            half_y,
            x_scale,
            y_scale
        );

        O[i * 3 + 0] = result[0];
        O[i * 3 + 1] = result[1];
        O[i * 3 + 2] = 1.0;
    }

    return out;
}

}  // namespace lensboy
