#pragma once
#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <array>
#include <cmath>
#include "./type_defs.hpp"

namespace py = pybind11;
namespace lensboy {

struct PinholeSplinedModelDefinition {
    uint32_t image_width;
    uint32_t image_height;
    double fov_deg_x;
    double fov_deg_y;
    uint32_t num_knots_x;
    uint32_t num_knots_y;
};

struct PinholeSplinedOptimizationConfig : PinholeSplinedModelDefinition {
    double smoothness_lambda;
};

struct PinholeSplinedIntrinsicsParameters {
    py::array_t<double, py::array::c_style | py::array::forcecast>
        pinhole_parameters;
    py::array_t<double, py::array::c_style | py::array::forcecast> dx_grid;
    py::array_t<double, py::array::c_style | py::array::forcecast> dy_grid;
};

template <typename T>
void project_pinhole(
    const T* const intrinsics,
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    Vec3<T> normalized_point = point_in_camera / point_in_camera[2];

    T fx = intrinsics[0];
    T fy = intrinsics[1];
    T cx = intrinsics[2];
    T cy = intrinsics[3];

    result << (normalized_point[0] * fx) + cx, (normalized_point[1] * fy) + cy;
}

template <typename T>
void distort_opencv(
    const T* const distortion_parameters,
    const Vec2<T>& normalized_point,
    Vec2<T>& result
) {
    // Distortion coeffs in OpenCV order:
    // (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty)
    const T k1 = distortion_parameters[0];
    const T k2 = distortion_parameters[1];
    const T p1 = distortion_parameters[2];
    const T p2 = distortion_parameters[3];
    const T k3 = distortion_parameters[4];
    const T k4 = distortion_parameters[5];
    const T k5 = distortion_parameters[6];
    const T k6 = distortion_parameters[7];
    const T s1 = distortion_parameters[8];
    const T s2 = distortion_parameters[9];
    const T s3 = distortion_parameters[10];
    const T s4 = distortion_parameters[11];
    const T tau_x = distortion_parameters[12];
    const T tau_y = distortion_parameters[13];

    const T x = normalized_point[0];
    const T y = normalized_point[1];

    const T r2 = x * x + y * y;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;

    // OpenCV rational radial model:
    // radial = (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)
    const T radial_num = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
    const T radial_den = T(1) + k4 * r2 + k5 * r4 + k6 * r6;
    const T radial = radial_num / radial_den;

    const T x_radial = x * radial;
    const T y_radial = y * radial;

    // Tangential (Brown-Conrady, same as OpenCV)
    const T x_tan = T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
    const T y_tan = p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

    // Thin prism distortion (OpenCV s1..s4)
    const T x_prism = s1 * r2 + s2 * r4;
    const T y_prism = s3 * r2 + s4 * r4;

    const T xd = x_radial + x_tan + x_prism;
    const T yd = y_radial + y_tan + y_prism;

    // Tilt distortion (OpenCV tilted sensor model)
    // R = Rx(tau_x) * Ry(tau_y), then matTilt = matProjZ * R
    // matProjZ normalises so that z maps to 1.
    const T cTx = cos(tau_x);
    const T sTx = sin(tau_x);
    const T cTy = cos(tau_y);
    const T sTy = sin(tau_y);

    // R = Rx * Ry  (3x3 rotation)
    const T r00 = cTy;
    const T r01 = sTx * sTy;
    const T r02 = -cTx * sTy;
    const T r10 = T(0);
    const T r11 = cTx;
    const T r12 = sTx;
    const T r20 = sTy;
    const T r21 = -sTx * cTy;
    const T r22 = cTx * cTy;

    // matProjZ * R  (projects so that z-row of R becomes denominator)
    // row0: r22*r00 - r02*r20,  r22*r01 - r02*r21,  r22*r02 - r02*r22
    // row1: r22*r10 - r12*r20,  r22*r11 - r12*r21,  r22*r12 - r12*r22
    // row2:                 0,                   0,                  1
    const T t00 = r22 * r00 - r02 * r20;
    const T t01 = r22 * r01 - r02 * r21;
    const T t02 = r22 * r02 - r02 * r22;  // = 0 analytically
    const T t10 = r22 * r10 - r12 * r20;
    const T t11 = r22 * r11 - r12 * r21;
    const T t12 = r22 * r12 - r12 * r22;  // = 0 analytically

    const T w = r20 * xd + r21 * yd + r22;
    const T x_distorted = (t00 * xd + t01 * yd + t02) / w;
    const T y_distorted = (t10 * xd + t11 * yd + t12) / w;

    result << x_distorted, y_distorted;
}

template <typename T>
void project_opencv(
    const T* const
        intrinsics,  // fx, fy, cx, cy, k1, k2, p1, p2, k3..k6, s1..s4, tx, ty
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    Vec2<T> normalized(
        point_in_camera[0] / point_in_camera[2],
        point_in_camera[1] / point_in_camera[2]
    );
    Vec2<T> distorted_normalized;

    distort_opencv(intrinsics + 4, normalized, distorted_normalized);

    const T fx = intrinsics[0];
    const T fy = intrinsics[1];
    const T cx = intrinsics[2];
    const T cy = intrinsics[3];

    result << fx * distorted_normalized[0] + cx,
        fy * distorted_normalized[1] + cy;
}

inline int clamp_int(
    int v,
    int lo,
    int hi
) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Convert normalized pinhole coordinates to stereographic projection.
// Given p = (x, y) in normalized space:
//   r = |p|, theta = arctan(r), p_stereo = (p / r) * 2 * tan(theta / 2)
template <typename T>
static inline void normalized_to_stereographic(
    const T& x_normalized,
    const T& y_normalized,
    T& x_stereo,
    T& y_stereo
) {
    using std::atan;
    using std::sqrt;
    using std::tan;

    const T r_sq = x_normalized * x_normalized + y_normalized * y_normalized;
    const T r = sqrt(r_sq + T(1e-30));  // avoid division by zero
    const T theta = atan(r);
    const T scale = T(2) * tan(theta / T(2)) / r;
    x_stereo = x_normalized * scale;
    y_stereo = y_normalized * scale;
}

// Compute stereographic half-range from FOV in radians.
// At the edge: theta = fov/2, so stereo_half = 2*tan(fov/4).
inline double stereo_half_range(
    double fov_rad
) {
    return 2.0 * std::tan(fov_rad / 4.0);
}

template <typename T>
static inline void cubic_bspline_basis_uniform(
    const T& u,
    T w[4]
) {
    // u in [0,1)
    const T u2 = u * u;
    const T u3 = u2 * u;
    // weights for control indices offsets [-1,0,1,2] relative to cell index
    w[0] = (T(1) - T(3) * u + T(3) * u2 - u3) / T(6);  // (1-u)^3 / 6
    w[1] = (T(4) - T(6) * u2 + T(3) * u3) / T(6);      // (3u^3 - 6u^2 + 4)/6
    w[2] = (T(1) + T(3) * u + T(3) * u2 - T(3) * u3) /
           T(6);       // (-3u^3 + 3u^2 + 3u + 1)/6
    w[3] = u3 / T(6);  // u^3 / 6
}

template <typename T>
static inline int clamp_int(
    int v,
    int lo,
    int hi
) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename T>
static inline T clamp_T(
    const T& v,
    const T& lo,
    const T& hi
) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Overload for double: returns the value directly
inline double scalar_value(
    const double& x
) {
    return x;
}

// Jet specialization
template <typename T, int N>
inline double scalar_value(
    const ceres::Jet<T, N>& x
) {
    return x.a;
}

template <typename T>
static inline T eval_bspline2d_uniform_cubic_clamped(
    const T* grid,  // row-major, size Ny*Nx
    int Nx,
    int Ny,
    const T& x_spline,  // spline coordinate (control points at integer coords)
    const T& y_spline
) {
    // We assume "clamped by edge replication" boundary behavior:
    // indices outside [0..Nx-1]/[0..Ny-1] are clamped.
    //
    // For cubic, valid interior is [1, Nx-2) in spline coords for non-clamped;
    // but with clamping we can evaluate anywhere and it'll replicate edges.
    T gx = x_spline;
    T gy = y_spline;

    // Keep floor() stable near the upper edge if gx is exactly integer at
    // boundary. (Not strictly necessary but avoids ix == Nx-1 leading to
    // neighborhood beyond.)
    const T eps = T(1e-12);
    gx = clamp_T(gx, T(0), T(Nx - 1) - eps);
    gy = clamp_T(gy, T(0), T(Ny - 1) - eps);

    const int ix = static_cast<int>(std::floor(scalar_value(gx)));
    const int iy = static_cast<int>(std::floor(scalar_value(gy)));

    const T u = gx - T(ix);
    const T v = gy - T(iy);

    T wx[4], wy[4];
    cubic_bspline_basis_uniform(u, wx);
    cubic_bspline_basis_uniform(v, wy);

    // neighborhood indices in each dimension: (i-1 .. i+2)
    const int xs[4] = {
        clamp_int(ix - 1, 0, Nx - 1),
        clamp_int(ix + 0, 0, Nx - 1),
        clamp_int(ix + 1, 0, Nx - 1),
        clamp_int(ix + 2, 0, Nx - 1)
    };
    const int ys[4] = {
        clamp_int(iy - 1, 0, Ny - 1),
        clamp_int(iy + 0, 0, Ny - 1),
        clamp_int(iy + 1, 0, Ny - 1),
        clamp_int(iy + 2, 0, Ny - 1)
    };

    // tensor-product sum: wy^T * patch * wx
    T acc = T(0);
    for (int b = 0; b < 4; ++b) {
        const int yy = ys[b];
        const T wyb = wy[b];
        const int row0 = yy * Nx;
        for (int a = 0; a < 4; ++a) {
            const int xx = xs[a];
            acc += grid[row0 + xx] * (wyb * wx[a]);
        }
    }
    return acc;
}

struct SplineMap {
    int Nx = 0;
    int Ny = 0;
    double half_x = 0.0;
    double half_y = 0.0;
    double x_scale = 0.0;
    double y_scale = 0.0;

    explicit SplineMap(
        const PinholeSplinedModelDefinition& cfg
    ) {
        this->Nx = static_cast<int>(cfg.num_knots_x);
        this->Ny = static_cast<int>(cfg.num_knots_y);

        const double fov_rad_x = cfg.fov_deg_x * M_PI / 180.0;
        const double fov_rad_y = cfg.fov_deg_y * M_PI / 180.0;
        this->half_x = stereo_half_range(fov_rad_x);
        this->half_y = stereo_half_range(fov_rad_y);

        this->x_scale = (Nx - 3) / (2.0 * this->half_x);
        this->y_scale = (Ny - 3) / (2.0 * this->half_y);
    }

    template <typename T>
    inline void normalized_to_grid_coords(
        const T& x_n,
        const T& y_n,
        T& gx,
        T& gy
    ) const {
        T x_s, y_s;
        normalized_to_stereographic(x_n, y_n, x_s, y_s);

        const T x_s_raw = T(1.0) + (x_s + T(this->half_x)) * T(this->x_scale);
        const T y_s_raw = T(1.0) + (y_s + T(this->half_y)) * T(this->y_scale);

        constexpr double eps = 1e-12;
        gx = clamp_T(x_s_raw, T(0.0), T(Nx - 1.0 - eps));
        gy = clamp_T(y_s_raw, T(0.0), T(Ny - 1.0 - eps));
    }

    inline void project_to_spline_coords(
        const double* cam6,
        const Vec3<double>& pw,
        double& gx,
        double& gy,
        double& x_n,
        double& y_n
    ) const {
        double pc[3];
        ceres::AngleAxisRotatePoint(cam6, pw.data(), pc);
        pc[0] += cam6[3];
        pc[1] += cam6[4];
        pc[2] += cam6[5];

        const double inv_z = 1.0 / pc[2];
        x_n = pc[0] * inv_z;
        y_n = pc[1] * inv_z;

        normalized_to_grid_coords(x_n, y_n, gx, gy);
    }

    inline void cell_index(
        const double* cam6,
        const Vec3<double>& pw,
        int& ix,
        int& iy
    ) const {
        double gx, gy, xn, yn;
        project_to_spline_coords(cam6, pw, gx, gy, xn, yn);
        ix = static_cast<int>(gx);
        iy = static_cast<int>(gy);
    }

    /// Check whether the 4x4 support patch for cell (ix, iy) has all
    /// unique knot indices. Near edges, clamping causes duplicates which
    /// Ceres forbids in a single residual block.
    inline bool is_inside_fov(
        int ix,
        int iy
    ) const {
        return ix >= 1 && ix <= Nx - 3 && iy >= 1 && iy <= Ny - 3;
    }

    inline void support_indices_4x4(
        int ix,
        int iy,
        std::array<int, 16>& flat
    ) const {
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            const int yy = clamp_int(iy + b - 1, 0, Ny - 1);
            for (int a = 0; a < 4; a++) {
                const int xx = clamp_int(ix + a - 1, 0, Nx - 1);
                flat[idx++] = yy * Nx + xx;
            }
        }
    }
};

template <typename T>
void project_pinhole_splined(
    PinholeSplinedModelDefinition* config,
    const T* const pinhole_parameters,  // fx, fy, cx, cy
    const T* const dx_grid,
    const T* const dy_grid,
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    const T x_normalized = point_in_camera[0] / point_in_camera[2];
    const T y_normalized = point_in_camera[1] / point_in_camera[2];

    const SplineMap map(*config);

    T x_spline, y_spline;
    map.normalized_to_grid_coords(
        x_normalized,
        y_normalized,
        x_spline,
        y_spline
    );

    const T dx = eval_bspline2d_uniform_cubic_clamped(
        dx_grid,
        map.Nx,
        map.Ny,
        x_spline,
        y_spline
    );
    const T dy = eval_bspline2d_uniform_cubic_clamped(
        dy_grid,
        map.Nx,
        map.Ny,
        x_spline,
        y_spline
    );

    const T fx = pinhole_parameters[0];
    const T fy = pinhole_parameters[1];
    const T cx = pinhole_parameters[2];
    const T cy = pinhole_parameters[3];

    result[0] = fx * (x_normalized + dx) + cx;
    result[1] = fy * (y_normalized + dy) + cy;
}

struct KnotSmoothness {
    double s;
    template <typename T>
    bool operator()(
        const T* const a,
        const T* const b,
        const T* const c,
        const T* const d,
        T* residuals
    ) const {
        // Third derivative: -a + 3b - 3c + d
        residuals[0] = T(s) * (-a[0] + T(3.0) * b[0] - T(3.0) * c[0] + d[0]);
        return true;
    }
};

}  // namespace lensboy
