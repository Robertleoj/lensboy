#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include "./cameramodels.hpp"
#include "./splined_fine_tune_utils.hpp"
#include "./type_defs.hpp"

namespace lensboy {

constexpr int SPLINE_SUPPORT_WIDTH = 4;
constexpr int SPLINE_SUPPORT_SIZE = SPLINE_SUPPORT_WIDTH * SPLINE_SUPPORT_WIDTH;

struct TargetWarpBasis {
    Vec3<double> center;
    Vec3<double> x_hat;
    Vec3<double> y_hat;
    Vec3<double> z_hat;
    double inv_x_scale;
    double inv_y_scale;
};

inline TargetWarpBasis make_target_warp_basis(
    const WarpCoordinates& warp
) {
    const Vec3<double> rv = warp.target_from_warp_frame.head<3>();
    const double angle = rv.norm();
    Eigen::Matrix3d R;
    if (angle < 1e-10) {
        R = Eigen::Matrix3d::Identity();
    } else {
        R = Eigen::AngleAxisd(angle, rv / angle).toRotationMatrix();
    }

    return TargetWarpBasis{
        warp.target_from_warp_frame.tail<3>(),
        R.col(0),
        R.col(1),
        R.col(2),
        1.0 / warp.x_scale,
        1.0 / warp.y_scale,
    };
}

template <typename T>
Vec3<T> apply_target_warp_with_basis(
    const Vec3<T>& p_target,
    const TargetWarpBasis& warp,
    const T* const coeffs
) {
    const Vec3<T> d = p_target - warp.center.cast<T>();

    const T wx = T(warp.x_hat[0]) * d[0] + T(warp.x_hat[1]) * d[1] +
                 T(warp.x_hat[2]) * d[2];
    const T wy = T(warp.y_hat[0]) * d[0] + T(warp.y_hat[1]) * d[1] +
                 T(warp.y_hat[2]) * d[2];
    const T wz = T(warp.z_hat[0]) * d[0] + T(warp.z_hat[1]) * d[1] +
                 T(warp.z_hat[2]) * d[2];

    const T xs = wx * T(warp.inv_x_scale);
    const T ys = wy * T(warp.inv_y_scale);

    const T xs2 = xs * xs;
    const T ys2 = ys * ys;
    const T p2x = T(0.5) * (T(3.0) * xs2 - T(1.0));
    const T p2y = T(0.5) * (T(3.0) * ys2 - T(1.0));
    const T p4x = T(0.125) * (T(35.0) * xs2 * xs2 - T(30.0) * xs2 + T(3.0));
    const T p4y = T(0.125) * (T(35.0) * ys2 * ys2 - T(30.0) * ys2 + T(3.0));

    const T z_warp = coeffs[0] * p2x + coeffs[1] * p2y + coeffs[2] * p2x * p2y +
                     coeffs[3] * p4x + coeffs[4] * p4y;

    Vec3<T> result = warp.center.cast<T>();
    result[0] += T(warp.x_hat[0]) * wx + T(warp.y_hat[0]) * wy +
                 T(warp.z_hat[0]) * (wz + z_warp);
    result[1] += T(warp.x_hat[1]) * wx + T(warp.y_hat[1]) * wy +
                 T(warp.z_hat[1]) * (wz + z_warp);
    result[2] += T(warp.x_hat[2]) * wx + T(warp.y_hat[2]) * wy +
                 T(warp.z_hat[2]) * (wz + z_warp);
    return result;
}

struct ReprojectionErrorSplinedWarp {
    const SplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    int ix0, iy0;
    double obs_x, obs_y;
    TargetWarpBasis warp_basis;

    template <typename T>
    bool operator()(
        const T* const cam,
        const T* const warp_coeffs,
        const T* const k00,
        const T* const k01,
        const T* const k02,
        const T* const k03,
        const T* const k04,
        const T* const k05,
        const T* const k06,
        const T* const k07,
        const T* const k08,
        const T* const k09,
        const T* const k10,
        const T* const k11,
        const T* const k12,
        const T* const k13,
        const T* const k14,
        const T* const k15,
        T* residuals
    ) const {
        const T* knots[16] = {k00, k01, k02, k03, k04, k05, k06, k07,
                              k08, k09, k10, k11, k12, k13, k14, k15};

        Vec3<T> pw_warped = apply_target_warp_with_basis(
            Vec3<T>(pw.cast<T>()),
            warp_basis,
            warp_coeffs
        );

        T pw_t[3] = {pw_warped[0], pw_warped[1], pw_warped[2]};
        T pc[3];
        ceres::AngleAxisRotatePoint(cam, pw_t, pc);
        pc[0] += cam[3];
        pc[1] += cam[4];
        pc[2] += cam[5];

        const T inv_z = T(1.0) / pc[2];
        const T x_n = pc[0] * inv_z;
        const T y_n = pc[1] * inv_z;

        T x_st, y_st;
        normalized_to_stereographic(x_n, y_n, x_st, y_st);

        const T x_s = T(1.0) + (x_st + T(map.half_x)) * T(map.x_scale);
        const T y_s = T(1.0) + (y_st + T(map.half_y)) * T(map.y_scale);
        constexpr double eps = 1e-12;
        const T gx = clamp_T(x_s, T(0.0), T(map.Nx - 1.0 - eps));
        const T gy = clamp_T(y_s, T(0.0), T(map.Ny - 1.0 - eps));

        const T u = gx - T(static_cast<double>(ix0));
        const T v = gy - T(static_cast<double>(iy0));

        T wx[4], wy[4];
        cubic_bspline_basis_uniform(u, wx);
        cubic_bspline_basis_uniform(v, wy);

        T dx_val(0.0), dy_val(0.0);
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                const T w = wy[b] * wx[a];
                dx_val += knots[idx][0] * w;
                dy_val += knots[idx][1] * w;
                idx++;
            }
        }

        residuals[0] = T(fx) * (x_n + dx_val) + T(cx) - T(obs_x);
        residuals[1] = T(fy) * (y_n + dy_val) + T(cy) - T(obs_y);
        return true;
    }
};

struct ReprojectionErrorSplinedNoWarp {
    const SplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    int ix0, iy0;
    double obs_x, obs_y;

    template <typename T>
    bool operator()(
        const T* const cam,
        const T* const k00,
        const T* const k01,
        const T* const k02,
        const T* const k03,
        const T* const k04,
        const T* const k05,
        const T* const k06,
        const T* const k07,
        const T* const k08,
        const T* const k09,
        const T* const k10,
        const T* const k11,
        const T* const k12,
        const T* const k13,
        const T* const k14,
        const T* const k15,
        T* residuals
    ) const {
        const T* knots[16] = {k00, k01, k02, k03, k04, k05, k06, k07,
                              k08, k09, k10, k11, k12, k13, k14, k15};

        T pw_t[3] = {T(pw[0]), T(pw[1]), T(pw[2])};
        T pc[3];
        ceres::AngleAxisRotatePoint(cam, pw_t, pc);
        pc[0] += cam[3];
        pc[1] += cam[4];
        pc[2] += cam[5];

        const T inv_z = T(1.0) / pc[2];
        const T x_n = pc[0] * inv_z;
        const T y_n = pc[1] * inv_z;

        T x_st, y_st;
        normalized_to_stereographic(x_n, y_n, x_st, y_st);

        const T x_s = T(1.0) + (x_st + T(map.half_x)) * T(map.x_scale);
        const T y_s = T(1.0) + (y_st + T(map.half_y)) * T(map.y_scale);
        constexpr double eps = 1e-12;
        const T gx = clamp_T(x_s, T(0.0), T(map.Nx - 1.0 - eps));
        const T gy = clamp_T(y_s, T(0.0), T(map.Ny - 1.0 - eps));

        const T u = gx - T(static_cast<double>(ix0));
        const T v = gy - T(static_cast<double>(iy0));

        T wx[4], wy[4];
        cubic_bspline_basis_uniform(u, wx);
        cubic_bspline_basis_uniform(v, wy);

        T dx_val(0.0), dy_val(0.0);
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                const T w = wy[b] * wx[a];
                dx_val += knots[idx][0] * w;
                dy_val += knots[idx][1] * w;
                idx++;
            }
        }

        residuals[0] = T(fx) * (x_n + dx_val) + T(cx) - T(obs_x);
        residuals[1] = T(fy) * (y_n + dy_val) + T(cy) - T(obs_y);
        return true;
    }
};

struct ObservationRecord {
    size_t cam_idx;
    int pt_idx;
    double obs_x;
    double obs_y;
    int ix;
    int iy;
};

struct SplineResidualRecord {
    ceres::ResidualBlockId residual_id;
    int residual_size;
    size_t frame_idx;
    bool empirical;
    std::vector<int> columns;
    std::vector<int> block_sizes;
};

inline bool any_spline_cell_changed(
    const SplineMap& map,
    const std::vector<Vec6<double>>& cams,
    const std::vector<Vec3<double>>& pts,
    const std::vector<ObservationRecord>& obs
) {
    for (const auto& r : obs) {
        int nix, niy;
        map.cell_index(cams[r.cam_idx].data(), pts[r.pt_idx], nix, niy);
        if (nix != r.ix || niy != r.iy) {
            return true;
        }
    }
    return false;
}

inline void add_recorded_spline_anchor(
    ceres::Problem& problem,
    const SplineMap& map,
    const std::vector<double*>& knot_blocks,
    std::vector<SplineResidualRecord>* residual_records,
    double x_n,
    double y_n,
    bool constrain_dx,
    bool constrain_dy,
    double weight
) {
    double gx, gy;
    map.normalized_to_grid_coords(x_n, y_n, gx, gy);
    const int ix = static_cast<int>(gx);
    const int iy = static_cast<int>(gy);
    if (!map.is_inside_fov(ix, iy)) {
        return;
    }

    double wx[4], wy[4];
    cubic_bspline_basis_uniform(gx - ix, wx);
    cubic_bspline_basis_uniform(gy - iy, wy);

    double basis[SPLINE_SUPPORT_SIZE];
    int basis_index = 0;
    for (int b = 0; b < 4; b++) {
        for (int a = 0; a < 4; a++) {
            basis[basis_index++] = wy[b] * wx[a];
        }
    }

    std::array<int, SPLINE_SUPPORT_SIZE> flat{};
    map.support_indices_4x4(ix, iy, flat);

    auto make_anchor = [&](int component) {
        SplineAnchorResidual anchor{weight, 0.0, component, {}};
        std::copy(basis, basis + SPLINE_SUPPORT_SIZE, anchor.basis);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<SplineAnchorResidual>(
            new SplineAnchorResidual(anchor)
        );
        std::vector<double*> blocks;
        std::vector<int> columns;
        for (int i = 0; i < SPLINE_SUPPORT_SIZE; i++) {
            cost->AddParameterBlock(2);
            blocks.push_back(knot_blocks[flat[i]]);
            columns.push_back(2 * flat[i]);
            columns.push_back(2 * flat[i] + 1);
        }
        cost->SetNumResiduals(1);
        const ceres::ResidualBlockId residual_id =
            problem.AddResidualBlock(cost, nullptr, blocks);
        if (residual_records != nullptr) {
            residual_records->push_back(
                SplineResidualRecord{
                    residual_id,
                    1,
                    0,
                    false,
                    columns,
                    std::vector<int>(SPLINE_SUPPORT_SIZE, 2)
                }
            );
        }
    };

    if (constrain_dx) {
        make_anchor(0);
    }
    if (constrain_dy) {
        make_anchor(1);
    }
}

inline void add_recorded_grid_smoothness(
    ceres::Problem& problem,
    const std::vector<double*>& knot_blocks,
    std::vector<SplineResidualRecord>* residual_records,
    int nx,
    int ny,
    double weight
) {
    auto add_block = [&](int k0, int k1, int k2, int k3) {
        const ceres::ResidualBlockId residual_id = problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<
                KnotSmoothness2DResidual,
                2,
                2,
                2,
                2,
                2>(new KnotSmoothness2DResidual{weight}),
            nullptr,
            knot_blocks[k0],
            knot_blocks[k1],
            knot_blocks[k2],
            knot_blocks[k3]
        );
        if (residual_records == nullptr) {
            return;
        }
        residual_records->push_back(SplineResidualRecord{
            residual_id,
            2,
            0,
            false,
            {2 * k0, 2 * k0 + 1, 2 * k1, 2 * k1 + 1, 2 * k2, 2 * k2 + 1,
             2 * k3, 2 * k3 + 1},
            {2, 2, 2, 2}
        });
    };

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x + 3 < nx; x++) {
            add_block(y * nx + x, y * nx + x + 1, y * nx + x + 2, y * nx + x + 3);
        }
    }

    for (int y = 0; y + 3 < ny; y++) {
        for (int x = 0; x < nx; x++) {
            add_block(y * nx + x, (y + 1) * nx + x, (y + 2) * nx + x, (y + 3) * nx + x);
        }
    }
}

inline void build_pinhole_splined_problem(
    ceres::Problem& problem,
    const PinholeSplinedOptimizationConfig& cfg,
    const SplineMap& map,
    const double* pinhole_params,
    std::vector<Vec2<double>>& knot_params,
    double* warp_coeffs,
    const std::optional<WarpCoordinates>& warp_coordinates,
    const std::vector<Vec6<double>>& cameras_from_target,
    const std::vector<Vec3<double>>& target_points,
    const std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::vector<double*>& knot_blocks,
    std::vector<ObservationRecord>& obs_records,
    std::vector<SplineResidualRecord>* residual_records,
    double smoothness_weight
) {
    const int normalized_x = static_cast<int>(cfg.num_knots_x);
    const int normalized_y = static_cast<int>(cfg.num_knots_y);
    const int n_knots = normalized_x * normalized_y;
    const bool has_warp = warp_coordinates.has_value();
    TargetWarpBasis warp_basis{};
    if (has_warp) {
        warp_basis = make_target_warp_basis(*warp_coordinates);
    }

    if (has_warp) {
        problem.AddParameterBlock(warp_coeffs, 5);
    }

    knot_blocks.resize(n_knots);
    for (int i = 0; i < n_knots; i++) {
        knot_blocks[i] = knot_params[i].data();
        problem.AddParameterBlock(knot_blocks[i], 2);
    }

    for (const auto& cam : cameras_from_target) {
        problem.AddParameterBlock(const_cast<double*>(cam.data()), 6);
    }

    obs_records.clear();
    if (residual_records != nullptr) {
        residual_records->clear();
    }

    constexpr double anchor_weight = 1000.0;
    add_recorded_spline_anchor(
        problem,
        map,
        knot_blocks,
        residual_records,
        0.0,
        0.0,
        true,
        true,
        anchor_weight
    );
    const double fov_rad_x = cfg.fov_deg_x * M_PI / 180.0;
    const double quarter_x_n = std::tan(fov_rad_x / 4.0);
    add_recorded_spline_anchor(
        problem,
        map,
        knot_blocks,
        residual_records,
        quarter_x_n,
        0.0,
        false,
        true,
        anchor_weight
    );

    for (size_t cam_idx = 0; cam_idx < frames.size(); cam_idx++) {
        const auto& ids = std::get<0>(frames[cam_idx]);
        const auto& obs = std::get<1>(frames[cam_idx]);
        const auto& cam6 = cameras_from_target[cam_idx];

        for (size_t oi = 0; oi < ids.size(); oi++) {
            const int pt_idx = ids[oi];
            const auto& pw = target_points[pt_idx];
            const double ox = obs[oi](0, 0);
            const double oy = obs[oi](1, 0);

            int ix, iy;
            map.cell_index(cam6.data(), pw, ix, iy);
            if (!map.is_inside_fov(ix, iy)) {
                continue;
            }

            std::array<int, SPLINE_SUPPORT_SIZE> flat{};
            map.support_indices_4x4(ix, iy, flat);

            std::array<double*, 18> blocks{};
            blocks[0] = const_cast<double*>(cam6.data());
            blocks[1] = warp_coeffs;
            for (int i = 0; i < SPLINE_SUPPORT_SIZE; i++) {
                blocks[2 + i] = knot_blocks[flat[i]];
            }

            ceres::ResidualBlockId residual_id;
            std::vector<int> columns;
            std::vector<int> block_sizes;
            for (int i = 0; i < 6; i++) {
                columns.push_back(2 * n_knots + 6 * static_cast<int>(cam_idx) + i);
            }

            if (has_warp) {
                block_sizes = {6, 5};
                auto* cost = new ceres::AutoDiffCostFunction<
                    ReprojectionErrorSplinedWarp,
                    2,
                    6,
                    5,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2>(new ReprojectionErrorSplinedWarp{
                    map, pinhole_params[0], pinhole_params[1], pinhole_params[2],
                    pinhole_params[3], pw, ix, iy, ox, oy, warp_basis
                });

                residual_id = problem.AddResidualBlock(
                    cost,
                    nullptr,
                    blocks[0],
                    blocks[1],
                    blocks[2],
                    blocks[3],
                    blocks[4],
                    blocks[5],
                    blocks[6],
                    blocks[7],
                    blocks[8],
                    blocks[9],
                    blocks[10],
                    blocks[11],
                    blocks[12],
                    blocks[13],
                    blocks[14],
                    blocks[15],
                    blocks[16],
                    blocks[17]
                );
                for (int i = 0; i < 5; i++) {
                    columns.push_back(2 * n_knots + 6 * static_cast<int>(frames.size()) + i);
                }
            } else {
                block_sizes = {6};
                auto* cost = new ceres::AutoDiffCostFunction<
                    ReprojectionErrorSplinedNoWarp,
                    2,
                    6,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2>(new ReprojectionErrorSplinedNoWarp{
                    map, pinhole_params[0], pinhole_params[1], pinhole_params[2],
                    pinhole_params[3], pw, ix, iy, ox, oy
                });

                residual_id = problem.AddResidualBlock(
                    cost,
                    nullptr,
                    blocks[0],
                    blocks[2],
                    blocks[3],
                    blocks[4],
                    blocks[5],
                    blocks[6],
                    blocks[7],
                    blocks[8],
                    blocks[9],
                    blocks[10],
                    blocks[11],
                    blocks[12],
                    blocks[13],
                    blocks[14],
                    blocks[15],
                    blocks[16],
                    blocks[17]
                );
            }

            for (int i = 0; i < SPLINE_SUPPORT_SIZE; i++) {
                columns.push_back(2 * flat[i]);
                columns.push_back(2 * flat[i] + 1);
                block_sizes.push_back(2);
            }
            if (residual_records != nullptr) {
                residual_records->push_back(SplineResidualRecord{
                    residual_id,
                    2,
                    cam_idx,
                    true,
                    columns,
                    block_sizes
                });
            }
            obs_records.push_back(ObservationRecord{cam_idx, pt_idx, ox, oy, ix, iy});
        }
    }

    add_recorded_grid_smoothness(
        problem,
        knot_blocks,
        residual_records,
        normalized_x,
        normalized_y,
        smoothness_weight
    );
}

}  // namespace lensboy
