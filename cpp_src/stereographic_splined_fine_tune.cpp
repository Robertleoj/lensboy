#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <thread>
#include <tuple>
#include <vector>

#include "./calibrate.hpp"
#include "./cameramodels.hpp"
#include "./ceres_geometry.hpp"
#include "./pybind_utils.hpp"
#include "./type_defs.hpp"

namespace lensboy {

static bool configure_sparse_schur_if_available(
    ceres::Solver::Options& options
) {
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        return true;
    }

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        return true;
    }

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::ACCELERATE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::ACCELERATE_SPARSE;
        return true;
    }

    return false;
}

struct StereographicTargetWarpBasis {
    Vec3<double> center;
    Vec3<double> x_hat;
    Vec3<double> y_hat;
    Vec3<double> z_hat;
    double inv_x_scale;
    double inv_y_scale;
};

static StereographicTargetWarpBasis make_target_warp_basis(
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

    return StereographicTargetWarpBasis{
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
    const StereographicTargetWarpBasis& warp,
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

struct ReprojectionErrorStereographicSplinedWarp {
    const StereographicSplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    int ix0, iy0;
    double obs_x, obs_y;
    StereographicTargetWarpBasis warp_basis;

    // clang-format off
    template <typename T>
    bool operator()(
        const T* const cam, const T* const warp_coeffs,
        const T* const k00, const T* const k01, const T* const k02, const T* const k03,
        const T* const k04, const T* const k05, const T* const k06, const T* const k07,
        const T* const k08, const T* const k09, const T* const k10, const T* const k11,
        const T* const k12, const T* const k13, const T* const k14, const T* const k15,
        T* residuals
    ) const {
        const T* knots[16] = {k00, k01, k02, k03, k04, k05, k06, k07,
                              k08, k09, k10, k11, k12, k13, k14, k15};
        // clang-format on

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

        const Vec3<T> ray = normalize_to_unit_ray(Vec3<T>(pc[0], pc[1], pc[2]));
        T x_stereo, y_stereo;
        unit_ray_to_stereographic(ray, x_stereo, y_stereo);

        T gx, gy;
        map.stereo_to_grid_coords(x_stereo, y_stereo, gx, gy);
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

        residuals[0] = T(fx) * (x_stereo + dx_val) + T(cx) - T(obs_x);
        residuals[1] = T(fy) * (y_stereo + dy_val) + T(cy) - T(obs_y);
        return true;
    }
};

struct ReprojectionErrorStereographicSplinedNoWarp {
    const StereographicSplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    int ix0, iy0;
    double obs_x, obs_y;

    // clang-format off
    template <typename T>
    bool operator()(
        const T* const cam,
        const T* const k00, const T* const k01, const T* const k02, const T* const k03,
        const T* const k04, const T* const k05, const T* const k06, const T* const k07,
        const T* const k08, const T* const k09, const T* const k10, const T* const k11,
        const T* const k12, const T* const k13, const T* const k14, const T* const k15,
        T* residuals
    ) const {
        const T* knots[16] = {k00, k01, k02, k03, k04, k05, k06, k07,
                              k08, k09, k10, k11, k12, k13, k14, k15};
        // clang-format on

        T pw_t[3] = {T(pw[0]), T(pw[1]), T(pw[2])};
        T pc[3];
        ceres::AngleAxisRotatePoint(cam, pw_t, pc);
        pc[0] += cam[3];
        pc[1] += cam[4];
        pc[2] += cam[5];

        const Vec3<T> ray = normalize_to_unit_ray(Vec3<T>(pc[0], pc[1], pc[2]));
        T x_stereo, y_stereo;
        unit_ray_to_stereographic(ray, x_stereo, y_stereo);

        T gx, gy;
        map.stereo_to_grid_coords(x_stereo, y_stereo, gx, gy);
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

        residuals[0] = T(fx) * (x_stereo + dx_val) + T(cx) - T(obs_x);
        residuals[1] = T(fy) * (y_stereo + dy_val) + T(cy) - T(obs_y);
        return true;
    }
};

struct StereographicKnotSmoothness2D {
    double weight;

    template <typename T>
    bool operator()(
        const T* const a,
        const T* const b,
        const T* const c,
        const T* const d,
        T* residuals
    ) const {
        residuals[0] =
            T(weight) * (-a[0] + T(3.0) * b[0] - T(3.0) * c[0] + d[0]);
        residuals[1] =
            T(weight) * (-a[1] + T(3.0) * b[1] - T(3.0) * c[1] + d[1]);
        return true;
    }
};

struct StereographicObservationRecord {
    size_t cam_idx;
    int pt_idx;
    int ix;
    int iy;
};

static bool any_cell_changed(
    const StereographicSplineMap& map,
    const std::vector<Vec6<double>>& cams,
    const std::vector<Vec3<double>>& pts,
    const std::vector<StereographicObservationRecord>& obs
) {
    for (auto& r : obs) {
        int ix, iy;
        map.cell_index(cams[r.cam_idx].data(), pts[r.pt_idx], ix, iy);
        if (ix != r.ix || iy != r.iy) {
            return true;
        }
    }
    return false;
}

static void add_grid_smoothness(
    ceres::Problem& problem,
    const std::vector<double*>& knot_blocks,
    int nx,
    int ny,
    double weight
) {
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x + 3 < nx; x++) {
            const int k0 = y * nx + x;
            const int k1 = y * nx + x + 1;
            const int k2 = y * nx + x + 2;
            const int k3 = y * nx + x + 3;
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<
                    StereographicKnotSmoothness2D,
                    2,
                    2,
                    2,
                    2,
                    2>(new StereographicKnotSmoothness2D{weight}),
                nullptr,
                knot_blocks[k0],
                knot_blocks[k1],
                knot_blocks[k2],
                knot_blocks[k3]
            );
        }
    }

    for (int y = 0; y + 3 < ny; y++) {
        for (int x = 0; x < nx; x++) {
            const int k0 = y * nx + x;
            const int k1 = (y + 1) * nx + x;
            const int k2 = (y + 2) * nx + x;
            const int k3 = (y + 3) * nx + x;
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<
                    StereographicKnotSmoothness2D,
                    2,
                    2,
                    2,
                    2,
                    2>(new StereographicKnotSmoothness2D{weight}),
                nullptr,
                knot_blocks[k0],
                knot_blocks[k1],
                knot_blocks[k2],
                knot_blocks[k3]
            );
        }
    }
}

static void build_problem(
    ceres::Problem& problem,
    const StereographicSplinedOptimizationConfig& cfg,
    const StereographicSplineMap& map,
    const double* params,
    std::vector<Vec2<double>>& knot_params,
    double* warp_coeffs,
    const std::optional<WarpCoordinates>& warp_coordinates,
    const std::vector<Vec6<double>>& cameras_from_target,
    const std::vector<Vec3<double>>& target_points,
    const std::vector<
        std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>& frames,
    std::vector<double*>& knot_blocks,
    std::vector<StereographicObservationRecord>& obs_records,
    double smoothness_weight
) {
    const int nx = static_cast<int>(cfg.num_knots_x);
    const int ny = static_cast<int>(cfg.num_knots_y);
    const int n_knots = nx * ny;
    const bool has_warp = warp_coordinates.has_value();
    StereographicTargetWarpBasis warp_basis{};
    if (has_warp) {
        warp_basis = make_target_warp_basis(*warp_coordinates);
        problem.AddParameterBlock(warp_coeffs, 5);
    }

    knot_blocks.resize(n_knots);
    for (int i = 0; i < n_knots; i++) {
        knot_blocks[i] = knot_params[i].data();
        problem.AddParameterBlock(knot_blocks[i], 2);
    }

    for (auto& cam : cameras_from_target) {
        problem.AddParameterBlock(const_cast<double*>(cam.data()), 6);
    }

    obs_records.clear();
    for (size_t cam_idx = 0; cam_idx < frames.size(); cam_idx++) {
        auto& ids = std::get<0>(frames[cam_idx]);
        auto& observations = std::get<1>(frames[cam_idx]);
        auto& cam6 = cameras_from_target[cam_idx];

        for (size_t obs_idx = 0; obs_idx < observations.size(); obs_idx++) {
            const int pt_idx = ids[obs_idx];
            const auto& pw = target_points[pt_idx];
            const auto& obs = observations[obs_idx];

            int ix, iy;
            map.cell_index(cam6.data(), pw, ix, iy);
            if (!map.is_inside_fov(ix, iy)) {
                continue;
            }

            std::array<int, 16> flat{};
            map.support_indices_4x4(ix, iy, flat);

            // clang-format off
            std::array<double*, 18> blocks{};
            blocks[0] = const_cast<double*>(cam6.data());
            blocks[1] = warp_coeffs;
            for (int i = 0; i < 16; i++) { blocks[2 + i] = knot_blocks[flat[i]]; }

            if (has_warp) {
                auto* cost = new ceres::AutoDiffCostFunction<
                    ReprojectionErrorStereographicSplinedWarp, 2, 6, 5,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2
                >(new ReprojectionErrorStereographicSplinedWarp{
                    map, params[0], params[1], params[2], params[3], pw, ix, iy, obs(0, 0), obs(1, 0), warp_basis
                });

                problem.AddResidualBlock(cost, nullptr,
                    blocks[0],  blocks[1],
                    blocks[2],  blocks[3],  blocks[4],  blocks[5],  blocks[6],  blocks[7],  blocks[8],  blocks[9],
                    blocks[10], blocks[11], blocks[12], blocks[13], blocks[14], blocks[15], blocks[16], blocks[17]
                );
            } else {
                auto* cost = new ceres::AutoDiffCostFunction<
                    ReprojectionErrorStereographicSplinedNoWarp, 2, 6,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2
                >(new ReprojectionErrorStereographicSplinedNoWarp{
                    map, params[0], params[1], params[2], params[3], pw, ix, iy, obs(0, 0), obs(1, 0)
                });

                problem.AddResidualBlock(cost, nullptr,
                    blocks[0],
                    blocks[2],  blocks[3],  blocks[4],  blocks[5],  blocks[6],  blocks[7],  blocks[8],  blocks[9],
                    blocks[10], blocks[11], blocks[12], blocks[13], blocks[14], blocks[15], blocks[16], blocks[17]
                );
            }
            // clang-format on

            obs_records.push_back(
                StereographicObservationRecord{cam_idx, pt_idx, ix, iy}
            );
        }
    }

    if (smoothness_weight > 0.0) {
        add_grid_smoothness(problem, knot_blocks, nx, ny, smoothness_weight);
    }
}

py::dict fine_tune_stereographic_splined(
    StereographicSplinedOptimizationConfig& model_config,
    StereographicSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        frames,
    std::optional<WarpCoordinates> warp_coordinates,
    std::array<double, 5> warp_coeffs_initial
) {
    auto dxb = intrinsics_parameters.dx_grid.request();
    auto dyb = intrinsics_parameters.dy_grid.request();
    require(
        static_cast<uint32_t>(dxb.shape[0]) == model_config.num_knots_y &&
            static_cast<uint32_t>(dxb.shape[1]) == model_config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        static_cast<uint32_t>(dyb.shape[0]) == model_config.num_knots_y &&
            static_cast<uint32_t>(dyb.shape[1]) == model_config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    auto params_buf = intrinsics_parameters.stereographic_parameters.request();
    require(
        params_buf.ndim == 1 && params_buf.shape[0] == 4,
        "stereographic_parameters must have shape (4,)"
    );
    const double* params = static_cast<const double*>(params_buf.ptr);
    double* dxp = static_cast<double*>(dxb.ptr);
    double* dyp = static_cast<double*>(dyb.ptr);

    const int nx = static_cast<int>(model_config.num_knots_x);
    const int ny = static_cast<int>(model_config.num_knots_y);
    const int n_knots = nx * ny;
    std::vector<Vec2<double>> knot_params(n_knots);
    for (int i = 0; i < n_knots; i++) {
        knot_params[i] = Vec2<double>(dxp[i], dyp[i]);
    }

    double warp_coeffs[5] = {
        warp_coeffs_initial[0],
        warp_coeffs_initial[1],
        warp_coeffs_initial[2],
        warp_coeffs_initial[3],
        warp_coeffs_initial[4]
    };

    const StereographicSplineMap map(model_config);
    const double smoothness_weight = std::sqrt(model_config.smoothness_lambda);

    ceres::Solver::Options options;
    options.num_threads =
        std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
    if (!configure_sparse_schur_if_available(options)) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
    }
    options.minimizer_progress_to_stdout = false;

    constexpr int max_rebuilds = 1000;
    std::vector<double*> knot_blocks;
    std::vector<StereographicObservationRecord> obs_records;
    ceres::Solver::Summary last_summary;

    double prev_cost = std::numeric_limits<double>::max();
    int outer;
    for (outer = 0; outer < max_rebuilds; outer++) {
        ceres::Problem problem;
        build_problem(
            problem,
            model_config,
            map,
            params,
            knot_params,
            warp_coeffs,
            warp_coordinates,
            cameras_from_target,
            target_points,
            frames,
            knot_blocks,
            obs_records,
            smoothness_weight
        );

        ceres::Solve(options, &problem, &last_summary);

        if (!any_cell_changed(map, cameras_from_target, target_points, obs_records)) {
            break;
        }

        const double cost = last_summary.final_cost;
        const double rel_improvement = (prev_cost - cost) / (prev_cost + 1e-30);
        if (outer > 0 && rel_improvement < 1e-6) {
            break;
        }
        prev_cost = cost;
    }
    spdlog::debug("Optimization done after {} rebuilds", outer);
    spdlog::debug(last_summary.FullReport());

    for (int i = 0; i < n_knots; i++) {
        dxp[i] = knot_params[i][0];
        dyp[i] = knot_params[i][1];
    }

    py::dict out;
    out["dx_grid"] = intrinsics_parameters.dx_grid;
    out["dy_grid"] = intrinsics_parameters.dy_grid;
    out["warp_coeffs"] = py::array_t<double>(5, warp_coeffs);

    std::vector<std::vector<double>> poses_out;
    poses_out.reserve(cameras_from_target.size());
    for (auto& cam : cameras_from_target) {
        poses_out.push_back(
            std::vector<double>(cam.data(), cam.data() + cam.size())
        );
    }
    out["cameras_from_target"] = poses_out;
    return out;
}

}  // namespace lensboy
