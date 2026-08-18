#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
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

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::ACCELERATE_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::ACCELERATE_SPARSE;
        return true;
    }

    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        return true;
    }

    return false;
}

struct TargetWarpBasis {
    Vec3<double> center;
    Vec3<double> x_hat;
    Vec3<double> y_hat;
    Vec3<double> z_hat;
    double inv_x_scale;
    double inv_y_scale;
};

static TargetWarpBasis make_target_warp_basis(
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

// Constrains the spline correction at a fixed point to a target value.
// Precomputed basis weights make this a simple weighted sum of 16 knots.
struct SplineAnchor {
    double weight;
    double target;
    int component;
    double basis[16];

    template <typename T>
    bool operator()(
        T const* const* knots,
        T* residuals
    ) const {
        T val(0.0);
        for (int i = 0; i < 16; i++) {
            val += knots[i][component] * T(basis[i]);
        }
        residuals[0] = T(weight) * (val - T(target));
        return true;
    }
};

struct ReprojectionErrorSplinedWarp {
    const SplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    int ix0, iy0;
    double obs_x, obs_y;
    TargetWarpBasis warp_basis;

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

struct KnotSmoothness2D {
    double s;

    template <typename T>
    bool operator()(
        const T* const a,
        const T* const b,
        const T* const c,
        const T* const d,
        T* residuals
    ) const {
        residuals[0] = T(s) * (-a[0] + T(3.0) * b[0] - T(3.0) * c[0] + d[0]);
        residuals[1] = T(s) * (-a[1] + T(3.0) * b[1] - T(3.0) * c[1] + d[1]);
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

static bool any_cell_changed(
    const SplineMap& map,
    const std::vector<Vec6<double>>& cams,
    const std::vector<Vec3<double>>& pts,
    const std::vector<ObservationRecord>& obs
) {
    for (auto& r : obs) {
        int nix, niy;
        map.cell_index(cams[r.cam_idx].data(), pts[r.pt_idx], nix, niy);
        if (nix != r.ix || niy != r.iy) {
            return true;
        }
    }
    return false;
}

static inline void BuildProblem(
    ceres::Problem& problem,
    const PinholeSplinedOptimizationConfig& cfg,
    const SplineMap& map,
    const double* pinhole_params,
    std::vector<Vec2<double>>& knot_params,
    double* warp_coeffs,
    const std::optional<WarpCoordinates>& warp_coordinates,
    const std::vector<Vec6<double>>& cameras_from_target,
    const std::vector<Vec3<double>>& target_points,
    const std::vector<
        std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>& frames,
    std::vector<double*>& knot_blocks,
    std::vector<ObservationRecord>& obs_records,
    double sqrt_lambda
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

    // per-knot blocks: (dx, dy)
    knot_blocks.resize(n_knots);
    for (int i = 0; i < n_knots; i++) {
        knot_blocks[i] = knot_params[i].data();
        problem.AddParameterBlock(knot_blocks[i], 2);
    }

    for (auto& cam : cameras_from_target) {
        problem.AddParameterBlock(const_cast<double*>(cam.data()), 6);
    }

    // Spline anchor constraints to prevent the spline from absorbing
    // global pose changes. We evaluate the spline at two fixed points in
    // normalized coords and constrain the output.
    auto add_spline_anchor = [&](double x_n,
                                 double y_n,
                                 bool constrain_dx,
                                 bool constrain_dy,
                                 double weight) {
        double gx, gy;
        map.normalized_to_grid_coords(x_n, y_n, gx, gy);
        const int ix = static_cast<int>(gx);
        const int iy = static_cast<int>(gy);

        if (!map.is_inside_fov(ix, iy)) {
            return;
        }

        const double u = gx - ix;
        const double v = gy - iy;
        double wx[4], wy[4];
        cubic_bspline_basis_uniform(u, wx);
        cubic_bspline_basis_uniform(v, wy);

        double basis[16];
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                basis[idx++] = wy[b] * wx[a];
            }
        }

        std::array<int, 16> flat{};
        map.support_indices_4x4(ix, iy, flat);

        auto make_anchor = [&](double target, int component) {
            SplineAnchor sa{weight, target, component, {}};
            std::copy(basis, basis + 16, sa.basis);
            auto* cost = new ceres::DynamicAutoDiffCostFunction<SplineAnchor>(
                new SplineAnchor(sa)
            );
            std::vector<double*> ptrs;
            for (int i = 0; i < 16; i++) {
                cost->AddParameterBlock(2);
                ptrs.push_back(knot_blocks[flat[i]]);
            }
            cost->SetNumResiduals(1);
            problem.AddResidualBlock(cost, nullptr, ptrs);
        };

        if (constrain_dx) {
            make_anchor(0.0, 0);
        }
        if (constrain_dy) {
            make_anchor(0.0, 1);
        }
    };

    constexpr double anchor_weight = 1000.0;
    // Point 1: optical center — constrain both dx and dy to 0
    add_spline_anchor(0.0, 0.0, true, true, anchor_weight);
    // Point 2: quarter FOV along x — constrain only dy to 0
    const double fov_rad_x = cfg.fov_deg_x * M_PI / 180.0;
    const double quarter_x_n = std::tan(fov_rad_x / 4.0);
    add_spline_anchor(quarter_x_n, 0.0, false, true, anchor_weight);

    // reprojection residuals (wired to correct 16 knots for each observation)
    // Track which cells contain at least one observation.
    std::vector<bool> cell_has_obs(normalized_x * normalized_y, false);
    obs_records.clear();
    const size_t num_cams = frames.size();
    for (size_t cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        auto& ids = std::get<0>(frames[cam_idx]);
        auto& obs = std::get<1>(frames[cam_idx]);
        auto& cam6 = cameras_from_target[cam_idx];

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

            cell_has_obs[iy * normalized_x + ix] = true;

            std::array<int, 16> flat{};
            map.support_indices_4x4(ix, iy, flat);

            // Create cost with fixed cell (ix,iy)
            // clang-format off
            // Build parameter list: cam + warp_coeffs + 16 knots
            std::array<double*, 18> blocks{};
            blocks[0] = const_cast<double*>(cam6.data());
            blocks[1] = warp_coeffs;
            for (int i = 0; i < 16; i++) { blocks[2 + i] = knot_blocks[flat[i]]; }

            if (has_warp) {
                auto* cost = new ceres::AutoDiffCostFunction<
                    ReprojectionErrorSplinedWarp, 2, 6, 5,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2
                >(new ReprojectionErrorSplinedWarp{
                    map, pinhole_params[0], pinhole_params[1], pinhole_params[2], pinhole_params[3], pw, ix, iy, ox, oy, warp_basis
                });

                problem.AddResidualBlock(cost, nullptr,
                    blocks[0],  blocks[1],
                    blocks[2],  blocks[3],  blocks[4],  blocks[5],  blocks[6],  blocks[7],  blocks[8],  blocks[9],
                    blocks[10], blocks[11], blocks[12], blocks[13], blocks[14], blocks[15], blocks[16], blocks[17]
                );
            } else {
                auto* cost = new ceres::AutoDiffCostFunction<
                    ReprojectionErrorSplinedNoWarp, 2, 6,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2
                >(new ReprojectionErrorSplinedNoWarp{
                    map, pinhole_params[0], pinhole_params[1], pinhole_params[2], pinhole_params[3], pw, ix, iy, ox, oy
                });

                problem.AddResidualBlock(cost, nullptr,
                    blocks[0],
                    blocks[2],  blocks[3],  blocks[4],  blocks[5],  blocks[6],  blocks[7],  blocks[8],  blocks[9],
                    blocks[10], blocks[11], blocks[12], blocks[13], blocks[14], blocks[15], blocks[16], blocks[17]
                );
            }
            // clang-format on

            obs_records.push_back(
                ObservationRecord{cam_idx, pt_idx, ox, oy, ix, iy}
            );
        }
    }

    // Third-derivative smoothness priors for cells without observations.
    // For each empty cell (cx, cy), add horizontal and vertical stencils
    // through both rows/columns of the cell's corner knots.
    for (int cy = 0; cy < normalized_y; cy++) {
        for (int cx = 0; cx < normalized_x; cx++) {
            if (cell_has_obs[cy * normalized_x + cx]) {
                continue;
            }

            // Horizontal: 4-knot stencil along rows cy and cy+1
            if (cx - 1 >= 0 && cx + 2 < normalized_x) {
                for (int row = cy; row <= cy + 1 && row < normalized_y; row++) {
                    const int k0 = row * normalized_x + (cx - 1);
                    const int k1 = row * normalized_x + cx;
                    const int k2 = row * normalized_x + (cx + 1);
                    const int k3 = row * normalized_x + (cx + 2);
                    problem.AddResidualBlock(
                        new ceres::
                            AutoDiffCostFunction<KnotSmoothness2D, 2, 2, 2, 2, 2>(
                                new KnotSmoothness2D{sqrt_lambda}
                            ),
                        nullptr,
                        knot_blocks[k0],
                        knot_blocks[k1],
                        knot_blocks[k2],
                        knot_blocks[k3]
                    );
                }
            }

            // Vertical: 4-knot stencil along columns cx and cx+1
            if (cy - 1 >= 0 && cy + 2 < normalized_y) {
                for (int col = cx; col <= cx + 1 && col < normalized_x; col++) {
                    const int k0 = (cy - 1) * normalized_x + col;
                    const int k1 = cy * normalized_x + col;
                    const int k2 = (cy + 1) * normalized_x + col;
                    const int k3 = (cy + 2) * normalized_x + col;
                    problem.AddResidualBlock(
                        new ceres::
                            AutoDiffCostFunction<KnotSmoothness2D, 2, 2, 2, 2, 2>(
                                new KnotSmoothness2D{sqrt_lambda}
                            ),
                        nullptr,
                        knot_blocks[k0],
                        knot_blocks[k1],
                        knot_blocks[k2],
                        knot_blocks[k3]
                    );
                }
            }
        }
    }
}

py::dict fine_tune_pinhole_splined(
    lensboy::PinholeSplinedOptimizationConfig& model_config,
    lensboy::PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
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

    double* pinhole_params = static_cast<double*>(
        intrinsics_parameters.pinhole_parameters.request().ptr
    );
    double* dxp = static_cast<double*>(dxb.ptr);
    double* dyp = static_cast<double*>(dyb.ptr);
    const int normalized_x = static_cast<int>(model_config.num_knots_x);
    const int normalized_y = static_cast<int>(model_config.num_knots_y);
    const int n_knots = normalized_x * normalized_y;

    std::vector<Vec2<double>> knot_params(n_knots);
    for (int i = 0; i < n_knots; i++) {
        knot_params[i] = Vec2<double>(dxp[i], dyp[i]);
    }

    const double sqrt_lambda = std::sqrt(model_config.smoothness_lambda);

    double warp_coeffs[5] = {
        warp_coeffs_initial[0],
        warp_coeffs_initial[1],
        warp_coeffs_initial[2],
        warp_coeffs_initial[3],
        warp_coeffs_initial[4]
    };

    SplineMap map(model_config);

    ceres::Solver::Options options;

    options.num_threads =
        std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
    if (!configure_sparse_schur_if_available(options)) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    options.minimizer_progress_to_stdout = false;

    constexpr int max_rebuilds = 1000;

    // declare these outside the loop so we don't reallocate on every rebuild
    std::vector<double*> knot_blocks;

    std::vector<ObservationRecord> obs_records;

    ceres::Solver::Summary last_summary;

    double prev_cost = std::numeric_limits<double>::max();
    int outer;
    for (outer = 0; outer < max_rebuilds; outer++) {
        ceres::Problem problem;

        BuildProblem(
            problem,
            model_config,
            map,
            pinhole_params,
            knot_params,
            warp_coeffs,
            warp_coordinates,
            cameras_from_target,
            target_points,
            frames,
            knot_blocks,
            obs_records,
            sqrt_lambda
        );

        spdlog::debug(
            "Solve pass {} (residuals wired for current cells)...",
            outer
        );

        ceres::Solve(options, &problem, &last_summary);

        if (!any_cell_changed(
                map,
                cameras_from_target,
                target_points,
                obs_records
            )) {
            spdlog::debug(
                "No cell changes detected. Done after {} rebuild(s).",
                outer
            );
            break;
        }

        const double cost = last_summary.final_cost;
        const double rel_improvement = (prev_cost - cost) / (prev_cost + 1e-30);
        if (outer > 0 && rel_improvement < 1e-6) {
            spdlog::debug(
                "Cost converged (rel improvement {:.2e}). Done after {} "
                "rebuild(s).",
                rel_improvement,
                outer
            );
            break;
        }
        prev_cost = cost;

        spdlog::debug("Cell change detected -> rebuilding problem.");
    }
    spdlog::debug("Optimization done after {} rebuilds", outer);

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
