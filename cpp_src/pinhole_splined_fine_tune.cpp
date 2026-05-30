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

// Constrains the spline correction at a fixed point to a target value.
// Precomputed basis weights make this a simple weighted sum of 16 knots.
struct SplineAnchor {
    double weight;
    double target;
    double basis[16];

    template <typename T>
    bool operator()(
        T const* const* knots,
        T* residuals
    ) const {
        T val(0.0);
        for (int i = 0; i < 16; i++) {
            val += knots[i][0] * T(basis[i]);
        }
        residuals[0] = T(weight) * (val - T(target));
        return true;
    }
};

struct ReprojectionErrorSplined {
    const SplineMap& map;
    double fx, fy, cx, cy;
    Vec3<double> pw;
    int ix0, iy0;
    double obs_x, obs_y;
    bool has_warp;
    WarpCoordinates warp_coords;

    // clang-format off
    template <typename T>
    bool operator()(
        const T* const cam, const T* const warp_coeffs,
        const T* const dx00, const T* const dx01, const T* const dx02, const T* const dx03,
        const T* const dx04, const T* const dx05, const T* const dx06, const T* const dx07,
        const T* const dx08, const T* const dx09, const T* const dx10, const T* const dx11,
        const T* const dx12, const T* const dx13, const T* const dx14, const T* const dx15,
        const T* const dy00, const T* const dy01, const T* const dy02, const T* const dy03,
        const T* const dy04, const T* const dy05, const T* const dy06, const T* const dy07,
        const T* const dy08, const T* const dy09, const T* const dy10, const T* const dy11,
        const T* const dy12, const T* const dy13, const T* const dy14, const T* const dy15,
        T* residuals
    ) const {
        const T* dx[16] = {dx00, dx01, dx02, dx03, dx04, dx05, dx06, dx07,
                           dx08, dx09, dx10, dx11, dx12, dx13, dx14, dx15};
        const T* dy[16] = {dy00, dy01, dy02, dy03, dy04, dy05, dy06, dy07,
                           dy08, dy09, dy10, dy11, dy12, dy13, dy14, dy15};
        // clang-format on

        Vec3<T> pw_warped;
        if (has_warp) {
            pw_warped = apply_warp_to_target_point(
                Vec3<T>(pw.cast<T>()),
                warp_coords,
                warp_coeffs
            );
        } else {
            pw_warped = pw.cast<T>();
        }

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
                dx_val += dx[idx][0] * w;
                dy_val += dy[idx][0] * w;
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
    double* dxp,
    double* dyp,
    double* warp_coeffs,
    const std::optional<WarpCoordinates>& warp_coordinates,
    const std::vector<Vec6<double>>& cameras_from_target,
    const std::vector<Vec3<double>>& target_points,
    const std::vector<
        std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>& frames,
    std::vector<double*>& dx_blocks,
    std::vector<double*>& dy_blocks,
    std::vector<ObservationRecord>& obs_records,
    double sqrt_lambda
) {
    const int normalized_x = static_cast<int>(cfg.num_knots_x);
    const int normalized_y = static_cast<int>(cfg.num_knots_y);
    const int n_knots = normalized_x * normalized_y;

    // pinhole_parameters constant
    problem.AddParameterBlock(const_cast<double*>(pinhole_params), 4);
    problem.SetParameterBlockConstant(const_cast<double*>(pinhole_params));

    // warp coeffs block (always present; constant when no warp)
    problem.AddParameterBlock(warp_coeffs, 5);
    if (!warp_coordinates.has_value()) {
        problem.SetParameterBlockConstant(warp_coeffs);
    }

    // per-knot blocks (size 1)
    dx_blocks.resize(n_knots);
    dy_blocks.resize(n_knots);
    for (int i = 0; i < n_knots; i++) {
        dx_blocks[i] = dxp + i;
        dy_blocks[i] = dyp + i;
        problem.AddParameterBlock(dx_blocks[i], 1);
        problem.AddParameterBlock(dy_blocks[i], 1);
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

        auto make_anchor = [&](double target, std::vector<double*>& blocks) {
            SplineAnchor sa{weight, target, {}};
            std::copy(basis, basis + 16, sa.basis);
            auto* cost = new ceres::DynamicAutoDiffCostFunction<SplineAnchor>(
                new SplineAnchor(sa)
            );
            std::vector<double*> ptrs;
            for (int i = 0; i < 16; i++) {
                cost->AddParameterBlock(1);
                ptrs.push_back(blocks[flat[i]]);
            }
            cost->SetNumResiduals(1);
            problem.AddResidualBlock(cost, nullptr, ptrs);
        };

        if (constrain_dx) {
            make_anchor(0.0, dx_blocks);
        }
        if (constrain_dy) {
            make_anchor(0.0, dy_blocks);
        }
    };

    constexpr double anchor_weight = 1000.0;
    // Point 1: optical center — constrain both dx and dy to 0
    add_spline_anchor(0.0, 0.0, true, true, anchor_weight);
    // Point 2: quarter FOV along x — constrain only dy to 0
    const double fov_rad_x = cfg.fov_deg_x * M_PI / 180.0;
    const double quarter_x_n = std::tan(fov_rad_x / 4.0);
    add_spline_anchor(quarter_x_n, 0.0, false, true, anchor_weight);

    for (auto& pt : target_points) {
        problem.AddParameterBlock(const_cast<double*>(pt.data()), 3);
        problem.SetParameterBlockConstant(const_cast<double*>(pt.data()));
    }

    // Filter out observations that project outside the calibrated FOV.
    std::vector<std::vector<size_t>> valid_observation_indices(frames.size());
    for (size_t cam_idx = 0; cam_idx < frames.size(); cam_idx++) {
        auto& ids = std::get<0>(frames[cam_idx]);
        auto& cam6 = cameras_from_target[cam_idx];
        for (size_t oi = 0; oi < ids.size(); oi++) {
            int ix, iy;
            map.cell_index(cam6.data(), target_points[ids[oi]], ix, iy);
            if (map.is_inside_fov(ix, iy)) {
                valid_observation_indices[cam_idx].push_back(oi);
            }
        }
    }

    // reprojection residuals (wired to correct 16 knots for each observation)
    // Track which cells contain at least one observation.
    std::vector<bool> cell_has_obs(normalized_x * normalized_y, false);
    obs_records.clear();
    const size_t num_cams = frames.size();
    for (size_t cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        auto& ids = std::get<0>(frames[cam_idx]);
        auto& obs = std::get<1>(frames[cam_idx]);
        auto& cam6 = cameras_from_target[cam_idx];

        for (size_t oi : valid_observation_indices[cam_idx]) {
            const int pt_idx = ids[oi];
            const auto& pw = target_points[pt_idx];
            const double ox = obs[oi](0, 0);
            const double oy = obs[oi](1, 0);

            int ix, iy;
            map.cell_index(cam6.data(), pw, ix, iy);

            cell_has_obs[iy * normalized_x + ix] = true;

            std::array<int, 16> flat{};
            map.support_indices_4x4(ix, iy, flat);

            // Create cost with fixed cell (ix,iy)
            const bool hw = warp_coordinates.has_value();
            const WarpCoordinates wc =
                hw ? *warp_coordinates : WarpCoordinates{};
            // clang-format off
            auto* cost = new ceres::AutoDiffCostFunction<
                ReprojectionErrorSplined, 2, 6, 5,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 16 dx knots
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   // 16 dy knots
            >(new ReprojectionErrorSplined{
                map, pinhole_params[0], pinhole_params[1], pinhole_params[2], pinhole_params[3], pw, ix, iy, ox, oy, hw, wc
            });

            // Build parameter list: cam + warp_coeffs + 16 dx + 16 dy
            std::array<double*, 34> blocks{};
            blocks[0] = const_cast<double*>(cam6.data());
            blocks[1] = warp_coeffs;
            for (int i = 0; i < 16; i++) { blocks[2 + i]  = dx_blocks[flat[i]]; }
            for (int i = 0; i < 16; i++) { blocks[18 + i] = dy_blocks[flat[i]]; }

            problem.AddResidualBlock(cost, nullptr,
                blocks[0],  blocks[1],
                blocks[2],  blocks[3],  blocks[4],  blocks[5],  blocks[6],  blocks[7],  blocks[8],  blocks[9],
                blocks[10], blocks[11], blocks[12], blocks[13], blocks[14], blocks[15], blocks[16], blocks[17],
                blocks[18], blocks[19], blocks[20], blocks[21], blocks[22], blocks[23], blocks[24], blocks[25],
                blocks[26], blocks[27], blocks[28], blocks[29], blocks[30], blocks[31], blocks[32], blocks[33]
            );
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
                            AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                                new KnotSmoothness{sqrt_lambda}
                            ),
                        nullptr,
                        dx_blocks[k0],
                        dx_blocks[k1],
                        dx_blocks[k2],
                        dx_blocks[k3]
                    );
                    problem.AddResidualBlock(
                        new ceres::
                            AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                                new KnotSmoothness{sqrt_lambda}
                            ),
                        nullptr,
                        dy_blocks[k0],
                        dy_blocks[k1],
                        dy_blocks[k2],
                        dy_blocks[k3]
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
                            AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                                new KnotSmoothness{sqrt_lambda}
                            ),
                        nullptr,
                        dx_blocks[k0],
                        dx_blocks[k1],
                        dx_blocks[k2],
                        dx_blocks[k3]
                    );
                    problem.AddResidualBlock(
                        new ceres::
                            AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                                new KnotSmoothness{sqrt_lambda}
                            ),
                        nullptr,
                        dy_blocks[k0],
                        dy_blocks[k1],
                        dy_blocks[k2],
                        dy_blocks[k3]
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
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.minimizer_progress_to_stdout = false;

    constexpr int max_rebuilds = 1000;

    // declare these outside the loop so we don't reallocate on every rebuild
    std::vector<double*> dx_blocks, dy_blocks;

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
            dxp,
            dyp,
            warp_coeffs,
            warp_coordinates,
            cameras_from_target,
            target_points,
            frames,
            dx_blocks,
            dy_blocks,
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
