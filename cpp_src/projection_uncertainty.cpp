#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include "./cameramodels.hpp"
#include "./ceres_geometry.hpp"
#include "./pinhole_splined_problem.hpp"
#include "./pybind_utils.hpp"
#include "./type_defs.hpp"

namespace lensboy {

namespace {

struct OpenCVResidualNoWarp {
    Vec3<double> point_target;
    double observed_x;
    double observed_y;

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const camera_from_target,
        T* residuals
    ) const {
        Vec6<T> pose(camera_from_target);
        const Vec3<T> point = point_target.cast<T>();
        Vec3<T> point_camera = transform_point(pose, point);
        Vec2<T> image;
        project_opencv(intrinsics, point_camera, image);
        residuals[0] = image[0] - T(observed_x);
        residuals[1] = image[1] - T(observed_y);
        return true;
    }
};

struct OpenCVResidualWarp {
    Vec3<double> point_target;
    double observed_x;
    double observed_y;
    WarpCoordinates warp_coordinates;

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const camera_from_target,
        const T* const warp_coeffs,
        T* residuals
    ) const {
        const Vec3<T> point = point_target.cast<T>();
        Vec3<T> point_warped =
            apply_warp_to_target_point(point, warp_coordinates, warp_coeffs);
        Vec6<T> pose(camera_from_target);
        Vec3<T> point_camera = transform_point(pose, point_warped);
        Vec2<T> image;
        project_opencv(intrinsics, point_camera, image);
        residuals[0] = image[0] - T(observed_x);
        residuals[1] = image[1] - T(observed_y);
        return true;
    }
};

struct OpenCVProjectRay {
    Vec3<double> ray;

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const point_camera,
        T* residuals
    ) const {
        Vec3<T> point(point_camera);
        Vec2<T> image;
        project_opencv(intrinsics, point, image);
        residuals[0] = image[0];
        residuals[1] = image[1];
        return true;
    }
};

struct SplineResidualNoWarp {
    const SplineMap& map;
    double fx;
    double fy;
    double cx;
    double cy;
    Vec3<double> point_target;
    int ix0;
    int iy0;
    double observed_x;
    double observed_y;

    template <typename T>
    bool operator()(
        const T* const camera_from_target,
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
        Vec6<T> pose(camera_from_target);
        const Vec3<T> point = point_target.cast<T>();
        Vec3<T> point_camera = transform_point(pose, point);
        const T x_n = point_camera[0] / point_camera[2];
        const T y_n = point_camera[1] / point_camera[2];

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
        T dx(0.0), dy(0.0);
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                const T w = wy[b] * wx[a];
                dx += knots[idx][0] * w;
                dy += knots[idx][1] * w;
                idx++;
            }
        }
        residuals[0] = T(fx) * (x_n + dx) + T(cx) - T(observed_x);
        residuals[1] = T(fy) * (y_n + dy) + T(cy) - T(observed_y);
        return true;
    }
};

struct SplineResidualWarp {
    const SplineMap& map;
    double fx;
    double fy;
    double cx;
    double cy;
    Vec3<double> point_target;
    int ix0;
    int iy0;
    double observed_x;
    double observed_y;
    WarpCoordinates warp_coordinates;

    template <typename T>
    bool operator()(
        const T* const camera_from_target,
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
        const Vec3<T> point = point_target.cast<T>();
        Vec3<T> point_warped =
            apply_warp_to_target_point(point, warp_coordinates, warp_coeffs);
        Vec6<T> pose(camera_from_target);
        Vec3<T> point_camera = transform_point(pose, point_warped);
        const T x_n = point_camera[0] / point_camera[2];
        const T y_n = point_camera[1] / point_camera[2];

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
        T dx(0.0), dy(0.0);
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                const T w = wy[b] * wx[a];
                dx += knots[idx][0] * w;
                dy += knots[idx][1] * w;
                idx++;
            }
        }
        residuals[0] = T(fx) * (x_n + dx) + T(cx) - T(observed_x);
        residuals[1] = T(fy) * (y_n + dy) + T(cy) - T(observed_y);
        return true;
    }
};

struct SplineProjectRay {
    const SplineMap& map;
    double fx;
    double fy;
    double cx;
    double cy;
    Vec3<double> ray;
    int ix0;
    int iy0;

    template <typename T>
    bool operator()(
        const T* const point_camera,
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
        Vec3<T> point(point_camera);
        const T x_n = point[0] / point[2];
        const T y_n = point[1] / point[2];
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
        T dx(0.0), dy(0.0);
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                const T w = wy[b] * wx[a];
                dx += knots[idx][0] * w;
                dy += knots[idx][1] * w;
                idx++;
            }
        }
        residuals[0] = T(fx) * (x_n + dx) + T(cx);
        residuals[1] = T(fy) * (y_n + dy) + T(cy);
        return true;
    }
};

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Matrix map_ceres_jacobian(
    const double* data,
    int rows,
    int cols
) {
    return Eigen::Map<const RowMajorMatrix>(data, rows, cols);
}

void accumulate_residual(
    Matrix& hessian,
    Matrix& group_gradients,
    const std::vector<int>& columns,
    const Matrix& jacobian,
    const double residuals[2],
    size_t group_idx
) {
    const Vector r = Eigen::Map<const Eigen::Vector2d>(residuals);
    const Matrix jt = jacobian.transpose();
    for (size_t local_a = 0; local_a < columns.size(); local_a++) {
        const int global_a = columns[local_a];
        group_gradients(group_idx, global_a) += jt.row(local_a).dot(r);
        for (size_t local_b = 0; local_b < columns.size(); local_b++) {
            hessian(global_a, columns[local_b]) +=
                jt.row(local_a).dot(jacobian.col(local_b));
        }
    }
}

void accumulate_residual_vector(
    Matrix& hessian,
    Matrix* group_gradients,
    const std::vector<int>& columns,
    const Matrix& jacobian,
    const Vector& residuals,
    size_t group_idx
) {
    const Matrix jt = jacobian.transpose();
    for (size_t local_a = 0; local_a < columns.size(); local_a++) {
        const int global_a = columns[local_a];
        if (group_gradients != nullptr) {
            (*group_gradients)(group_idx, global_a) += jt.row(local_a).dot(residuals);
        }
        for (size_t local_b = 0; local_b < columns.size(); local_b++) {
            hessian(global_a, columns[local_b]) +=
                jt.row(local_a).dot(jacobian.col(local_b));
        }
    }
}

void accumulate_recorded_residual(
    const ceres::Problem& problem,
    const SplineResidualRecord& record,
    Matrix& hessian,
    Matrix& group_gradients
) {
    const int num_blocks = static_cast<int>(record.block_sizes.size());
    std::vector<std::vector<double>> jacobian_storage;
    jacobian_storage.reserve(num_blocks);
    std::vector<double*> jacobians;
    jacobians.reserve(num_blocks);
    for (int block_size : record.block_sizes) {
        jacobian_storage.emplace_back(record.residual_size * block_size);
        jacobians.push_back(jacobian_storage.back().data());
    }

    Vector residuals(record.residual_size);
    double cost = 0.0;
    const bool ok = problem.EvaluateResidualBlock(
        record.residual_id,
        false,
        &cost,
        residuals.data(),
        jacobians.data()
    );
    require(ok, "Failed to evaluate spline residual block for uncertainty.");

    Matrix jacobian(record.residual_size, record.columns.size());
    int col = 0;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int block_size = record.block_sizes[block_idx];
        jacobian.block(0, col, record.residual_size, block_size) =
            map_ceres_jacobian(
                jacobian_storage[block_idx].data(),
                record.residual_size,
                block_size
            );
        col += block_size;
    }

    Matrix* gradients = nullptr;
    if (record.empirical) {
        gradients = &group_gradients;
    }
    accumulate_residual_vector(
        hessian,
        gradients,
        record.columns,
        jacobian,
        residuals,
        record.frame_idx
    );
}

Matrix solve_self_adjoint_regularized(
    const Matrix& hessian,
    const Matrix& rhs,
    double damping,
    double relative_floor,
    py::dict& metadata
) {
    Eigen::SelfAdjointEigenSolver<Matrix> solver(hessian);
    const Vector evals = solver.eigenvalues();
    const double max_eval = std::max(evals.maxCoeff(), 0.0);
    const double floor = std::max(damping, relative_floor * std::max(max_eval, 1.0));
    Vector inv(evals.size());
    int regularized = 0;
    for (int i = 0; i < evals.size(); i++) {
        double value = evals[i];
        if (value < floor) {
            value = floor;
            regularized++;
        }
        inv[i] = 1.0 / value;
    }
    metadata["hessian_min_eigenvalue"] = evals.minCoeff();
    metadata["hessian_max_eigenvalue"] = evals.maxCoeff();
    metadata["hessian_eigenvalue_floor"] = floor;
    metadata["hessian_regularized_modes"] = regularized;
    metadata["hessian_solve"] = "regularized self-adjoint solve";
    return solver.eigenvectors() *
           (inv.asDiagonal() * (solver.eigenvectors().transpose() * rhs));
}

struct ProfiledSystem {
    Matrix hessian;
    Matrix gradients;
    py::dict metadata;
};

ProfiledSystem profile_camera_system(
    const Matrix& hessian,
    const Matrix& gradients,
    int num_camera_params,
    double damping,
    double relative_floor
) {
    const int total_params = static_cast<int>(hessian.rows());
    const int num_nuisance = total_params - num_camera_params;
    py::dict metadata;
    metadata["num_camera_params"] = num_camera_params;
    metadata["num_nuisance_params"] = num_nuisance;
    metadata["damping"] = damping;
    metadata["relative_eigen_floor"] = relative_floor;

    if (num_nuisance == 0) {
        metadata["nuisance_profiled"] = false;
        return ProfiledSystem{
            hessian.topLeftCorner(num_camera_params, num_camera_params),
            gradients.leftCols(num_camera_params),
            metadata
        };
    }

    const Matrix h_cc = hessian.topLeftCorner(num_camera_params, num_camera_params);
    const Matrix h_cn =
        hessian.topRightCorner(num_camera_params, num_nuisance);
    const Matrix h_nc =
        hessian.bottomLeftCorner(num_nuisance, num_camera_params);
    const Matrix h_nn =
        hessian.bottomRightCorner(num_nuisance, num_nuisance);
    const Matrix g_c = gradients.leftCols(num_camera_params);
    const Matrix g_n = gradients.rightCols(num_nuisance);

    py::dict nuisance_metadata;
    const Matrix x = solve_self_adjoint_regularized(
        h_nn,
        h_nc,
        damping,
        relative_floor,
        nuisance_metadata
    );
    metadata["nuisance_profiled"] = true;
    metadata["nuisance_min_eigenvalue"] = nuisance_metadata["hessian_min_eigenvalue"];
    metadata["nuisance_max_eigenvalue"] = nuisance_metadata["hessian_max_eigenvalue"];
    metadata["nuisance_eigenvalue_floor"] =
        nuisance_metadata["hessian_eigenvalue_floor"];
    metadata["nuisance_regularized_modes"] =
        nuisance_metadata["hessian_regularized_modes"];

    return ProfiledSystem{h_cc - h_cn * x, g_c - g_n * x, metadata};
}

void add_spline_anchor_hessian(
    Matrix& hessian,
    const SplineMap& map,
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

    std::array<double, 16> basis{};
    int basis_index = 0;
    for (int b = 0; b < 4; b++) {
        for (int a = 0; a < 4; a++) {
            basis[basis_index] = wy[b] * wx[a];
            basis_index++;
        }
    }

    std::array<int, 16> flat{};
    map.support_indices_4x4(ix, iy, flat);

    auto add_component = [&](int component) {
        for (int i = 0; i < 16; i++) {
            const int col_i = 2 * flat[i] + component;
            const double jac_i = weight * basis[i];
            for (int j = 0; j < 16; j++) {
                const int col_j = 2 * flat[j] + component;
                const double jac_j = weight * basis[j];
                hessian(col_i, col_j) += jac_i * jac_j;
            }
        }
    };

    if (constrain_dx) {
        add_component(0);
    }
    if (constrain_dy) {
        add_component(1);
    }
}

void add_spline_smoothness_hessian(
    Matrix& hessian,
    int nx,
    int ny,
    double weight
) {
    constexpr std::array<double, 4> coeffs = {-1.0, 3.0, -3.0, 1.0};

    auto add_stencil = [&](const std::array<int, 4>& knot_indices) {
        for (int component = 0; component < 2; component++) {
            for (int i = 0; i < 4; i++) {
                const int col_i = 2 * knot_indices[i] + component;
                const double jac_i = weight * coeffs[i];
                for (int j = 0; j < 4; j++) {
                    const int col_j = 2 * knot_indices[j] + component;
                    const double jac_j = weight * coeffs[j];
                    hessian(col_i, col_j) += jac_i * jac_j;
                }
            }
        }
    };

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x + 3 < nx; x++) {
            add_stencil(
                {y * nx + x, y * nx + x + 1, y * nx + x + 2, y * nx + x + 3}
            );
        }
    }

    for (int y = 0; y + 3 < ny; y++) {
        for (int x = 0; x < nx; x++) {
            add_stencil(
                {y * nx + x,
                 (y + 1) * nx + x,
                 (y + 2) * nx + x,
                 (y + 3) * nx + x}
            );
        }
    }
}

py::dict finish_uncertainty(
    const ProfiledSystem& profiled,
    const std::vector<Matrix>& query_jacobians,
    const std::vector<Matrix>* query_rotation_bases,
    const std::vector<Matrix>* alignment_jacobians,
    const std::vector<Matrix>* alignment_rotation_bases,
    double damping,
    double relative_floor
) {
    const int num_queries = static_cast<int>(query_jacobians.size());
    const int num_groups = static_cast<int>(profiled.gradients.rows());
    int num_alignment_queries = 0;
    if (alignment_jacobians != nullptr) {
        num_alignment_queries = static_cast<int>(alignment_jacobians->size());
    }

    Matrix rhs(profiled.hessian.rows(), (num_queries + num_alignment_queries) * 2);
    for (int q = 0; q < num_queries; q++) {
        rhs.col(2 * q + 0) = query_jacobians[q].row(0).transpose();
        rhs.col(2 * q + 1) = query_jacobians[q].row(1).transpose();
    }
    for (int q = 0; q < num_alignment_queries; q++) {
        const int col = 2 * (num_queries + q);
        rhs.col(col + 0) = (*alignment_jacobians)[q].row(0).transpose();
        rhs.col(col + 1) = (*alignment_jacobians)[q].row(1).transpose();
    }

    py::dict metadata = profiled.metadata;
    Matrix solved = solve_self_adjoint_regularized(
        profiled.hessian,
        rhs,
        damping,
        relative_floor,
        metadata
    );
    metadata["num_groups"] = num_groups;
    metadata["num_queries"] = num_queries;
    metadata["loss_normalization"] = "sum Gauss-Newton sandwich";

    Matrix empirical_gradients = profiled.gradients;
    double covariance_scale = 1.0;
    if (num_groups > 1) {
        const Eigen::RowVectorXd mean_gradient = empirical_gradients.colwise().mean();
        empirical_gradients.rowwise() -= mean_gradient;
        covariance_scale = static_cast<double>(num_groups) /
                           static_cast<double>(num_groups - 1);
        metadata["empirical_gradients_centered"] = true;
        metadata["empirical_gradient_mean_norm"] = mean_gradient.norm();
        metadata["empirical_covariance_scale"] = covariance_scale;
    } else {
        metadata["empirical_gradients_centered"] = false;
        metadata["empirical_gradient_mean_norm"] = 0.0;
        metadata["empirical_covariance_scale"] = covariance_scale;
    }

    std::vector<Matrix> z_by_group;
    z_by_group.reserve(num_groups);
    for (int g = 0; g < num_groups; g++) {
        Matrix z(num_queries, 2);
        for (int q = 0; q < num_queries; q++) {
            z(q, 0) = empirical_gradients.row(g).dot(solved.col(2 * q + 0));
            z(q, 1) = empirical_gradients.row(g).dot(solved.col(2 * q + 1));
        }
        z_by_group.push_back(z);
    }

    if (query_rotation_bases != nullptr && alignment_jacobians != nullptr &&
        alignment_rotation_bases != nullptr && !alignment_rotation_bases->empty()) {
        require(
            static_cast<int>(query_rotation_bases->size()) == num_queries,
            "Query rotation quotient basis count must match query count."
        );
        require(
            static_cast<int>(alignment_rotation_bases->size()) == num_alignment_queries,
            "Alignment rotation quotient basis count must match alignment query count."
        );
        Matrix query_basis(2 * num_queries, 3);
        for (int q = 0; q < num_queries; q++) {
            query_basis.block(2 * q, 0, 2, 3) = (*query_rotation_bases)[q];
        }
        Matrix alignment_basis(2 * num_alignment_queries, 3);
        for (int q = 0; q < num_alignment_queries; q++) {
            alignment_basis.block(2 * q, 0, 2, 3) = (*alignment_rotation_bases)[q];
        }
        Eigen::CompleteOrthogonalDecomposition<Matrix> decomposition(alignment_basis);
        for (int g = 0; g < num_groups; g++) {
            Vector z_alignment(2 * num_alignment_queries);
            const int alignment_col_offset = 2 * num_queries;
            for (int q = 0; q < num_alignment_queries; q++) {
                z_alignment[2 * q + 0] = empirical_gradients.row(g).dot(
                    solved.col(alignment_col_offset + 2 * q + 0)
                );
                z_alignment[2 * q + 1] = empirical_gradients.row(g).dot(
                    solved.col(alignment_col_offset + 2 * q + 1)
                );
            }
            const Vector rotation = decomposition.solve(z_alignment);
            Vector z_query(2 * num_queries);
            for (int q = 0; q < num_queries; q++) {
                z_query[2 * q + 0] = z_by_group[g](q, 0);
                z_query[2 * q + 1] = z_by_group[g](q, 1);
            }
            const Vector aligned = z_query - query_basis * rotation;
            for (int q = 0; q < num_queries; q++) {
                z_by_group[g](q, 0) = aligned[2 * q + 0];
                z_by_group[g](q, 1) = aligned[2 * q + 1];
            }
        }
    }

    py::array_t<double> cov_array({num_queries, 2, 2});
    py::array_t<double> trace_array(num_queries);
    auto cov_out = cov_array.mutable_unchecked<3>();
    auto trace_out = trace_array.mutable_unchecked<1>();

    for (int q = 0; q < num_queries; q++) {
        Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
        for (int g = 0; g < num_groups; g++) {
            const Eigen::Vector2d z = z_by_group[g].row(q).transpose();
            cov += z * z.transpose();
        }
        cov *= covariance_scale;
        cov = 0.5 * (cov + cov.transpose());
        cov_out(q, 0, 0) = cov(0, 0);
        cov_out(q, 0, 1) = cov(0, 1);
        cov_out(q, 1, 0) = cov(1, 0);
        cov_out(q, 1, 1) = cov(1, 1);
        trace_out(q) = std::sqrt(std::max(0.0, cov.trace()));
    }

    py::dict out;
    out["covariances_px"] = cov_array;
    out["trace_std_px"] = trace_array;
    out["metadata"] = metadata;
    return out;
}

std::vector<Vec3<double>> read_rays(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_rays
) {
    auto buf = query_rays.request();
    require(buf.ndim == 2 && buf.shape[1] == 3, "query_rays must have shape (N, 3)");
    const double* ptr = static_cast<const double*>(buf.ptr);
    std::vector<Vec3<double>> rays(buf.shape[0]);
    for (ssize_t i = 0; i < buf.shape[0]; i++) {
        rays[i] = Vec3<double>(ptr[3 * i + 0], ptr[3 * i + 1], ptr[3 * i + 2]);
    }
    return rays;
}

Matrix post_rotation_basis(
    const Matrix& ray_jacobian,
    const Vec3<double>& ray
) {
    Matrix tangent(3, 3);
    tangent << 0.0, -ray[2], ray[1],
        ray[2], 0.0, -ray[0],
        -ray[1], ray[0], 0.0;
    return ray_jacobian * tangent;
}

}  // namespace

py::dict projection_uncertainty_opencv(
    std::vector<double>& intrinsics,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>& frames,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_rays,
    py::array_t<double, py::array::c_style | py::array::forcecast> alignment_rays,
    std::optional<WarpCoordinates> warp_coordinates,
    std::array<double, 5> warp_coeffs_initial,
    double damping,
    double relative_eigen_floor
) {
    require(
        intrinsics_param_optimize_mask.size() == 18,
        "OpenCV uncertainty optimization mask must have length 18."
    );
    std::array<int, 18> active_intrinsic_columns{};
    active_intrinsic_columns.fill(-1);
    int c_params = 0;
    for (int k = 0; k < 18; k++) {
        if (!intrinsics_param_optimize_mask[k]) {
            continue;
        }
        active_intrinsic_columns[k] = c_params;
        c_params++;
    }
    require(c_params > 0, "OpenCV uncertainty needs at least one active intrinsic parameter.");

    const bool has_warp = warp_coordinates.has_value();
    const int n_params = static_cast<int>(6 * cameras_from_target.size()) +
                         (has_warp ? 5 : 0);
    const int total_params = c_params + n_params;
    Matrix hessian = Matrix::Zero(total_params, total_params);
    Matrix group_gradients = Matrix::Zero(cameras_from_target.size(), total_params);
    double warp_coeffs[5] = {warp_coeffs_initial[0], warp_coeffs_initial[1],
                             warp_coeffs_initial[2], warp_coeffs_initial[3],
                             warp_coeffs_initial[4]};

    for (size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
        auto& ids = std::get<0>(frames[frame_idx]);
        auto& observations = std::get<1>(frames[frame_idx]);
        double* cam = cameras_from_target[frame_idx].data();
        for (size_t obs_idx = 0; obs_idx < ids.size(); obs_idx++) {
            const Vec3<double>& point = target_points[ids[obs_idx]];
            const Vec2<double>& obs = observations[obs_idx];
            double residuals[2];
            if (has_warp) {
                auto* cost = new ceres::AutoDiffCostFunction<
                    OpenCVResidualWarp,
                    2,
                    18,
                    6,
                    5>(new OpenCVResidualWarp{point, obs[0], obs[1], *warp_coordinates});
                double* blocks[3] = {intrinsics.data(), cam, warp_coeffs};
                std::array<double, 2 * 18> j_intr{};
                std::array<double, 2 * 6> j_cam{};
                std::array<double, 2 * 5> j_warp{};
                double* jac[3] = {j_intr.data(), j_cam.data(), j_warp.data()};
                cost->Evaluate(blocks, residuals, jac);
                delete cost;
                std::vector<int> cols;
                Matrix j(2, c_params + 11);
                cols.reserve(c_params + 11);
                int col = 0;
                for (int k = 0; k < 18; k++) {
                    if (active_intrinsic_columns[k] < 0) {
                        continue;
                    }
                    cols.push_back(active_intrinsic_columns[k]);
                    j.col(col) = map_ceres_jacobian(j_intr.data(), 2, 18).col(k);
                    col++;
                }
                for (int k = 0; k < 6; k++) cols.push_back(c_params + 6 * frame_idx + k);
                for (int k = 0; k < 5; k++) cols.push_back(c_params + 6 * frames.size() + k);
                j.block(0, col, 2, 6) = map_ceres_jacobian(j_cam.data(), 2, 6);
                col += 6;
                j.block(0, col, 2, 5) = map_ceres_jacobian(j_warp.data(), 2, 5);
                accumulate_residual(hessian, group_gradients, cols, j, residuals, frame_idx);
            } else {
                auto* cost = new ceres::AutoDiffCostFunction<
                    OpenCVResidualNoWarp,
                    2,
                    18,
                    6>(new OpenCVResidualNoWarp{point, obs[0], obs[1]});
                double* blocks[2] = {intrinsics.data(), cam};
                std::array<double, 2 * 18> j_intr{};
                std::array<double, 2 * 6> j_cam{};
                double* jac[2] = {j_intr.data(), j_cam.data()};
                cost->Evaluate(blocks, residuals, jac);
                delete cost;
                std::vector<int> cols;
                Matrix j(2, c_params + 6);
                cols.reserve(c_params + 6);
                int col = 0;
                for (int k = 0; k < 18; k++) {
                    if (active_intrinsic_columns[k] < 0) {
                        continue;
                    }
                    cols.push_back(active_intrinsic_columns[k]);
                    j.col(col) = map_ceres_jacobian(j_intr.data(), 2, 18).col(k);
                    col++;
                }
                for (int k = 0; k < 6; k++) cols.push_back(c_params + 6 * frame_idx + k);
                j.block(0, col, 2, 6) = map_ceres_jacobian(j_cam.data(), 2, 6);
                accumulate_residual(hessian, group_gradients, cols, j, residuals, frame_idx);
            }
        }
    }

    ProfiledSystem profiled = profile_camera_system(
        hessian,
        group_gradients,
        c_params,
        damping,
        relative_eigen_floor
    );

    std::vector<Vec3<double>> rays = read_rays(query_rays);
    std::vector<Matrix> query_jacobians;
    std::vector<Matrix> rotation_bases;
    query_jacobians.reserve(rays.size());
    rotation_bases.reserve(rays.size());

    for (const auto& ray : rays) {
        auto* cost = new ceres::AutoDiffCostFunction<OpenCVProjectRay, 2, 18, 3>(
            new OpenCVProjectRay{ray}
        );
        Vec3<double> ray_copy = ray;
        double* blocks[2] = {intrinsics.data(), ray_copy.data()};
        double residuals[2];
        std::array<double, 2 * 18> j_intr{};
        std::array<double, 2 * 3> j_ray{};
        double* jac[2] = {j_intr.data(), j_ray.data()};
        cost->Evaluate(blocks, residuals, jac);
        delete cost;
        Matrix j_active(2, c_params);
        int col = 0;
        for (int k = 0; k < 18; k++) {
            if (active_intrinsic_columns[k] < 0) {
                continue;
            }
            j_active.col(col) = map_ceres_jacobian(j_intr.data(), 2, 18).col(k);
            col++;
        }
        query_jacobians.push_back(j_active);
        rotation_bases.push_back(
            post_rotation_basis(map_ceres_jacobian(j_ray.data(), 2, 3), ray)
        );
    }

    std::vector<Vec3<double>> alignment_ray_values = read_rays(alignment_rays);
    std::vector<Matrix> alignment_jacobians;
    std::vector<Matrix> alignment_rotation_bases;
    alignment_jacobians.reserve(alignment_ray_values.size());
    alignment_rotation_bases.reserve(alignment_ray_values.size());
    for (const auto& ray : alignment_ray_values) {
        auto* cost = new ceres::AutoDiffCostFunction<OpenCVProjectRay, 2, 18, 3>(
            new OpenCVProjectRay{ray}
        );
        Vec3<double> ray_copy = ray;
        double* blocks[2] = {intrinsics.data(), ray_copy.data()};
        double residuals[2];
        std::array<double, 2 * 18> j_intr{};
        std::array<double, 2 * 3> j_ray{};
        double* jac[2] = {j_intr.data(), j_ray.data()};
        cost->Evaluate(blocks, residuals, jac);
        delete cost;
        Matrix j_active(2, c_params);
        int col = 0;
        for (int k = 0; k < 18; k++) {
            if (active_intrinsic_columns[k] < 0) {
                continue;
            }
            j_active.col(col) = map_ceres_jacobian(j_intr.data(), 2, 18).col(k);
            col++;
        }
        alignment_jacobians.push_back(j_active);
        alignment_rotation_bases.push_back(
            post_rotation_basis(map_ceres_jacobian(j_ray.data(), 2, 3), ray)
        );
    }

    py::dict out = finish_uncertainty(
        profiled,
        query_jacobians,
        &rotation_bases,
        &alignment_jacobians,
        &alignment_rotation_bases,
        damping,
        relative_eigen_floor
    );
    out["metadata"].cast<py::dict>()["opencv_active_intrinsic_params"] = c_params;
    out["metadata"].cast<py::dict>()["opencv_inactive_intrinsic_params"] = 18 - c_params;
    return out;
}

py::dict projection_uncertainty_pinhole_splined(
    PinholeSplinedOptimizationConfig& model_config,
    PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>& frames,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_rays,
    std::optional<WarpCoordinates> warp_coordinates,
    std::array<double, 5> warp_coeffs_initial,
    double damping,
    double relative_eigen_floor
) {
    auto pinhole_buf = intrinsics_parameters.pinhole_parameters.request();
    auto dxb = intrinsics_parameters.dx_grid.request();
    auto dyb = intrinsics_parameters.dy_grid.request();
    double* pinhole = static_cast<double*>(pinhole_buf.ptr);
    double* dxp = static_cast<double*>(dxb.ptr);
    double* dyp = static_cast<double*>(dyb.ptr);
    const int num_knots = static_cast<int>(model_config.num_knots_x * model_config.num_knots_y);
    std::vector<Vec2<double>> knots(num_knots);
    for (int i = 0; i < num_knots; i++) {
        knots[i] = Vec2<double>(dxp[i], dyp[i]);
    }

    const int c_params = 2 * num_knots;
    const bool has_warp = warp_coordinates.has_value();
    const int n_params = static_cast<int>(6 * cameras_from_target.size()) +
                         (has_warp ? 5 : 0);
    const int total_params = c_params + n_params;
    Matrix hessian = Matrix::Zero(total_params, total_params);
    Matrix group_gradients = Matrix::Zero(cameras_from_target.size(), total_params);
    double warp_coeffs[5] = {warp_coeffs_initial[0], warp_coeffs_initial[1],
                             warp_coeffs_initial[2], warp_coeffs_initial[3],
                             warp_coeffs_initial[4]};
    const SplineMap map(model_config);

    ceres::Problem problem;
    std::vector<double*> knot_blocks;
    std::vector<ObservationRecord> obs_records;
    std::vector<SplineResidualRecord> residual_records;
    build_pinhole_splined_problem(
        problem,
        model_config,
        map,
        pinhole,
        knots,
        warp_coeffs,
        warp_coordinates,
        cameras_from_target,
        target_points,
        frames,
        knot_blocks,
        obs_records,
        &residual_records,
        std::sqrt(model_config.smoothness_lambda)
    );

    for (const SplineResidualRecord& record : residual_records) {
        accumulate_recorded_residual(problem, record, hessian, group_gradients);
    }

    ProfiledSystem profiled = profile_camera_system(
        hessian,
        group_gradients,
        c_params,
        damping,
        relative_eigen_floor
    );
    profiled.metadata["spline_anchor_weight"] = 1000.0;
    profiled.metadata["spline_smoothness_lambda"] = model_config.smoothness_lambda;
    profiled.metadata["spline_regularizers_in_hessian"] = true;
    profiled.metadata["spline_regularizers_in_empirical_gradients"] = false;
    profiled.metadata["spline_uncertainty_uses_shared_problem"] = true;

    std::vector<Vec3<double>> rays = read_rays(query_rays);
    std::vector<Matrix> query_jacobians;
    query_jacobians.reserve(rays.size());
    for (const auto& ray : rays) {
        double gx, gy;
        map.normalized_to_grid_coords(ray[0] / ray[2], ray[1] / ray[2], gx, gy);
        const int ix = static_cast<int>(gx);
        const int iy = static_cast<int>(gy);
        Matrix j_full = Matrix::Zero(2, c_params);
        std::array<int, 16> flat{};
        map.support_indices_4x4(ix, iy, flat);
        auto* cost = new ceres::AutoDiffCostFunction<
            SplineProjectRay,
            2,
            3,
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
            2>(new SplineProjectRay{map, pinhole[0], pinhole[1], pinhole[2], pinhole[3], ray, ix, iy});
        Vec3<double> ray_copy = ray;
        std::array<double*, 17> blocks{};
        blocks[0] = ray_copy.data();
        for (int k = 0; k < 16; k++) blocks[1 + k] = knots[flat[k]].data();
        double residuals[2];
        std::array<std::array<double, 4>, 16> j_knots{};
        std::array<double*, 17> jac{};
        jac[0] = nullptr;
        for (int k = 0; k < 16; k++) jac[1 + k] = j_knots[k].data();
        cost->Evaluate(blocks.data(), residuals, jac.data());
        delete cost;
        for (int k = 0; k < 16; k++) {
            j_full.block(0, 2 * flat[k], 2, 2) +=
                map_ceres_jacobian(j_knots[k].data(), 2, 2);
        }
        query_jacobians.push_back(j_full);
    }

    py::dict out = finish_uncertainty(
        profiled,
        query_jacobians,
        nullptr,
        nullptr,
        nullptr,
        damping,
        relative_eigen_floor
    );
    out["metadata"].cast<py::dict>()["fixed_pinhole_parameters"] = true;
    out["metadata"].cast<py::dict>()["spline_anchor_weight"] = 1000.0;
    out["metadata"].cast<py::dict>()["spline_smoothness_lambda"] =
        model_config.smoothness_lambda;
    out["metadata"].cast<py::dict>()["spline_regularizers_in_hessian"] = true;
    out["metadata"].cast<py::dict>()["spline_regularizers_in_empirical_gradients"] =
        false;
    return out;
}

}  // namespace lensboy
