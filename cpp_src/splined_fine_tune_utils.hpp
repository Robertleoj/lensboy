#pragma once

#include <ceres/ceres.h>
#include <algorithm>
#include <array>
#include <vector>

namespace lensboy {

struct SplineAnchorResidual {
    double weight;
    double target;
    int component;
    double basis[16];

    template <typename T>
    bool operator()(T const* const* knots, T* residuals) const {
        T value(0.0);
        for (int i = 0; i < 16; i++) {
            value += knots[i][component] * T(basis[i]);
        }
        residuals[0] = T(weight) * (value - T(target));
        return true;
    }
};

template <typename SplineMapT>
void add_spline_anchor_at_grid(
    ceres::Problem& problem,
    const SplineMapT& map,
    const std::vector<double*>& knot_blocks,
    double gx,
    double gy,
    bool constrain_dx,
    bool constrain_dy,
    double weight
) {
    const int ix = static_cast<int>(gx);
    const int iy = static_cast<int>(gy);
    if (!map.is_inside_fov(ix, iy)) {
        return;
    }

    double wx[4], wy[4];
    cubic_bspline_basis_uniform(gx - ix, wx);
    cubic_bspline_basis_uniform(gy - iy, wy);
    double basis[16];
    int basis_index = 0;
    for (int b = 0; b < 4; b++) {
        for (int a = 0; a < 4; a++) {
            basis[basis_index++] = wy[b] * wx[a];
        }
    }

    std::array<int, 16> flat{};
    map.support_indices_4x4(ix, iy, flat);
    auto make_anchor = [&](int component) {
        SplineAnchorResidual anchor{weight, 0.0, component, {}};
        std::copy(basis, basis + 16, anchor.basis);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<SplineAnchorResidual>(
            new SplineAnchorResidual(anchor)
        );
        std::vector<double*> blocks;
        for (int i = 0; i < 16; i++) {
            cost->AddParameterBlock(2);
            blocks.push_back(knot_blocks[flat[i]]);
        }
        cost->SetNumResiduals(1);
        problem.AddResidualBlock(cost, nullptr, blocks);
    };

    if (constrain_dx) {
        make_anchor(0);
    }
    if (constrain_dy) {
        make_anchor(1);
    }
}

struct KnotSmoothness2DResidual {
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

inline void add_grid_smoothness(
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
        }
    }
}

}  // namespace lensboy
