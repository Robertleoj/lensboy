#include <ceres/jet.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "./cameramodels.hpp"
#include "./pybind_utils.hpp"

namespace lensboy {
namespace py = pybind11;

// ---------------------------------------------------------------------------
// Hash grid for O(1) nearest-neighbor lookup over seed pixels.
//
// Cell size = minimum adjacent seed spacing.  Each cell holds a short list
// of seed indices.  Query checks the cell + 8 neighbors.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Flat-array spatial index for O(1) nearest-neighbor lookup.
//
// Cell size = min adjacent seed spacing.  Flat 2D array of short vectors.
// Query checks the cell + 8 neighbors.
// ---------------------------------------------------------------------------

class SeedGrid {
   public:
    static constexpr int CELL_CAPACITY = 4;

    SeedGrid(
        const double* pixel_xy,
        int num_seeds,
        int seed_width,
        int seed_height
    ) {
        // Find min adjacent edge length in pixel space
        double min_edge_squared = std::numeric_limits<double>::max();
        for (int row = 0; row < seed_height; row++) {
            for (int col = 0; col < seed_width; col++) {
                int index = row * seed_width + col;
                double seed_x = pixel_xy[index * 2];
                double seed_y = pixel_xy[index * 2 + 1];
                if (!std::isfinite(seed_x)) {
                    continue;
                }
                if (col + 1 < seed_width) {
                    int neighbor_index = row * seed_width + (col + 1);
                    double neighbor_x = pixel_xy[neighbor_index * 2];
                    double neighbor_y = pixel_xy[neighbor_index * 2 + 1];
                    if (std::isfinite(neighbor_x)) {
                        double distance_squared =
                            (neighbor_x - seed_x) * (neighbor_x - seed_x) +
                            (neighbor_y - seed_y) * (neighbor_y - seed_y);
                        if (distance_squared > 0) {
                            min_edge_squared =
                                std::min(min_edge_squared, distance_squared);
                        }
                    }
                }
                if (row + 1 < seed_height) {
                    int neighbor_index = (row + 1) * seed_width + col;
                    double neighbor_x = pixel_xy[neighbor_index * 2];
                    double neighbor_y = pixel_xy[neighbor_index * 2 + 1];
                    if (std::isfinite(neighbor_x)) {
                        double distance_squared =
                            (neighbor_x - seed_x) * (neighbor_x - seed_x) +
                            (neighbor_y - seed_y) * (neighbor_y - seed_y);
                        if (distance_squared > 0) {
                            min_edge_squared =
                                std::min(min_edge_squared, distance_squared);
                        }
                    }
                }
            }
        }
        cell_size_ = std::sqrt(min_edge_squared);
        if (cell_size_ < 1e-6) {
            cell_size_ = 1.0;
        }
        inverse_cell_size_ = 1.0 / cell_size_;

        // Bounding box
        double pixel_x_min = 1e30, pixel_x_max = -1e30;
        double pixel_y_min = 1e30, pixel_y_max = -1e30;
        for (int i = 0; i < num_seeds; i++) {
            double seed_x = pixel_xy[i * 2];
            double seed_y = pixel_xy[i * 2 + 1];
            if (!std::isfinite(seed_x)) {
                continue;
            }
            pixel_x_min = std::min(pixel_x_min, seed_x);
            pixel_x_max = std::max(pixel_x_max, seed_x);
            pixel_y_min = std::min(pixel_y_min, seed_y);
            pixel_y_max = std::max(pixel_y_max, seed_y);
        }
        pixel_x_min_ = pixel_x_min;
        pixel_y_min_ = pixel_y_min;
        grid_width_ =
            (int)std::ceil((pixel_x_max - pixel_x_min) * inverse_cell_size_) +
            1;
        grid_height_ =
            (int)std::ceil((pixel_y_max - pixel_y_min) * inverse_cell_size_) +
            1;

        // Flat array, each cell stores up to CELL_CAPACITY indices (enough for
        // ~1 point per cell). Use -1 as sentinel.
        cells_.assign((size_t)grid_width_ * grid_height_ * CELL_CAPACITY, -1);

        for (int i = 0; i < num_seeds; i++) {
            double seed_x = pixel_xy[i * 2];
            double seed_y = pixel_xy[i * 2 + 1];
            if (!std::isfinite(seed_x) || !std::isfinite(seed_y)) {
                continue;
            }
            int cell_x = to_cell_x(seed_x);
            int cell_y = to_cell_y(seed_y);
            int base = (cell_y * grid_width_ + cell_x) * CELL_CAPACITY;
            for (int slot = 0; slot < CELL_CAPACITY; slot++) {
                if (cells_[base + slot] < 0) {
                    cells_[base + slot] = i;
                    break;
                }
            }
        }
    }

    int nearest(
        double query_x,
        double query_y,
        const double* pixel_xy
    ) const {
        int cell_x = to_cell_x(query_x);
        int cell_y = to_cell_y(query_y);
        int best_index = -1;
        double best_distance_squared = std::numeric_limits<double>::max();
        for (int delta_y = -1; delta_y <= 1; delta_y++) {
            int neighbor_y = cell_y + delta_y;
            if (neighbor_y < 0 || neighbor_y >= grid_height_) {
                continue;
            }
            for (int delta_x = -1; delta_x <= 1; delta_x++) {
                int neighbor_x = cell_x + delta_x;
                if (neighbor_x < 0 || neighbor_x >= grid_width_) {
                    continue;
                }
                int base =
                    (neighbor_y * grid_width_ + neighbor_x) * CELL_CAPACITY;
                for (int slot = 0; slot < CELL_CAPACITY; slot++) {
                    int seed_index = cells_[base + slot];
                    if (seed_index < 0) {
                        break;
                    }
                    double delta_pixel_x = query_x - pixel_xy[seed_index * 2];
                    double delta_pixel_y =
                        query_y - pixel_xy[seed_index * 2 + 1];
                    double distance_squared = delta_pixel_x * delta_pixel_x +
                                              delta_pixel_y * delta_pixel_y;
                    if (distance_squared < best_distance_squared) {
                        best_distance_squared = distance_squared;
                        best_index = seed_index;
                    }
                }
            }
        }
        return best_index;
    }

   private:
    double cell_size_, inverse_cell_size_, pixel_x_min_, pixel_y_min_;
    int grid_width_, grid_height_;
    std::vector<int> cells_;

    int to_cell_x(
        double x
    ) const {
        return std::max(
            0,
            std::min(
                grid_width_ - 1,
                (int)std::floor((x - pixel_x_min_) * inverse_cell_size_)
            )
        );
    }
    int to_cell_y(
        double y
    ) const {
        return std::max(
            0,
            std::min(
                grid_height_ - 1,
                (int)std::floor((y - pixel_y_min_) * inverse_cell_size_)
            )
        );
    }
};

// ---------------------------------------------------------------------------
// Bilinear interpolation of seed normals over the quad surrounding a query.
//
// The NN tells us grid node (nearest_index_x, nearest_index_y). The query
// pixel's position relative to that node picks one of the 4 adjacent quads.
// Local (fractional_x, fractional_y) coords inside that quad are recovered by
// projecting the query onto the quad's two edge vectors, then we bilinearly
// interpolate the 4 corner seed-normals -- no iteration needed.
// ---------------------------------------------------------------------------

static bool bilinear_interpolate_normal_in_quad(
    double pixel_x,
    double pixel_y,
    int nearest_index,
    int seed_width,
    int seed_height,
    const double* seed_pixels,
    const double* seed_normals,
    double& normalized_x_out,
    double& normalized_y_out
) {
    int nearest_index_x = nearest_index % seed_width;
    int nearest_index_y = nearest_index / seed_width;
    double nearest_pixel_x = seed_pixels[nearest_index * 2];
    double nearest_pixel_y = seed_pixels[nearest_index * 2 + 1];

    // Pick the quad based on which side of the NN the query falls
    int quad_index_x =
        (pixel_x >= nearest_pixel_x) ? nearest_index_x : nearest_index_x - 1;
    int quad_index_y =
        (pixel_y >= nearest_pixel_y) ? nearest_index_y : nearest_index_y - 1;
    // Clamp to valid quad range
    quad_index_x = std::max(0, std::min(seed_width - 2, quad_index_x));
    quad_index_y = std::max(0, std::min(seed_height - 2, quad_index_y));

    int corner_indices[4] = {
        quad_index_y * seed_width + quad_index_x,
        quad_index_y * seed_width + (quad_index_x + 1),
        (quad_index_y + 1) * seed_width + quad_index_x,
        (quad_index_y + 1) * seed_width + (quad_index_x + 1),
    };

    // Check all 4 corners are valid
    for (int corner = 0; corner < 4; corner++) {
        if (!std::isfinite(seed_pixels[corner_indices[corner] * 2])) {
            return false;
        }
    }

    // Approximate bilinear (fractional_x, fractional_y) by projecting onto the
    // quad edges.
    const double* pixel_x0_y0 = &seed_pixels[corner_indices[0] * 2];
    const double* pixel_x1_y0 = &seed_pixels[corner_indices[1] * 2];
    const double* pixel_x0_y1 = &seed_pixels[corner_indices[2] * 2];
    double edge_x_pixel_x = pixel_x1_y0[0] - pixel_x0_y0[0];
    double edge_x_pixel_y = pixel_x1_y0[1] - pixel_x0_y0[1];
    double edge_y_pixel_x = pixel_x0_y1[0] - pixel_x0_y0[0];
    double edge_y_pixel_y = pixel_x0_y1[1] - pixel_x0_y0[1];
    double query_offset_x = pixel_x - pixel_x0_y0[0];
    double query_offset_y = pixel_y - pixel_x0_y0[1];

    double determinant =
        edge_x_pixel_x * edge_y_pixel_y - edge_y_pixel_x * edge_x_pixel_y;
    if (std::abs(determinant) < 1e-30) {
        return false;
    }
    double inverse_determinant = 1.0 / determinant;
    double fractional_x = std::max(
        0.0,
        std::min(
            1.0,
            (query_offset_x * edge_y_pixel_y - edge_y_pixel_x * query_offset_y
            ) * inverse_determinant
        )
    );
    double fractional_y = std::max(
        0.0,
        std::min(
            1.0,
            (edge_x_pixel_x * query_offset_y - query_offset_x * edge_x_pixel_y
            ) * inverse_determinant
        )
    );

    double one_minus_fractional_x = 1.0 - fractional_x;
    double one_minus_fractional_y = 1.0 - fractional_y;
    const double* normal_x0_y0 = &seed_normals[corner_indices[0] * 2];
    const double* normal_x1_y0 = &seed_normals[corner_indices[1] * 2];
    const double* normal_x0_y1 = &seed_normals[corner_indices[2] * 2];
    const double* normal_x1_y1 = &seed_normals[corner_indices[3] * 2];
    normalized_x_out =
        one_minus_fractional_x * one_minus_fractional_y * normal_x0_y0[0] +
        fractional_x * one_minus_fractional_y * normal_x1_y0[0] +
        one_minus_fractional_x * fractional_y * normal_x0_y1[0] +
        fractional_x * fractional_y * normal_x1_y1[0];
    normalized_y_out =
        one_minus_fractional_x * one_minus_fractional_y * normal_x0_y0[1] +
        fractional_x * one_minus_fractional_y * normal_x1_y0[1] +
        one_minus_fractional_x * fractional_y * normal_x0_y1[1] +
        fractional_x * fractional_y * normal_x1_y1[1];
    return true;
}

// ---------------------------------------------------------------------------
// Newton refinement: generic 2D solver using Ceres Jet autodiff.
// ---------------------------------------------------------------------------

// Forward-project (normalized_x, normalized_y) -> (pixel_x, pixel_y) for the
// OpenCV model.
template <typename T>
static inline void forward_opencv(
    const T& normalized_x,
    const T& normalized_y,
    const T* intrinsics,  // fx, fy, cx, cy, dist[14]
    T& pixel_x,
    T& pixel_y
) {
    Vec3<T> point(normalized_x, normalized_y, T(1));
    Vec2<T> result;
    project_opencv(intrinsics, point, result);
    pixel_x = result[0];
    pixel_y = result[1];
}

// Forward-project (normalized_x, normalized_y) -> (pixel_x, pixel_y) for the
// pinhole-splined model.
template <typename T>
static inline void forward_splined(
    const T& normalized_x,
    const T& normalized_y,
    PinholeSplinedModelDefinition* config,
    const T* pinhole_params,
    const T* dx_grid,
    const T* dy_grid,
    T& pixel_x,
    T& pixel_y
) {
    Vec3<T> point(normalized_x, normalized_y, T(1));
    Vec2<T> result;
    project_pinhole_splined(
        config,
        pinhole_params,
        dx_grid,
        dy_grid,
        point,
        result
    );
    pixel_x = result[0];
    pixel_y = result[1];
}

// Newton refinement for OpenCV model, starting from initial guess
// (normalized_x, normalized_y).
static void refine_opencv(
    double target_pixel_x,
    double target_pixel_y,
    double& normalized_x,
    double& normalized_y,
    const double* intrinsics  // fx, fy, cx, cy, dist[14]
) {
    constexpr int max_iterations = 20;
    constexpr double tolerance_squared = 1e-14;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        double pixel_x, pixel_y;
        forward_opencv(
            normalized_x,
            normalized_y,
            intrinsics,
            pixel_x,
            pixel_y
        );

        double residual_x = pixel_x - target_pixel_x;
        double residual_y = pixel_y - target_pixel_y;
        if (residual_x * residual_x + residual_y * residual_y <
            tolerance_squared) {
            break;
        }

        const double step_x = 1e-6 * std::max(1.0, std::abs(normalized_x));
        const double step_y = 1e-6 * std::max(1.0, std::abs(normalized_y));
        double plus_x_x, plus_x_y, minus_x_x, minus_x_y;
        double plus_y_x, plus_y_y, minus_y_x, minus_y_y;
        forward_opencv(
            normalized_x + step_x,
            normalized_y,
            intrinsics,
            plus_x_x,
            plus_x_y
        );
        forward_opencv(
            normalized_x - step_x,
            normalized_y,
            intrinsics,
            minus_x_x,
            minus_x_y
        );
        forward_opencv(
            normalized_x,
            normalized_y + step_y,
            intrinsics,
            plus_y_x,
            plus_y_y
        );
        forward_opencv(
            normalized_x,
            normalized_y - step_y,
            intrinsics,
            minus_y_x,
            minus_y_y
        );
        double jacobian_x_x = (plus_x_x - minus_x_x) / (2.0 * step_x);
        double jacobian_y_x = (plus_x_y - minus_x_y) / (2.0 * step_x);
        double jacobian_x_y = (plus_y_x - minus_y_x) / (2.0 * step_y);
        double jacobian_y_y = (plus_y_y - minus_y_y) / (2.0 * step_y);
        double determinant =
            jacobian_x_x * jacobian_y_y - jacobian_x_y * jacobian_y_x;
        if (std::abs(determinant) < 1e-30) {
            break;
        }
        double inverse_determinant = 1.0 / determinant;
        normalized_x -= inverse_determinant *
                        (jacobian_y_y * residual_x - jacobian_x_y * residual_y);
        normalized_y -= inverse_determinant * (-jacobian_y_x * residual_x +
                                               jacobian_x_x * residual_y);
    }
}

struct SplineConstants {
    int num_knots_x, num_knots_y;
    double stereo_half_range_x, stereo_half_range_y;
    double stereo_to_grid_scale_x, stereo_to_grid_scale_y;
    double fx, fy, cx, cy;

    explicit SplineConstants(
        PinholeSplinedModelDefinition* config,
        const double* pinhole_params
    )
        : num_knots_x((int)config->num_knots_x),
          num_knots_y((int)config->num_knots_y),
          fx(pinhole_params[0]),
          fy(pinhole_params[1]),
          cx(pinhole_params[2]),
          cy(pinhole_params[3]) {
        const double fov_rad_x = config->fov_deg_x * M_PI / 180.0;
        const double fov_rad_y = config->fov_deg_y * M_PI / 180.0;
        stereo_half_range_x = stereo_half_range(fov_rad_x);
        stereo_half_range_y = stereo_half_range(fov_rad_y);
        stereo_to_grid_scale_x =
            (num_knots_x - 3) / (2.0 * stereo_half_range_x);
        stereo_to_grid_scale_y =
            (num_knots_y - 3) / (2.0 * stereo_half_range_y);
    }
};

// Newton refinement for pinhole-splined model.
static void refine_splined(
    double target_pixel_x,
    double target_pixel_y,
    double& normalized_x,
    double& normalized_y,
    const SplineConstants& spline_constants,
    const double* dx_grid,
    const double* dy_grid,
    int* out_num_rebuilds = nullptr,
    int* out_num_iterations = nullptr
) {
    using Jet = ceres::Jet<double, 2>;
    constexpr int max_newton_iterations = 15;
    constexpr int max_rebuilds = 5;
    constexpr double tolerance_squared = 1e-20;
    constexpr double epsilon = 1e-12;

    const int num_knots_x = spline_constants.num_knots_x;
    const int num_knots_y = spline_constants.num_knots_y;
    const double stereo_half_range_x = spline_constants.stereo_half_range_x;
    const double stereo_half_range_y = spline_constants.stereo_half_range_y;
    const double stereo_to_grid_scale_x =
        spline_constants.stereo_to_grid_scale_x;
    const double stereo_to_grid_scale_y =
        spline_constants.stereo_to_grid_scale_y;

    int num_rebuilds = 0, num_iterations = 0;
    for (int rebuild = 0; rebuild < max_rebuilds; rebuild++) {
        num_rebuilds++;
        double stereographic_x, stereographic_y;
        normalized_to_stereographic(
            normalized_x,
            normalized_y,
            stereographic_x,
            stereographic_y
        );
        double grid_x = std::max(
            0.0,
            std::min(
                1.0 + (stereographic_x + stereo_half_range_x) *
                          stereo_to_grid_scale_x,
                num_knots_x - 1.0 - epsilon
            )
        );
        double grid_y = std::max(
            0.0,
            std::min(
                1.0 + (stereographic_y + stereo_half_range_y) *
                          stereo_to_grid_scale_y,
                num_knots_y - 1.0 - epsilon
            )
        );
        const int base_index_x = (int)std::floor(grid_x);
        const int base_index_y = (int)std::floor(grid_y);

        double local_distortion_x[16], local_distortion_y[16];
        int flat_index = 0;
        for (int row = 0; row < 4; row++) {
            const int clamped_y =
                clamp_int(base_index_y + row - 1, 0, num_knots_y - 1);
            for (int col = 0; col < 4; col++) {
                const int clamped_x =
                    clamp_int(base_index_x + col - 1, 0, num_knots_x - 1);
                local_distortion_x[flat_index] =
                    dx_grid[clamped_y * num_knots_x + clamped_x];
                local_distortion_y[flat_index] =
                    dy_grid[clamped_y * num_knots_x + clamped_x];
                flat_index++;
            }
        }

        for (int iteration = 0; iteration < max_newton_iterations;
             iteration++) {
            num_iterations++;
            Jet jet_normalized_x(normalized_x, 0);
            Jet jet_normalized_y(normalized_y, 1);
            Jet jet_stereographic_x, jet_stereographic_y;
            normalized_to_stereographic(
                jet_normalized_x,
                jet_normalized_y,
                jet_stereographic_x,
                jet_stereographic_y
            );
            Jet jet_grid_x = clamp_T(
                Jet(1.0) + (jet_stereographic_x + Jet(stereo_half_range_x)) *
                               Jet(stereo_to_grid_scale_x),
                Jet(0.0),
                Jet(num_knots_x - 1.0 - epsilon)
            );
            Jet jet_grid_y = clamp_T(
                Jet(1.0) + (jet_stereographic_y + Jet(stereo_half_range_y)) *
                               Jet(stereo_to_grid_scale_y),
                Jet(0.0),
                Jet(num_knots_y - 1.0 - epsilon)
            );
            Jet jet_fractional_x = jet_grid_x - Jet((double)base_index_x);
            Jet jet_fractional_y = jet_grid_y - Jet((double)base_index_y);
            Jet weights_x[4], weights_y[4];
            cubic_bspline_basis_uniform(jet_fractional_x, weights_x);
            cubic_bspline_basis_uniform(jet_fractional_y, weights_y);
            Jet distortion_x(0.0), distortion_y(0.0);
            int stencil_index = 0;
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    Jet weight = weights_y[row] * weights_x[col];
                    distortion_x +=
                        Jet(local_distortion_x[stencil_index]) * weight;
                    distortion_y +=
                        Jet(local_distortion_y[stencil_index]) * weight;
                    stencil_index++;
                }
            }
            Jet residual_jet_x =
                Jet(spline_constants.fx) * (jet_normalized_x + distortion_x) +
                Jet(spline_constants.cx) - Jet(target_pixel_x);
            Jet residual_jet_y =
                Jet(spline_constants.fy) * (jet_normalized_y + distortion_y) +
                Jet(spline_constants.cy) - Jet(target_pixel_y);
            double residual_x = residual_jet_x.a;
            double residual_y = residual_jet_y.a;
            if (residual_x * residual_x + residual_y * residual_y <
                tolerance_squared) {
                break;
            }
            double jacobian_x_x = residual_jet_x.v[0];
            double jacobian_x_y = residual_jet_x.v[1];
            double jacobian_y_x = residual_jet_y.v[0];
            double jacobian_y_y = residual_jet_y.v[1];
            double determinant =
                jacobian_x_x * jacobian_y_y - jacobian_x_y * jacobian_y_x;
            if (std::abs(determinant) < 1e-30) {
                break;
            }
            double inverse_determinant = 1.0 / determinant;
            normalized_x -= inverse_determinant * (jacobian_y_y * residual_x -
                                                   jacobian_x_y * residual_y);
            normalized_y -= inverse_determinant * (-jacobian_y_x * residual_x +
                                                   jacobian_x_x * residual_y);
        }

        normalized_to_stereographic(
            normalized_x,
            normalized_y,
            stereographic_x,
            stereographic_y
        );
        grid_x = std::max(
            0.0,
            std::min(
                1.0 + (stereographic_x + stereo_half_range_x) *
                          stereo_to_grid_scale_x,
                num_knots_x - 1.0 - epsilon
            )
        );
        grid_y = std::max(
            0.0,
            std::min(
                1.0 + (stereographic_y + stereo_half_range_y) *
                          stereo_to_grid_scale_y,
                num_knots_y - 1.0 - epsilon
            )
        );
        if ((int)std::floor(grid_x) == base_index_x &&
            (int)std::floor(grid_y) == base_index_y) {
            break;
        }
    }
    if (out_num_rebuilds) {
        *out_num_rebuilds = num_rebuilds;
    }
    if (out_num_iterations) {
        *out_num_iterations = num_iterations;
    }
}

// ---------------------------------------------------------------------------
// Shared kernel: validation, NN + bilinear initial guess, parallel Newton
// refinement.
// Per-model parts (intrinsics parsing, residual+jacobian) live in the entry
// points and the refine_* functions above.
// ---------------------------------------------------------------------------

namespace {

void validate_seeded_inputs(
    const py::buffer_info& seed_pixels_buffer,
    const py::buffer_info& seed_normals_buffer,
    const py::buffer_info& query_pixels_buffer,
    int seed_width,
    int seed_height
) {
    require(
        seed_pixels_buffer.ndim == 2 && seed_pixels_buffer.shape[1] == 2,
        "seed_pixels must be (M, 2)"
    );
    require(
        seed_normals_buffer.ndim == 2 && seed_normals_buffer.shape[1] == 2,
        "seed_normals must be (M, 2)"
    );
    require(
        seed_pixels_buffer.shape[0] == seed_normals_buffer.shape[0],
        "seed_pixels and seed_normals must have same length"
    );
    require(
        seed_pixels_buffer.shape[0] == (ssize_t)(seed_width * seed_height),
        "seed length must equal seed_width * seed_height"
    );
    require(
        query_pixels_buffer.ndim == 2 && query_pixels_buffer.shape[1] == 2,
        "query_pixels must be (N, 2)"
    );
}

template <typename Refiner>
py::array_t<double> run_seeded_normalize(
    const double* seed_pixels,
    const double* seed_normals,
    int num_seeds,
    int seed_width,
    int seed_height,
    const double* query_pixels,
    ssize_t num_queries,
    double fx,
    double fy,
    double cx,
    double cy,
    Refiner refine
) {
    py::array_t<double> out({num_queries, (ssize_t)3});
    double* output_data = static_cast<double*>(out.request().ptr);

    SeedGrid grid(seed_pixels, num_seeds, seed_width, seed_height);

    py::gil_scoped_release release;

    // clang-format off
    #pragma omp parallel for schedule(static)
    // clang-format on
    for (ssize_t i = 0; i < num_queries; i++) {
        double pixel_x = query_pixels[i * 2];
        double pixel_y = query_pixels[i * 2 + 1];

        double normalized_x = (pixel_x - cx) / fx;
        double normalized_y = (pixel_y - cy) / fy;

        int nearest_index = grid.nearest(pixel_x, pixel_y, seed_pixels);
        if (nearest_index >= 0) {
            double interp_normalized_x, interp_normalized_y;
            if (bilinear_interpolate_normal_in_quad(
                    pixel_x,
                    pixel_y,
                    nearest_index,
                    seed_width,
                    seed_height,
                    seed_pixels,
                    seed_normals,
                    interp_normalized_x,
                    interp_normalized_y
                )) {
                normalized_x = interp_normalized_x;
                normalized_y = interp_normalized_y;
            } else {
                normalized_x = seed_normals[nearest_index * 2];
                normalized_y = seed_normals[nearest_index * 2 + 1];
            }
        }

        refine(pixel_x, pixel_y, normalized_x, normalized_y);

        output_data[i * 3 + 0] = normalized_x;
        output_data[i * 3 + 1] = normalized_y;
        output_data[i * 3 + 2] = 1.0;
    }

    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Python-facing entry points.
// ---------------------------------------------------------------------------

py::array_t<double> seeded_normalize_opencv(
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_normals,
    int seed_width,
    int seed_height,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics
) {
    auto seed_pixels_buffer = seed_pixels.request();
    auto seed_normals_buffer = seed_normals.request();
    auto query_pixels_buffer = query_pixels.request();
    auto intrinsics_buffer = intrinsics.request();

    validate_seeded_inputs(
        seed_pixels_buffer,
        seed_normals_buffer,
        query_pixels_buffer,
        seed_width,
        seed_height
    );
    require(
        intrinsics_buffer.ndim == 1 && intrinsics_buffer.shape[0] == 18,
        "intrinsics must be (18,): fx, fy, cx, cy, dist[14]"
    );

    const int num_seeds = (int)seed_pixels_buffer.shape[0];
    const ssize_t num_queries = query_pixels_buffer.shape[0];
    const double* seed_pixels_data =
        static_cast<const double*>(seed_pixels_buffer.ptr);
    const double* seed_normals_data =
        static_cast<const double*>(seed_normals_buffer.ptr);
    const double* query_pixels_data =
        static_cast<const double*>(query_pixels_buffer.ptr);
    const double* intrinsics_data =
        static_cast<const double*>(intrinsics_buffer.ptr);

    return run_seeded_normalize(
        seed_pixels_data,
        seed_normals_data,
        num_seeds,
        seed_width,
        seed_height,
        query_pixels_data,
        num_queries,
        intrinsics_data[0],
        intrinsics_data[1],
        intrinsics_data[2],
        intrinsics_data[3],
        [intrinsics_data](
            double target_pixel_x,
            double target_pixel_y,
            double& normalized_x,
            double& normalized_y
        ) {
            refine_opencv(
                target_pixel_x,
                target_pixel_y,
                normalized_x,
                normalized_y,
                intrinsics_data
            );
        }
    );
}

py::array_t<double> seeded_normalize_splined(
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_pixels,
    py::array_t<double, py::array::c_style | py::array::forcecast> seed_normals,
    int seed_width,
    int seed_height,
    py::array_t<double, py::array::c_style | py::array::forcecast> query_pixels,
    PinholeSplinedModelDefinition& config,
    PinholeSplinedIntrinsicsParameters& params
) {
    auto seed_pixels_buffer = seed_pixels.request();
    auto seed_normals_buffer = seed_normals.request();
    auto query_pixels_buffer = query_pixels.request();

    validate_seeded_inputs(
        seed_pixels_buffer,
        seed_normals_buffer,
        query_pixels_buffer,
        seed_width,
        seed_height
    );

    auto dx_grid_buffer = params.dx_grid.request();
    auto dy_grid_buffer = params.dy_grid.request();
    require(
        (uint32_t)dx_grid_buffer.shape[0] == config.num_knots_y &&
            (uint32_t)dx_grid_buffer.shape[1] == config.num_knots_x,
        "dx_grid shape mismatch"
    );
    require(
        (uint32_t)dy_grid_buffer.shape[0] == config.num_knots_y &&
            (uint32_t)dy_grid_buffer.shape[1] == config.num_knots_x,
        "dy_grid shape mismatch"
    );

    auto pinhole_parameters_buffer = params.pinhole_parameters.request();
    require(
        pinhole_parameters_buffer.ndim == 1 &&
            pinhole_parameters_buffer.shape[0] == 4,
        "pinhole_parameters must be (4,)"
    );
    const double* pinhole_params =
        static_cast<const double*>(pinhole_parameters_buffer.ptr);
    const double fx = pinhole_params[0], fy = pinhole_params[1],
                 cx = pinhole_params[2], cy = pinhole_params[3];
    require(fx != 0.0 && fy != 0.0, "fx/fy must be non-zero");

    const double* dx_grid_data = static_cast<const double*>(dx_grid_buffer.ptr);
    const double* dy_grid_data = static_cast<const double*>(dy_grid_buffer.ptr);

    const int num_seeds = (int)seed_pixels_buffer.shape[0];
    const ssize_t num_queries = query_pixels_buffer.shape[0];
    const double* seed_pixels_data =
        static_cast<const double*>(seed_pixels_buffer.ptr);
    const double* seed_normals_data =
        static_cast<const double*>(seed_normals_buffer.ptr);
    const double* query_pixels_data =
        static_cast<const double*>(query_pixels_buffer.ptr);

    SplineConstants spline_constants(&config, pinhole_params);

    return run_seeded_normalize(
        seed_pixels_data,
        seed_normals_data,
        num_seeds,
        seed_width,
        seed_height,
        query_pixels_data,
        num_queries,
        fx,
        fy,
        cx,
        cy,
        [&spline_constants, dx_grid_data, dy_grid_data](
            double target_pixel_x,
            double target_pixel_y,
            double& normalized_x,
            double& normalized_y
        ) {
            refine_splined(
                target_pixel_x,
                target_pixel_y,
                normalized_x,
                normalized_y,
                spline_constants,
                dx_grid_data,
                dy_grid_data
            );
        }
    );
}

}  // namespace lensboy
