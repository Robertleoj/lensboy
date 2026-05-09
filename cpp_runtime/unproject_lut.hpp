#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace lensboy {

enum class InterpolationMode {
    kNearest,
    kBilinear,
    kBicubic,
};

struct PixelXY {
    double xy[2] = {0.0, 0.0};
};

struct UnprojectLUTMetadata {
    std::size_t image_width = 0;
    std::size_t image_height = 0;
    std::size_t grid_width = 0;
    std::size_t grid_height = 0;
    double grid_x_min = 0.0;
    double grid_x_max = 0.0;
    double grid_y_min = 0.0;
    double grid_y_max = 0.0;
    std::string lensboy_version;
};

struct UnprojectLUTQueryResult {
    bool valid;
    double ray[3];

    static UnprojectLUTQueryResult invalid() noexcept {
        double const nan = std::numeric_limits<double>::quiet_NaN();
        return UnprojectLUTQueryResult{false, {nan, nan, nan}};
    }
};

class UnprojectLUT {
   public:
    // Load a LUT from a directory containing metadata.json and xy_grid.npy.
    static UnprojectLUT load(std::string_view dir_path);

    UnprojectLUTMetadata const& metadata() const noexcept;

    UnprojectLUTQueryResult query(
        double pixel_x,
        double pixel_y,
        InterpolationMode interpolation = InterpolationMode::kBilinear,
        bool normalize = true
    ) const;

    std::vector<UnprojectLUTQueryResult> query(
        std::vector<PixelXY> const& pixels,
        InterpolationMode interpolation = InterpolationMode::kBilinear,
        bool normalize = true
    ) const;

   private:
    UnprojectLUT(
        UnprojectLUTMetadata metadata,
        std::vector<double> xy_grid
    );

    std::size_t flat_index(
        std::size_t x,
        std::size_t y
    ) const noexcept;

    PixelXY sample_node(
        std::size_t x,
        std::size_t y
    ) const noexcept;

    double grid_coordinate_x(
        double pixel_x
    ) const noexcept;

    double grid_coordinate_y(
        double pixel_y
    ) const noexcept;

    PixelXY query_nearest(
        double gx,
        double gy
    ) const noexcept;

    PixelXY query_bilinear(
        double gx,
        double gy
    ) const noexcept;

    PixelXY query_bicubic(
        double gx,
        double gy
    ) const noexcept;

    UnprojectLUTMetadata metadata_;
    std::vector<double> xy_grid_;
    double grid_scale_x_ = 0.0;
    double grid_scale_y_ = 0.0;
};

}  // namespace lensboy
