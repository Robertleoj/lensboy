#pragma once

#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace lensboy {

enum class InterpolationMode {
    kNearest,
    kBilinear,
    kBicubic,
};

enum class BoundsMode {
    kStrict,
    kClamp,
    kExtrapolate,
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
    double grid_stride_x = 0.0;
    double grid_stride_y = 0.0;
    std::string_view storage_encoding;
    std::string_view default_interpolation;
    std::string_view default_bounds;
    std::string_view lensboy_version;
    std::size_t payload_offset_bytes = 0;
};

struct UnprojectLUTQueryResult {
    bool valid = false;
    double ray[3] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
    };
};

class UnprojectLUT {
   public:
    static UnprojectLUT load(std::string_view path);

    UnprojectLUTMetadata const& metadata() const noexcept;

    UnprojectLUTQueryResult query(
        double pixel_x,
        double pixel_y,
        InterpolationMode interpolation = InterpolationMode::kBilinear,
        BoundsMode bounds = BoundsMode::kStrict,
        bool normalize = true
    ) const;

    std::vector<UnprojectLUTQueryResult> query(
        std::vector<PixelXY> const& pixels,
        InterpolationMode interpolation = InterpolationMode::kBilinear,
        BoundsMode bounds = BoundsMode::kStrict,
        bool normalize = true
    ) const;

   private:
    UnprojectLUT(
        UnprojectLUTMetadata metadata,
        std::vector<double> xy_grid,
        std::vector<char> string_storage
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

    std::vector<char> string_storage_;
    UnprojectLUTMetadata metadata_;
    std::vector<double> xy_grid_;
    double grid_scale_x_ = 0.0;
    double grid_scale_y_ = 0.0;
};

}  // namespace lensboy
