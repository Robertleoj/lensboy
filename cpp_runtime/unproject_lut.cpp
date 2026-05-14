#include "unproject_lut.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "npy.hpp"

namespace lensboy {
namespace {

bool is_finite(
    PixelXY const& value
) {
    return std::isfinite(value.xy[0]) and std::isfinite(value.xy[1]);
}

std::array<double, 4> catmull_rom_weights(
    double t
) {
    double const t2 = t * t;
    double const t3 = t2 * t;
    return {
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    };
}

PixelXY add_scaled(
    PixelXY const& a,
    PixelXY const& b,
    double scale
) {
    return {
        a.xy[0] + b.xy[0] * scale,
        a.xy[1] + b.xy[1] * scale,
    };
}

void normalize_ray(
    double ray[3]
) {
    double const norm = std::sqrt(
        ray[0] * ray[0] +
        ray[1] * ray[1] +
        ray[2] * ray[2]
    );
    if (norm == 0.0 or not std::isfinite(norm)) {
        throw std::runtime_error("Cannot normalize a non-finite or zero-length ray.");
    }
    ray[0] /= norm;
    ray[1] /= norm;
    ray[2] /= norm;
}

}  // namespace

UnprojectLUT::UnprojectLUT(
    UnprojectLUTMetadata metadata,
    std::vector<double> xy_grid
) : metadata_(std::move(metadata)),
    xy_grid_(std::move(xy_grid)) {
    if (metadata_.image_width < 2 or metadata_.image_height < 2) {
        throw std::runtime_error("Image dimensions must be at least 2.");
    }
    if (metadata_.grid_width < 2 or metadata_.grid_height < 2) {
        throw std::runtime_error("Grid dimensions must be at least 2 in each axis.");
    }
    if (xy_grid_.size() != metadata_.grid_width * metadata_.grid_height * 2) {
        throw std::runtime_error("xy_grid size does not match metadata.");
    }

    grid_scale_x_ = static_cast<double>(metadata_.grid_width - 1) /
                    static_cast<double>(metadata_.image_width - 1);
    grid_scale_y_ = static_cast<double>(metadata_.grid_height - 1) /
                    static_cast<double>(metadata_.image_height - 1);
}

UnprojectLUT UnprojectLUT::load(
    std::string_view const dir_path
) {
    std::string const dir(dir_path);
    std::string const metadata_path = dir + "/metadata.json";
    std::string const xy_grid_path = dir + "/xy_grid.npy";

    std::ifstream metadata_file(metadata_path);
    if (not metadata_file) {
        throw std::runtime_error("Failed to open metadata file: " + metadata_path);
    }
    nlohmann::json metadata = nlohmann::json::parse(metadata_file);

    std::string lensboy_version =
        metadata.at("lensboy-version").get<std::string>();
    auto const dot = lensboy_version.find('.');
    std::string const major_str = lensboy_version.substr(0, dot);
    int const major_version = std::stoi(major_str);
    if (major_version < 3) {
        throw std::runtime_error(
            "This unproject LUT was created with an incompatible version of "
            "lensboy (< 3.0.0). Please regenerate it with the current version."
        );
    }

    std::vector<unsigned long> shape;
    std::vector<float> raw;
    bool fortran_order = false;
    npy::LoadArrayFromNumpy<float>(xy_grid_path, shape, fortran_order, raw);
    if (fortran_order) {
        throw std::runtime_error("xy_grid.npy must be C-contiguous.");
    }

    if (shape.size() != 3 or shape[2] != 2) {
        throw std::runtime_error("xy_grid.npy must have shape (H, W, 2).");
    }
    std::size_t const grid_height = static_cast<std::size_t>(shape[0]);
    std::size_t const grid_width = static_cast<std::size_t>(shape[1]);

    std::vector<double> xy_grid(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (not std::isfinite(raw[i])) {
            throw std::runtime_error("xy_grid contains non-finite values.");
        }
        xy_grid[i] = static_cast<double>(raw[i]);
    }

    UnprojectLUTMetadata lut_metadata;
    lut_metadata.image_width =
        metadata.at("image_width").get<std::size_t>();
    lut_metadata.image_height =
        metadata.at("image_height").get<std::size_t>();
    lut_metadata.grid_width = grid_width;
    lut_metadata.grid_height = grid_height;
    lut_metadata.lensboy_version = std::move(lensboy_version);

    return UnprojectLUT(std::move(lut_metadata), std::move(xy_grid));
}

UnprojectLUTMetadata const& UnprojectLUT::metadata() const noexcept {
    return metadata_;
}

std::size_t UnprojectLUT::flat_index(
    std::size_t x,
    std::size_t y
) const noexcept {
    return (y * metadata_.grid_width + x) * 2;
}

PixelXY UnprojectLUT::sample_node(
    std::size_t x,
    std::size_t y
) const noexcept {
    std::size_t const idx = flat_index(x, y);
    return {{xy_grid_[idx], xy_grid_[idx + 1]}};
}

double UnprojectLUT::grid_coordinate_x(
    double pixel_x
) const noexcept {
    return pixel_x * grid_scale_x_;
}

double UnprojectLUT::grid_coordinate_y(
    double pixel_y
) const noexcept {
    return pixel_y * grid_scale_y_;
}

PixelXY UnprojectLUT::query_nearest(
    double gx,
    double gy
) const noexcept {
    long long const ix = std::llround(gx);
    long long const iy = std::llround(gy);
    std::size_t const sample_ix = static_cast<std::size_t>(
        std::clamp<long long>(ix, 0, static_cast<long long>(metadata_.grid_width) - 1)
    );
    std::size_t const sample_iy = static_cast<std::size_t>(
        std::clamp<long long>(iy, 0, static_cast<long long>(metadata_.grid_height) - 1)
    );
    return sample_node(sample_ix, sample_iy);
}

PixelXY UnprojectLUT::query_bilinear(
    double gx,
    double gy
) const noexcept {
    double const gx_work = std::clamp(
        gx,
        0.0,
        static_cast<double>(metadata_.grid_width - 1)
    );
    double const gy_work = std::clamp(
        gy,
        0.0,
        static_cast<double>(metadata_.grid_height - 1)
    );

    std::size_t const x0 = static_cast<std::size_t>(std::min(
        static_cast<long long>(std::floor(gx_work)),
        static_cast<long long>(metadata_.grid_width) - 2
    ));
    std::size_t const y0 = static_cast<std::size_t>(std::min(
        static_cast<long long>(std::floor(gy_work)),
        static_cast<long long>(metadata_.grid_height) - 2
    ));
    std::size_t const x1 = x0 + 1;
    std::size_t const y1 = y0 + 1;

    double const tx = gx_work - static_cast<double>(x0);
    double const ty = gy_work - static_cast<double>(y0);

    PixelXY const v00 = sample_node(x0, y0);
    PixelXY const v10 = sample_node(x1, y0);
    PixelXY const v01 = sample_node(x0, y1);
    PixelXY const v11 = sample_node(x1, y1);

    PixelXY const top = {{
        v00.xy[0] * (1.0 - tx) + v10.xy[0] * tx,
        v00.xy[1] * (1.0 - tx) + v10.xy[1] * tx,
    }};
    PixelXY const bottom = {{
        v01.xy[0] * (1.0 - tx) + v11.xy[0] * tx,
        v01.xy[1] * (1.0 - tx) + v11.xy[1] * tx,
    }};
    return {{
        top.xy[0] * (1.0 - ty) + bottom.xy[0] * ty,
        top.xy[1] * (1.0 - ty) + bottom.xy[1] * ty,
    }};
}

PixelXY UnprojectLUT::query_bicubic(
    double gx,
    double gy
) const noexcept {
    if (metadata_.grid_width < 4 or metadata_.grid_height < 4) {
        return query_bilinear(gx, gy);
    }

    double const gx_work =
        std::clamp(gx, 0.0, static_cast<double>(metadata_.grid_width - 1));
    double const gy_work =
        std::clamp(gy, 0.0, static_cast<double>(metadata_.grid_height - 1));

    long long const anchor_x = static_cast<long long>(std::floor(gx_work));
    long long const anchor_y = static_cast<long long>(std::floor(gy_work));
    bool const has_full_support =
        anchor_x >= 1 and
        anchor_x <= static_cast<long long>(metadata_.grid_width) - 3 and
        anchor_y >= 1 and
        anchor_y <= static_cast<long long>(metadata_.grid_height) - 3;
    if (not has_full_support) {
        return query_bilinear(gx, gy);
    }

    double const tx = gx_work - static_cast<double>(anchor_x);
    double const ty = gy_work - static_cast<double>(anchor_y);

    std::array<double, 4> const wx = catmull_rom_weights(tx);
    std::array<double, 4> const wy = catmull_rom_weights(ty);
    PixelXY accum = {{0.0, 0.0}};
    for (int j = 0; j < 4; ++j) {
        std::size_t const sample_y_idx =
            static_cast<std::size_t>(anchor_y + j - 1);
        PixelXY row = {{0.0, 0.0}};
        for (int i = 0; i < 4; ++i) {
            std::size_t const sample_x_idx =
                static_cast<std::size_t>(anchor_x + i - 1);
            PixelXY const node = sample_node(sample_x_idx, sample_y_idx);
            row = add_scaled(row, node, wx[i]);
        }
        accum = add_scaled(accum, row, wy[j]);
    }
    return accum;
}

UnprojectLUTQueryResult UnprojectLUT::query(
    double pixel_x,
    double pixel_y,
    InterpolationMode interpolation,
    bool const normalize
) const {
    bool const inside =
        pixel_x >= 0.0 and
        pixel_x <= static_cast<double>(metadata_.image_width - 1) and
        pixel_y >= 0.0 and
        pixel_y <= static_cast<double>(metadata_.image_height - 1);

    if (not inside) {
        return UnprojectLUTQueryResult::invalid();
    }

    double const gx = grid_coordinate_x(pixel_x);
    double const gy = grid_coordinate_y(pixel_y);

    PixelXY xy = {{0.0, 0.0}};
    switch (interpolation) {
        case InterpolationMode::NEAREST:
            xy = query_nearest(gx, gy);
            break;
        case InterpolationMode::BILINEAR:
            xy = query_bilinear(gx, gy);
            break;
        case InterpolationMode::BICUBIC:
            xy = query_bicubic(gx, gy);
            break;
    }

    if (not is_finite(xy)) {
        throw std::runtime_error("Query produced non-finite values.");
    }

    UnprojectLUTQueryResult result;
    result.valid = true;
    result.ray[0] = xy.xy[0];
    result.ray[1] = xy.xy[1];
    result.ray[2] = 1.0;
    if (normalize) {
        normalize_ray(result.ray);
    }
    return result;
}

std::vector<UnprojectLUTQueryResult> UnprojectLUT::query(
    std::vector<PixelXY> const& pixels,
    InterpolationMode interpolation,
    bool const normalize
) const {
    std::vector<UnprojectLUTQueryResult> results;
    results.reserve(pixels.size());
    for (PixelXY const& pixel : pixels) {
        results.push_back(
            query(pixel.xy[0], pixel.xy[1], interpolation, normalize)
        );
    }
    return results;
}

}  // namespace lensboy
