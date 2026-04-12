#include "unproject_lut.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "npy.hpp"

namespace lensboy {
namespace {

double quiet_nan() {
    return std::numeric_limits<double>::quiet_NaN();
}

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

std::string_view append_string_view(
    std::vector<char>& storage,
    std::string const& text
) {
    if (text.empty()) {
        return {};
    }
    std::size_t const offset = storage.size();
    storage.insert(storage.end(), text.begin(), text.end());
    storage.push_back('\0');
    return std::string_view(storage.data() + offset, text.size());
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

UnprojectLUTQueryResult invalid_result() {
    return UnprojectLUTQueryResult{};
}

}  // namespace

UnprojectLUT::UnprojectLUT(
    UnprojectLUTMetadata metadata,
    std::vector<double> xy_grid,
    std::vector<char> string_storage
) : string_storage_(std::move(string_storage)),
    metadata_(std::move(metadata)),
    xy_grid_(std::move(xy_grid)) {
    if (metadata_.image_width == 0 or metadata_.image_height == 0) {
        throw std::runtime_error("Image dimensions must be positive.");
    }
    if (metadata_.grid_width == 0 or metadata_.grid_height == 0) {
        throw std::runtime_error("Grid dimensions must be positive.");
    }
    if (
        metadata_.grid_x_max < metadata_.grid_x_min or
        metadata_.grid_y_max < metadata_.grid_y_min
    ) {
        throw std::runtime_error("Grid extents must be ordered from min to max.");
    }
    if (xy_grid_.size() != metadata_.grid_width * metadata_.grid_height * 2) {
        throw std::runtime_error("xy_grid size does not match metadata.");
    }

    if (metadata_.grid_width <= 1 or metadata_.grid_x_max == metadata_.grid_x_min) {
        grid_scale_x_ = 0.0;
    } else {
        grid_scale_x_ = static_cast<double>(metadata_.grid_width - 1) /
                        (metadata_.grid_x_max - metadata_.grid_x_min);
    }
    if (
        metadata_.grid_height <= 1 or
        metadata_.grid_y_max == metadata_.grid_y_min
    ) {
        grid_scale_y_ = 0.0;
    } else {
        grid_scale_y_ = static_cast<double>(metadata_.grid_height - 1) /
                        (metadata_.grid_y_max - metadata_.grid_y_min);
    }
}

namespace {

}  // namespace

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

    std::string const lensboy_version =
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

    std::vector<char> string_storage;
    string_storage.reserve(lensboy_version.size() + 1);
    auto const append = [&string_storage](std::string const& text) {
        if (text.empty()) return std::string_view{};
        std::size_t const offset = string_storage.size();
        string_storage.insert(string_storage.end(), text.begin(), text.end());
        string_storage.push_back('\0');
        return std::string_view(string_storage.data() + offset, text.size());
    };

    UnprojectLUTMetadata lut_metadata;
    lut_metadata.image_width =
        metadata.at("image_width").get<std::size_t>();
    lut_metadata.image_height =
        metadata.at("image_height").get<std::size_t>();
    lut_metadata.grid_width = grid_width;
    lut_metadata.grid_height = grid_height;
    lut_metadata.grid_x_min = metadata.at("grid_x_min").get<double>();
    lut_metadata.grid_x_max = metadata.at("grid_x_max").get<double>();
    lut_metadata.grid_y_min = metadata.at("grid_y_min").get<double>();
    lut_metadata.grid_y_max = metadata.at("grid_y_max").get<double>();
    lut_metadata.lensboy_version = append(lensboy_version);

    return UnprojectLUT(
        std::move(lut_metadata),
        std::move(xy_grid),
        std::move(string_storage)
    );
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
    return (pixel_x - metadata_.grid_x_min) * grid_scale_x_;
}

double UnprojectLUT::grid_coordinate_y(
    double pixel_y
) const noexcept {
    return (pixel_y - metadata_.grid_y_min) * grid_scale_y_;
}

UnprojectLUTQueryResult UnprojectLUT::query(
    double pixel_x,
    double pixel_y,
    InterpolationMode interpolation,
    BoundsMode bounds,
    bool const normalize
) const {
    bool const inside = pixel_x >= metadata_.grid_x_min and
                        pixel_x <= metadata_.grid_x_max and
                        pixel_y >= metadata_.grid_y_min and
                        pixel_y <= metadata_.grid_y_max;

    if (bounds == BoundsMode::kStrict and not inside) {
        return invalid_result();
    }

    double sample_x = pixel_x;
    double sample_y = pixel_y;
    if (bounds == BoundsMode::kClamp) {
        sample_x = std::clamp(sample_x, metadata_.grid_x_min, metadata_.grid_x_max);
        sample_y = std::clamp(sample_y, metadata_.grid_y_min, metadata_.grid_y_max);
    }

    double const gx = grid_coordinate_x(sample_x);
    double const gy = grid_coordinate_y(sample_y);

    PixelXY xy = {{quiet_nan(), quiet_nan()}};

    auto const bilinear_xy = [this, gx, gy, bounds]() -> PixelXY {
        long long ix0 = 0;
        long long iy0 = 0;
        double tx = 0.0;
        double ty = 0.0;

        if (metadata_.grid_width > 1) {
            double gx_work = gx;
            if (bounds != BoundsMode::kExtrapolate) {
                gx_work = std::clamp(
                    gx_work,
                    0.0,
                    static_cast<double>(metadata_.grid_width - 1)
                );
            }
            ix0 = static_cast<long long>(std::floor(gx_work));
            ix0 = std::clamp<long long>(
                ix0,
                0,
                static_cast<long long>(metadata_.grid_width) - 2
            );
            tx = gx_work - static_cast<double>(ix0);
        }

        if (metadata_.grid_height > 1) {
            double gy_work = gy;
            if (bounds != BoundsMode::kExtrapolate) {
                gy_work = std::clamp(
                    gy_work,
                    0.0,
                    static_cast<double>(metadata_.grid_height - 1)
                );
            }
            iy0 = static_cast<long long>(std::floor(gy_work));
            iy0 = std::clamp<long long>(
                iy0,
                0,
                static_cast<long long>(metadata_.grid_height) - 2
            );
            ty = gy_work - static_cast<double>(iy0);
        }

        std::size_t const x0 = static_cast<std::size_t>(ix0);
        std::size_t const x1 =
            std::min<std::size_t>(x0 + 1, metadata_.grid_width - 1);
        std::size_t const y0 = static_cast<std::size_t>(iy0);
        std::size_t const y1 =
            std::min<std::size_t>(y0 + 1, metadata_.grid_height - 1);

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
    };

    if (interpolation == InterpolationMode::kNearest) {
        long long const ix = std::llround(gx);
        long long const iy = std::llround(gy);
        std::size_t const sample_ix = static_cast<std::size_t>(
            std::clamp<long long>(ix, 0, static_cast<long long>(metadata_.grid_width) - 1)
        );
        std::size_t const sample_iy = static_cast<std::size_t>(
            std::clamp<long long>(
                iy,
                0,
                static_cast<long long>(metadata_.grid_height) - 1
            )
        );
        xy = sample_node(sample_ix, sample_iy);
    } else if (interpolation == InterpolationMode::kBilinear) {
        xy = bilinear_xy();
    } else if (interpolation == InterpolationMode::kBicubic) {
        if (metadata_.grid_width < 4 or metadata_.grid_height < 4) {
            xy = bilinear_xy();
        } else {
            double const gx_work = bounds == BoundsMode::kExtrapolate
                ? gx
                : std::clamp(gx, 0.0, static_cast<double>(metadata_.grid_width - 1));
            double const gy_work = bounds == BoundsMode::kExtrapolate
                ? gy
                : std::clamp(gy, 0.0, static_cast<double>(metadata_.grid_height - 1));

            long long const anchor_x = static_cast<long long>(std::floor(gx_work));
            long long const anchor_y = static_cast<long long>(std::floor(gy_work));
            bool const has_full_support =
                anchor_x >= 1 and
                anchor_x <= static_cast<long long>(metadata_.grid_width) - 3 and
                anchor_y >= 1 and
                anchor_y <= static_cast<long long>(metadata_.grid_height) - 3;
            if (not has_full_support) {
                xy = bilinear_xy();
            } else {
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
                        PixelXY const node = sample_node(
                            sample_x_idx,
                            sample_y_idx
                        );
                        row = add_scaled(row, node, wx[i]);
                    }
                    accum = add_scaled(accum, row, wy[j]);
                }
                xy = accum;
            }
        }
    } else {
        throw std::runtime_error("Unreachable interpolation mode.");
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
    BoundsMode bounds,
    bool const normalize
) const {
    std::vector<UnprojectLUTQueryResult> results;
    results.reserve(pixels.size());
    for (PixelXY const& pixel : pixels) {
        results.push_back(
            query(pixel.xy[0], pixel.xy[1], interpolation, bounds, normalize)
        );
    }
    return results;
}

}  // namespace lensboy
