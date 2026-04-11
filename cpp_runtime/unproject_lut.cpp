#include "unproject_lut.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lensboy {
namespace {

constexpr char const* k_format_name = "lensboy_unproject_LUT";
constexpr int k_format_version = 1;
constexpr char const* k_header_end_marker = "END_HEADER";
constexpr char const* k_payload_layout = "row_major_interleaved_xy";
constexpr char const* k_payload_endianness = "little";
constexpr std::size_t k_max_header_bytes = 512 * 1024 * 1024;

std::string trim(
    std::string const& text
) {
    std::size_t const first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    std::size_t const last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

double quiet_nan() {
    return std::numeric_limits<double>::quiet_NaN();
}

bool is_finite(
    PixelXY const& value
) {
    return std::isfinite(value.xy[0]) and std::isfinite(value.xy[1]);
}

uint16_t read_little_endian_16(
    char const* data
) {
    return static_cast<uint16_t>(static_cast<unsigned char>(data[0])) |
           (static_cast<uint16_t>(static_cast<unsigned char>(data[1])) << 8);
}

uint32_t read_little_endian_32(
    char const* data
) {
    return static_cast<uint32_t>(static_cast<unsigned char>(data[0])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(data[1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(data[2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(data[3])) << 24);
}

uint64_t read_little_endian_64(
    char const* data
) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(data[i]))
                 << (8 * i);
    }
    return value;
}

double float16_to_double(
    uint16_t bits
) {
    uint16_t const sign = static_cast<uint16_t>((bits >> 15) & 0x1);
    uint16_t const exponent = static_cast<uint16_t>((bits >> 10) & 0x1F);
    uint16_t const mantissa = static_cast<uint16_t>(bits & 0x03FF);

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign == 0 ? 0.0 : -0.0;
        }
        double const fraction = static_cast<double>(mantissa) / 1024.0;
        double const magnitude = std::ldexp(fraction, -14);
        return sign == 0 ? magnitude : -magnitude;
    }

    if (exponent == 31) {
        if (mantissa == 0) {
            return sign == 0 ? std::numeric_limits<double>::infinity()
                             : -std::numeric_limits<double>::infinity();
        }
        return quiet_nan();
    }

    double const fraction = 1.0 + static_cast<double>(mantissa) / 1024.0;
    double const magnitude = std::ldexp(fraction, static_cast<int>(exponent) - 15);
    return sign == 0 ? magnitude : -magnitude;
}

double decode_float32(
    char const* data
) {
    uint32_t const bits = read_little_endian_32(data);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
}

double decode_float64(
    char const* data
) {
    uint64_t const bits = read_little_endian_64(data);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double decode_payload_value(
    char const* data,
    std::string const& storage_encoding
) {
    if (storage_encoding == "float16_xy") {
        return float16_to_double(read_little_endian_16(data));
    }
    if (storage_encoding == "float32_xy") {
        return decode_float32(data);
    }
    if (storage_encoding == "float64_xy") {
        return decode_float64(data);
    }
    throw std::runtime_error("Unsupported storage encoding: " + storage_encoding);
}

std::size_t payload_item_size(
    std::string const& storage_encoding
) {
    if (storage_encoding == "float16_xy") {
        return 2;
    }
    if (storage_encoding == "float32_xy") {
        return 4;
    }
    if (storage_encoding == "float64_xy") {
        return 8;
    }
    throw std::runtime_error("Unsupported storage encoding: " + storage_encoding);
}

std::array<int, 2> parse_pair_of_ints(
    std::string const& text,
    std::string const& field_name
) {
    std::array<int, 2> values{};
    std::stringstream stream(text);
    std::string part;
    for (int i = 0; i < 2; ++i) {
        if (not std::getline(stream, part, ',')) {
            throw std::runtime_error(
                field_name + " must contain exactly 2 comma-separated values."
            );
        }
        values[i] = std::stoi(trim(part));
    }
    if (std::getline(stream, part, ',')) {
        throw std::runtime_error(
            field_name + " must contain exactly 2 comma-separated values."
        );
    }
    return values;
}

std::array<double, 4> parse_quad_of_doubles(
    std::string const& text,
    std::string const& field_name
) {
    std::array<double, 4> values{};
    std::stringstream stream(text);
    std::string part;
    for (int i = 0; i < 4; ++i) {
        if (not std::getline(stream, part, ',')) {
            throw std::runtime_error(
                field_name + " must contain exactly 4 comma-separated values."
            );
        }
        values[i] = std::stod(trim(part));
    }
    if (std::getline(stream, part, ',')) {
        throw std::runtime_error(
            field_name + " must contain exactly 4 comma-separated values."
        );
    }
    return values;
}

std::array<double, 2> parse_pair_of_doubles(
    std::string const& text,
    std::string const& field_name
) {
    std::array<double, 2> values{};
    std::stringstream stream(text);
    std::string part;
    for (int i = 0; i < 2; ++i) {
        if (not std::getline(stream, part, ',')) {
            throw std::runtime_error(
                field_name + " must contain exactly 2 comma-separated values."
            );
        }
        values[i] = std::stod(trim(part));
    }
    if (std::getline(stream, part, ',')) {
        throw std::runtime_error(
            field_name + " must contain exactly 2 comma-separated values."
        );
    }
    return values;
}

std::string parse_optional_string(
    std::string const& text
) {
    if (text == "not_computed") {
        return "";
    }
    return text;
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

UnprojectLUT UnprojectLUT::load(
    std::string_view const path
) {
    std::ifstream file(std::string(path), std::ios::binary);
    if (not file) {
        throw std::runtime_error("Failed to open LUT file: " + std::string(path));
    }

    std::unordered_map<std::string, std::string> header;
    std::string line;
    std::size_t header_bytes = 0;
    while (std::getline(file, line)) {
        header_bytes += line.size() + 1;
        if (header_bytes > k_max_header_bytes) {
            throw std::runtime_error("Header exceeds the maximum supported size.");
        }
        std::string const trimmed = trim(line);
        if (trimmed == k_header_end_marker) {
            break;
        }

        std::size_t const colon = trimmed.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error(
                "Invalid header line. Expected 'key: value'."
            );
        }

        std::string const key = trim(trimmed.substr(0, colon));
        std::string const value = trim(trimmed.substr(colon + 1));
        if (key.empty()) {
            throw std::runtime_error("Header keys must be non-empty.");
        }
        if (header.find(key) != header.end()) {
            throw std::runtime_error("Duplicate header key: " + key);
        }
        header[key] = value;
    }

    if (not file or trim(line) != k_header_end_marker) {
        throw std::runtime_error("Reached end of file before END_HEADER.");
    }

    auto const require_field = [&header](std::string const& key) -> std::string const& {
        auto const it = header.find(key);
        if (it == header.end()) {
            throw std::runtime_error("Missing required header field: " + key);
        }
        return it->second;
    };

    for (auto const& [key, _value] : header) {
        bool const is_removed_field =
            key == "error_report_mode" or
            key == "error_report_max_depth" or
            key == "error_report_min_cell_size" or
            key.rfind("estimated_max_angular_error_", 0) == 0 or
            key.rfind("estimated_median_angular_error_", 0) == 0;
        if (is_removed_field) {
            throw std::runtime_error(
                "This runtime-only .unproject_LUT format does not support legacy "
                "error-report header fields."
            );
        }
    }

    if (require_field("format") != k_format_name) {
        throw std::runtime_error("Unsupported LUT format.");
    }
    int const format_version = std::stoi(require_field("format_version"));
    if (format_version != k_format_version) {
        throw std::runtime_error("Unsupported LUT format_version.");
    }
    if (require_field("payload_layout") != k_payload_layout) {
        throw std::runtime_error("Unsupported payload_layout.");
    }
    if (require_field("payload_endianness") != k_payload_endianness) {
        throw std::runtime_error("Unsupported payload_endianness.");
    }
    std::streampos const payload_offset = file.tellg();
    if (payload_offset < 0) {
        throw std::runtime_error("Failed to determine payload offset.");
    }
    std::size_t const payload_offset_bytes = static_cast<std::size_t>(payload_offset);
    std::size_t const declared_payload_offset_bytes = static_cast<std::size_t>(
        std::stoull(require_field("payload_offset_bytes"))
    );
    if (payload_offset_bytes != declared_payload_offset_bytes) {
        throw std::runtime_error("payload_offset_bytes does not match payload position.");
    }

    std::array<int, 2> const image_size =
        parse_pair_of_ints(require_field("image_size_wh"), "image_size_wh");
    std::array<int, 2> const grid_size =
        parse_pair_of_ints(require_field("grid_size_wh"), "grid_size_wh");
    std::array<double, 4> const extents =
        parse_quad_of_doubles(require_field("grid_extents_xy"), "grid_extents_xy");
    std::array<double, 2> const grid_stride =
        parse_pair_of_doubles(require_field("grid_stride_xy"), "grid_stride_xy");
    double const expected_grid_stride_x =
        grid_size[0] <= 1 ? 0.0 : (extents[1] - extents[0]) / static_cast<double>(grid_size[0] - 1);
    double const expected_grid_stride_y =
        grid_size[1] <= 1 ? 0.0 : (extents[3] - extents[2]) / static_cast<double>(grid_size[1] - 1);
    if (std::abs(grid_stride[0] - expected_grid_stride_x) > 1e-12 or
        std::abs(grid_stride[1] - expected_grid_stride_y) > 1e-12) {
        throw std::runtime_error("grid_stride_xy does not match grid_extents_xy and grid_size_wh.");
    }

    std::string const storage_encoding = require_field("storage_encoding");
    std::size_t const item_size = payload_item_size(storage_encoding);
    std::size_t const expected_payload_bytes =
        static_cast<std::size_t>(grid_size[0]) *
        static_cast<std::size_t>(grid_size[1]) * 2 * item_size;

    std::vector<char> payload(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    if (payload.size() != expected_payload_bytes) {
        throw std::runtime_error("Unexpected payload size.");
    }

    std::vector<double> xy_grid(
        static_cast<std::size_t>(grid_size[0]) *
        static_cast<std::size_t>(grid_size[1]) * 2
    );
    for (std::size_t i = 0; i < xy_grid.size(); ++i) {
        xy_grid[i] =
            decode_payload_value(payload.data() + i * item_size, storage_encoding);
        if (not std::isfinite(xy_grid[i])) {
            throw std::runtime_error("Payload contains non-finite values.");
        }
    }

    std::string const default_interpolation = require_field("default_interpolation");
    std::string const default_bounds = require_field("default_bounds");
    std::string const source_model_type =
        parse_optional_string(require_field("source_model_type"));
    std::string const source_model_spec_json =
        parse_optional_string(require_field("source_model_spec_json"));
    std::string const source_model_spec_json_sha256 =
        parse_optional_string(require_field("source_model_spec_json_sha256"));
    std::string const lensboy_version = require_field("lensboy_version");

    std::vector<char> string_storage;
    string_storage.reserve(
        storage_encoding.size() +
        default_interpolation.size() +
        default_bounds.size() +
        source_model_type.size() +
        source_model_spec_json.size() +
        source_model_spec_json_sha256.size() +
        lensboy_version.size() +
        8
    );

    UnprojectLUTMetadata metadata;
    metadata.image_width = static_cast<std::size_t>(image_size[0]);
    metadata.image_height = static_cast<std::size_t>(image_size[1]);
    metadata.grid_width = static_cast<std::size_t>(grid_size[0]);
    metadata.grid_height = static_cast<std::size_t>(grid_size[1]);
    metadata.grid_x_min = extents[0];
    metadata.grid_x_max = extents[1];
    metadata.grid_y_min = extents[2];
    metadata.grid_y_max = extents[3];
    metadata.grid_stride_x = grid_stride[0];
    metadata.grid_stride_y = grid_stride[1];
    metadata.storage_encoding = append_string_view(string_storage, storage_encoding);
    metadata.default_interpolation = append_string_view(
        string_storage,
        default_interpolation
    );
    metadata.default_bounds = append_string_view(
        string_storage,
        default_bounds
    );
    metadata.source_model_type = append_string_view(
        string_storage,
        source_model_type
    );
    metadata.source_model_spec_json = append_string_view(
        string_storage,
        source_model_spec_json
    );
    metadata.source_model_spec_json_sha256 = append_string_view(
        string_storage,
        source_model_spec_json_sha256
    );
    metadata.lensboy_version = append_string_view(
        string_storage,
        lensboy_version
    );
    metadata.payload_offset_bytes = payload_offset_bytes;

    return UnprojectLUT(
        std::move(metadata),
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
