# Unproject LUT Guide

`lensboy` can export a camera model's `normalize_points()` field into a regular-grid lookup table for fast runtime queries.

The `.unproject_LUT` file format is intentionally split into two concerns:

- `UnprojectLUT` is the runtime object. It handles grid construction, file I/O, and fast lookup.
- The analysis functions (`estimate_lut_accuracy`, `compute_lut_error_heatmap`) take the LUT and the original camera model to compute accuracy summaries or error heatmaps.

This keeps the runtime file format compact while still making later analysis possible.

## Build a runtime LUT

```python
import lensboy as lb

model = lb.OpenCV.load("camera.json")
lut = model.get_unproject_lut(
    pixel_stride=32,
    storage_encoding="float32_xy",
)
lut.save("camera.unproject_LUT")
```

You can control the LUT size in two ways:

- `pixel_stride=...` chooses an approximate spacing in image pixels between cached samples.
- `storage_encoding=...` chooses the on-disk precision:
  - `float64_xy`
  - `float32_xy`
  - `float16_xy`

The LUT stores only `x` and `y`. The queried ray is always reconstructed as `[x, y, 1]`.

## Load and query the LUT

```python
runtime_lut = lb.UnprojectLUT.load("camera.unproject_LUT")

rays = runtime_lut.normalize_points(
    pixel_coords,
    interpolation="bilinear",
    bounds="strict",
)
```

Supported runtime interpolation modes:

- `nearest`
- `bilinear`
- `bicubic`

Supported bounds modes:

- `strict`
- `clamp`
- `extrapolate`

`bilinear` is the default. `strict` is the default bounds behavior.

## Analyze accuracy later

```python
from lensboy.analysis import estimate_lut_accuracy, compute_lut_error_heatmap

report = estimate_lut_accuracy(
    runtime_lut, model,
    interpolations=("nearest", "bilinear"),
)
heatmap = compute_lut_error_heatmap(runtime_lut, model, interpolation="bilinear")
heatmap.save("camera_bilinear_error_heatmap.npz")
```

The accuracy report contains:

- `max_angular_error_mdeg`
- `median_angular_error_mdeg`
- the interpolation modes that were analyzed
- the adaptive estimator settings that produced the result

## Plot a heatmap

```python
from lensboy.analysis import plot_unproject_lut_error_heatmap

fig = plot_unproject_lut_error_heatmap(
    heatmap,
    angular_unit="mdeg",
    return_figure=True,
)
```

The plot helper accepts either:

- an in-memory `UnprojectLUTErrorHeatmap`
- a saved `.npz` heatmap path

## File format

The `.unproject_LUT` file starts with an ASCII header followed by a binary payload. The payload offset is written into the header itself, so the header length can grow as needed.

The header looks like this:

```text
format: lensboy_unproject_LUT
payload_offset_bytes: 382
format_version: 1
lensboy_version: 3.0.1
image_size_wh: 3088, 2064
grid_size_wh: 98, 66
grid_extents_xy: 0, 3087, 0, 2063
grid_stride_xy: 31.824742268041238, 31.738461538461539
storage_encoding: float32_xy
default_interpolation: bilinear
default_bounds: strict
payload_layout: row_major_interleaved_xy
payload_endianness: little
END_HEADER
```

Notes:

- `grid_stride_xy` is derived from the extents and grid size, so it may be fractional.
- The binary payload is little-endian, row-major, interleaved `x/y`.
- The LUT is a pure runtime artifact -- it does not store the source camera model.

## Standalone C++ runtime

The repository ships a small standalone runtime in:

- `cpp_runtime/unproject_lut.hpp`
- `cpp_runtime/unproject_lut.cpp`

You can copy those two files into another project and load/query the LUT there:

```cpp
#include "unproject_lut.hpp"

auto lut = lensboy::UnprojectLUT::load("camera.unproject_LUT");
auto result = lut.query(
    1280.0,
    720.0,
    lensboy::InterpolationMode::kBilinear,
    lensboy::BoundsMode::kStrict
);

if (result.valid) {
    do_stuff(result.ray);
}
```

By default, `query()` returns a unit-length ray. Pass `normalize=false` if you want the stored LUT convention of `[x, y, 1]`.
