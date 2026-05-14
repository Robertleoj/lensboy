# Unproject LUT Guide

Unprojection - turning a pixel back into a 3D camera-frame ray - is the
inverse of `project_points()`. For most camera models this has no closed
form and has to be solved iteratively, which is too slow for tight
runtime loops.

`UnprojectLUT` precomputes `normalize_points()` on a regular pixel grid so
that queries become a single interpolation. This page covers building a
LUT, querying it from Python or C++, and checking how accurate the cached
rays are.

## Build a runtime LUT

```python
import lensboy as lb

model = lb.OpenCV.load("camera.json")
lut = model.get_unproject_lut(pixel_stride=32)
lut.save("camera_lut/")
```

The grid always spans the full image, from `(0, 0)` to
`(image_width - 1, image_height - 1)`. The seeded Newton solver runs in
C++ and is fast: a full per-pixel grid on a ~6 MP image builds in well
under a second. Queries afterwards are constant-time per pixel.

Two mutually exclusive sizing knobs:

- `pixel_stride=...` - approximate sample spacing in pixels. Accepts a
  single float or a `(stride_x, stride_y)` tuple.
- `grid_size_wh=(W, H)` - cached grid size directly, in samples.

If neither is given, the LUT samples once per image pixel. That's
usually more than you need - use the error heatmap below to find the
loosest stride that still meets your accuracy target.

### Choosing a stride

Smaller stride → denser grid → more accurate rays, but more memory. The
right value depends on how much angular error your downstream code
tolerates. The workflow is:

1. Pick a stride (start at 32 or 64).
2. Build the LUT.
3. Run `compute_lut_error_heatmap` (below) to measure the worst-case
   angular error your interpolation produces.
4. Halve the stride if you need more accuracy; double it if you have
   headroom.

## Load and query the LUT

```python
runtime_lut = lb.UnprojectLUT.load("camera_lut/")

rays, valid_mask = runtime_lut.normalize_points(pixel_coords)
```

`rays` has shape `(N, 3)`. Pixels outside the image get NaN rays and
`False` in `valid_mask`.

Interpolation modes (default is `bicubic`):

- `nearest` - 1 sample, no interpolation.
- `bilinear` - 4 samples per query.
- `bicubic` (Catmull-Rom) - 16 samples per query. Falls back to bilinear
  in a one-cell border where the 4×4 neighbourhood doesn't fit, and on
  grids smaller than 4×4.

Bicubic is the default because the accuracy gap is huge for a modest
cost. On the bundled OpenCV test calibration (3088×2064, stride 32,
98×66 grid), the median over cells of the per-cell maximum angular
error and the vectorised Python per-pixel query time (best of 5 over a
full-image batch) come out as:

| mode     | median max-cell error | error vs bicubic | query time | time vs bicubic |
|----------|----------------------:|-----------------:|-----------:|----------------:|
| nearest  | 911.6 mdeg            | ~10 000×         |  98 ns/pix | 0.10×           |
| bilinear | 8.09 mdeg             | ~100×            | 212 ns/pix | 0.21×           |
| bicubic  | 0.084 mdeg            | 1×               | 997 ns/pix | 1×              |

So bilinear is ~5× cheaper than bicubic but ~100× less accurate at the
same stride; nearest is cheaper still and another ~100× worse on top.
Median is used rather than mean because bicubic falls back to bilinear
in a one-cell border, which biases mean-error comparisons. Your numbers
will vary with the camera model and the stride - the heatmap is the
authoritative answer for your case.

> **Ray convention differs between Python and C++.**
> Python's `normalize_points()` returns rays of the form `[x, y, 1]` (not
> unit length) - only the cached `x` and `y` are stored; the third
> component is reconstructed on query. The C++ `query()` defaults to
> *unit-length* rays and only returns `[x, y, 1]` if you opt out of
> normalisation. See the C++ section below.

## Analyse accuracy

```python
from lensboy.analysis import compute_lut_error_heatmap

heatmap = compute_lut_error_heatmap(runtime_lut, model)  # defaults to bicubic
heatmap.save("camera_bicubic_error_heatmap.npz")
```

`compute_lut_error_heatmap` runs a per-cell gradient-ascent maximiser in
C++ that finds, for each LUT cell, the in-cell pixel where the
interpolated ray deviates most (angularly) from the exact camera model.

The returned `UnprojectLUTErrorHeatmap` stores, per cell:

- `peak_pixel_xy` - the pixel where the worst angular error sits.
- `exact_xy` - the camera-model normalised `xy` at that pixel.
- `approx_xy` - the LUT-interpolated normalised `xy` at the same pixel.

Derived quantities are exposed as properties:

- `max_angular_error_deg` - per-cell peak angular error, shape `(H, W)`.
- `error_delta_xy` - `approx_xy − exact_xy`.
- `error_direction_xy` - unit-length residual direction.

To plot:

```python
heatmap.plot(angular_unit="mdeg")
```

See the `plot()` docstring for the full set of styling knobs (colorbar
units and clip, arrow grid density, arrow scaling, colormap, figure
size).

## File format

`UnprojectLUT.save(dir_path)` writes a directory (creating it if needed)
containing two files:

- `metadata.json` - scalar parameters (image size, grid extents, lensboy
  version).
- `xy_grid.npy` - the cached `(grid_height, grid_width, 2)` array of
  normalised `x/y` ray components, stored as little-endian `float32`. The
  third ray component is implicit, so it isn't stored.

`metadata.json` looks like this:

```json
{
    "lensboy-version": "3.x.y",
    "image_width": 3088,
    "image_height": 2064,
    "grid_x_min": 0.0,
    "grid_x_max": 3087.0,
    "grid_y_min": 0.0,
    "grid_y_max": 2063.0
}
```

Grid sample count comes from the `xy_grid.npy` shape, not from
`metadata.json`. `load()` rejects LUTs whose `lensboy-version` has a major
version below 3.

The LUT is a pure runtime artifact - it does not store the source camera
model. Keep the model around (or rebuild it from your calibration) if you
later want to run `compute_lut_error_heatmap`.

## Standalone C++ runtime

The repository ships a small standalone runtime in `cpp_runtime/`:

- `unproject_lut.hpp` / `unproject_lut.cpp` - the LUT itself.
- `json.hpp` - vendored [nlohmann/json](https://github.com/nlohmann/json),
  used to parse `metadata.json`.
- `npy.hpp` - vendored [llohse/libnpy](https://github.com/llohse/libnpy),
  used to read `xy_grid.npy`.

Copy the whole directory into another project and load/query the LUT
there:

```cpp
#include "unproject_lut.hpp"

auto lut = lensboy::UnprojectLUT::load("camera_lut/");
auto result = lut.query(1280.0, 720.0);  // bicubic, unit-length ray

if (result.valid) {
    do_stuff(result.ray);
}
```

`query()` returns a **unit-length** ray by default - this differs from
Python, which returns `[x, y, 1]`. To get the Python convention, pass
`false` as the fourth argument (the `normalize` parameter):

```cpp
auto result = lut.query(1280.0, 720.0, lensboy::InterpolationMode::BICUBIC, false);
```
