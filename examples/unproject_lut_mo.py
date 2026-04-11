#!/usr/bin/env -S uv run --with marimo marimo -y edit --no-sandbox
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "lensboy>=3.0.1",
#     "marimo>=0.23.0",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import numpy as np

    import lensboy as lb
    from lensboy.analysis import (
        UnprojectLUTAnalyzer,
        UnprojectLUTErrorHeatmap,
        plot_unproject_lut_error_heatmap,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Unproject LUT (LookUp Table) example

    The function which goes from pixel location to 3D ray is called `unproject()`. It is usually not possible to compute in closed form, so it must be iteratively approximated. Unfortunately, this process is slow, which can be a problem for some applications.

    Lensboy allows you to precompute these values once ahead of time, for fast lookup during operation. The runtime LUT stays compact and fast, while the accuracy analysis is available separately on demand.

    This notebook builds a `.unproject_LUT` file from the bundled OpenCV test
    calibration, saves it to `examples/generated/`, reloads it, and compares the
    cached rays against the exact camera model.

    The controls intentionally stay in a moderate range so the notebook remains
    interactive. If you want a denser cache, edit the values in the cells below.
    """)
    return


@app.cell
def _():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "test_datasets"
    output_dir = repo_root / "examples" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir, repo_root


@app.cell(hide_code=True)
def _(model):
    mo.md(f"""
    ## Source camera model

    Using a OpenCV calibration that comes with Lensboy for example purposes:

    - `{model = !r}`
    """)
    return


@app.cell
def _(data_dir):
    model = lb.OpenCV.load(data_dir / "opencv.json")
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compute the lookup table

    Use the drop-down widget below to provide paramters for how the LUT is created and used.
    """)
    return


@app.cell(hide_code=True)
def _():
    pixel_stride_widget = mo.ui.dropdown(
        options=[
            "1",
            "2",
            "3",
            "4",
            "6",
            "8",
            "12",
            "16",
            "24",
            "32",
            "48",
            "64",
            "96",
            "128",
        ],
        value="64",
        label="Approximate pixel stride",
    )
    storage_encoding_widget = mo.ui.dropdown(
        options=["float64_xy", "float32_xy", "float16_xy"],
        value="float32_xy",
        label="Storage encoding",
    )
    num_workers_widget = mo.ui.slider(
        start=1,
        step=1,
        stop=64,
        value=8,
        label="Number of workers",
        show_value=True,
        debounce=True,
    )
    controls_ui = mo.vstack(
        [
            pixel_stride_widget,
            storage_encoding_widget,
            num_workers_widget,
        ]
    )
    controls_ui
    return num_workers_widget, pixel_stride_widget, storage_encoding_widget


@app.cell(hide_code=True)
def _(num_workers_widget, pixel_stride_widget, storage_encoding_widget):
    controls = SimpleNamespace(
        pixel_stride=float(pixel_stride_widget.value),
        storage_encoding=storage_encoding_widget.value,
        num_workers=num_workers_widget.value,
    )
    return (controls,)


@app.cell
def _(controls, model):
    # Be patient! This may take a while.
    lut = model.get_unproject_lut(
        pixel_stride=controls.pixel_stride,
        storage_encoding=controls.storage_encoding,
        num_workers=controls.num_workers,
    )
    return (lut,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save the lookup-table to disk
    """)
    return


@app.cell
def _(controls, lut, output_dir):
    lut_filename = f"opencv_stride_{int(controls.pixel_stride)}_{controls.storage_encoding}.unproject_LUT"
    lut_path = output_dir / lut_filename

    lut.save(lut_path)
    return (lut_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load the lookup-table from disk
    """)
    return


@app.cell
def _(lut_path):
    loaded = lb.UnprojectLUT.load(lut_path)
    return (loaded,)


@app.cell(hide_code=True)
def _(loaded, lut_path, repo_root):
    mo.md(rf"""
    ## Loaded file header data

    Read `{lut_path.relative_to(repo_root)}`.

    - total size: `{loaded.total_bytes / 1024:.1f} KiB`
    - payload size: `{loaded.payload_bytes / 1024:.1f} KiB`
    - grid size: `{loaded.grid_width} x {loaded.grid_height}`
    - full file header:
    ```text
    {loaded.header_preview()}
    ```
    """)
    return


@app.cell
def _(loaded):
    analyzer = UnprojectLUTAnalyzer(loaded)
    return (analyzer,)


@app.cell(hide_code=True)
def _():
    interpolation_widget = mo.ui.dropdown(
        options=["nearest", "bilinear", "bicubic"],
        value="bilinear",
        label="Query interpolation mode",
    )
    analyzer_ui = mo.vstack(
        [
            interpolation_widget,
        ]
    )
    analyzer_ui
    return (interpolation_widget,)


@app.cell(hide_code=True)
def _(interpolation_widget):
    interpolation_mode = interpolation_widget.value
    return (interpolation_mode,)


@app.cell(hide_code=True)
def _(accuracy_report, interpolation_mode, sample_accuracy):
    mo.md(f"""
    ## Accuracy on a dense sample

    Queried `{sample_accuracy.sample_count}` evenly spaced pixels with
    `{interpolation_mode}` interpolation.

    - sample grid: `{sample_accuracy.sample_grid_width} x {sample_accuracy.sample_grid_height}`
    - max observed angular error on this sample: `{sample_accuracy.max_angular_error_mdeg:.3f} milli degrees`
    - mean angular error on this sample: `{sample_accuracy.mean_angular_error_mdeg:.3f} milli degrees`
    - analyzer-estimated max for `{interpolation_mode}`: `{accuracy_report.max_angular_error_mdeg[interpolation_mode]:.3f} milli degrees`
    - analyzer-estimated median for `{interpolation_mode}`: `{accuracy_report.median_angular_error_mdeg[interpolation_mode]:.3f} milli degrees`
    """)
    return


@app.cell
def _(analyzer, interpolation_mode):
    accuracy_report = analyzer.estimate_accuracy(interpolations=interpolation_mode)
    sample_accuracy = analyzer.sample_accuracy_grid(
        interpolation=interpolation_mode,
        target_sample_count=2500,
    )
    return accuracy_report, sample_accuracy


@app.cell(hide_code=True)
def _(heatmap_path, interpolation_mode):
    mo.md(f"""
    ## Error heatmap

    Computed the heatmap on demand from the loaded LUT, exported `{heatmap_path.name}`,
    reloaded it from disk, and plotted the per-cell maximum angular error for
    `{interpolation_mode}` interpolation in `milli degrees`.

    The cyan arrows show the direction of the local peak `x/y`
    interpolation error within each sampled cell.
    """)
    return


@app.cell
def _(analyzer, controls, interpolation_mode, output_dir):
    heatmap_filename = f"opencv_stride_{int(controls.pixel_stride)}_{controls.storage_encoding}_{interpolation_mode}_error_heatmap.npz"
    heatmap_path = output_dir / heatmap_filename

    heatmap = analyzer.compute_error_heatmap(interpolation=interpolation_mode)
    heatmap.save(heatmap_path)
    loaded_heatmap = UnprojectLUTErrorHeatmap.load(heatmap_path)
    fig = plot_unproject_lut_error_heatmap(
        loaded_heatmap,
        angular_unit="mdeg",
        figsize=(7.8, 5.3),
        return_figure=True,
    )
    mo.mpl.interactive(fig)
    return (heatmap_path,)


@app.cell(hide_code=True)
def _(bounds_demo_pixels, clamp_rays, strict_rays, valid_mask):
    mo.md(f"""
    ## Bounds behavior

    `strict` keeps queries safe by flagging pixels outside the LUT domain.

    - demo pixels:
      `{np.array2string(bounds_demo_pixels, precision=2, separator=", ")}`
    - strict valid mask:
      `{np.array2string(valid_mask, separator=", ")}`
    - first strict ray:
      `{np.array2string(strict_rays[0], precision=5, separator=", ")}`
    - first clamped ray:
      `{np.array2string(clamp_rays[0], precision=5, separator=", ")}`
    """)
    return


@app.cell
def _(loaded, model):
    bounds_demo_pixels = np.array(
        [
            [-20.0, 50.0],
            [0.0, 0.0],
            [model.image_width - 1, model.image_height - 1],
            [model.image_width + 20.0, model.image_height / 2.0],
        ]
    )
    strict_rays, valid_mask = loaded.normalize_points(
        bounds_demo_pixels,
        bounds="strict",
        return_valid_mask=True,
    )
    clamp_rays = loaded.normalize_points(bounds_demo_pixels, bounds="clamp")
    return bounds_demo_pixels, clamp_rays, strict_rays, valid_mask


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## C++ runtime

    The same file can be loaded from the standalone runtime in `cpp_runtime/`. Simply copy-paste those files into your project and use them like so:

    ```cpp
    #include "unproject_lut.hpp"

    auto lut = lensboy::UnprojectLUT::load("camera_runtime.unproject_LUT");
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
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
