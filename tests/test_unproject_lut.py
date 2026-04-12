"""Tests for UnprojectLUT creation, serialization, querying, and C++ loading."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

import lensboy as lb

DATA_DIR = Path(__file__).parent.parent / "data/test_datasets"
REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_RUNTIME_DIR = REPO_ROOT / "cpp_runtime"


def _make_linear_pinhole_model() -> lb.OpenCV:
    """Zero-distortion OpenCV model that behaves like a pure pinhole."""
    return lb.OpenCV(
        image_width=17,
        image_height=13,
        fx=23.0,
        fy=19.0,
        cx=8.0,
        cy=6.0,
        distortion_coeffs=np.zeros(14, dtype=np.float64),
    )


def _load_opencv_model() -> lb.OpenCV:
    return lb.OpenCV.load(DATA_DIR / "opencv.json")


def _load_spline_model() -> lb.PinholeSplined:
    return lb.PinholeSplined.load(DATA_DIR / "spline.json")


def _random_pixels(
    model: lb.CameraModel,
    n: int = 128,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, model.image_width - 1, n)
    ys = rng.uniform(0.0, model.image_height - 1, n)
    return np.column_stack([xs, ys])


def _query_error_deg(
    reference: np.ndarray,
    approx: np.ndarray,
) -> float:
    reference_unit = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    approx_unit = approx / np.linalg.norm(approx, axis=1, keepdims=True)
    dots = np.einsum("ij,ij->i", reference_unit, approx_unit)
    return float(np.max(np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))))


def test_unproject_lut_save_load_writes_expected_files(tmp_path: Path) -> None:
    """save() writes a directory with metadata.json and xy_grid.npy."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(5, 4))

    dir_path = tmp_path / "lut"
    lut.save(dir_path)

    assert (dir_path / "metadata.json").is_file()
    assert (dir_path / "xy_grid.npy").is_file()

    import json as _json

    metadata = _json.loads((dir_path / "metadata.json").read_text())
    assert metadata["lensboy-version"].split(".")[0] >= "3"
    assert metadata["image_width"] == 17
    assert metadata["image_height"] == 13

    xy_grid = np.load(dir_path / "xy_grid.npy")
    assert xy_grid.shape == (4, 5, 2)
    assert xy_grid.dtype == np.float32


@pytest.mark.parametrize(
    "factory",
    [_load_opencv_model, _load_spline_model],
)
def test_unproject_lut_round_trip_for_camera_models(
    tmp_path: Path,
    factory,
) -> None:
    """OpenCV and spline models survive LUT save/load round trips."""
    model = factory()
    lut = model.get_unproject_lut(grid_size_wh=(11, 9))

    dir_path = tmp_path / "round_trip_lut"
    lut.save(dir_path)
    loaded = lb.UnprojectLUT.load(dir_path)

    assert loaded.grid_size_wh == (11, 9)

    x_coords = np.linspace(0.0, model.image_width - 1, loaded.grid_width)
    y_coords = np.linspace(0.0, model.image_height - 1, loaded.grid_height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    expected = model.normalize_points(pixels)
    actual = loaded.normalize_points(pixels, interpolation="bilinear")
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_linear_pinhole_lut_round_trips_through_float32(tmp_path: Path) -> None:
    """Linear pinhole data round-trips through the float32 LUT format."""
    model = _make_linear_pinhole_model()
    sample_pixels = _random_pixels(model, seed=4)
    expected = model.normalize_points(sample_pixels)

    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    dir_path = tmp_path / "lut"
    lut.save(dir_path)
    loaded = lb.UnprojectLUT.load(dir_path)

    approx = loaded.normalize_points(sample_pixels, interpolation="bilinear")
    assert isinstance(approx, np.ndarray)
    max_abs_error = float(np.max(np.abs(approx[:, :2] - expected[:, :2])))
    assert max_abs_error < 1e-6


def test_unproject_lut_analyzer_can_report_multiple_interpolations() -> None:
    """The analyzer can report multiple interpolation modes at once."""
    from lensboy.analysis import estimate_lut_accuracy

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    report = estimate_lut_accuracy(
        lut, model, interpolations=("nearest", "bilinear", "bicubic")
    )

    assert report.interpolations == ("nearest", "bilinear", "bicubic")
    assert np.isfinite(report.max_angular_error_mdeg["nearest"])
    assert np.isfinite(report.max_angular_error_mdeg["bilinear"])
    assert np.isfinite(report.max_angular_error_mdeg["bicubic"])
    assert np.isfinite(report.median_angular_error_mdeg["nearest"])
    assert np.isfinite(report.median_angular_error_mdeg["bilinear"])
    assert np.isfinite(report.median_angular_error_mdeg["bicubic"])
    assert (
        report.max_angular_error_mdeg["bilinear"]
        < report.max_angular_error_mdeg["nearest"]
    )


def test_unproject_lut_analyzer_accepts_single_interpolation() -> None:
    """A single interpolation mode can be passed directly to the analyzer."""
    from lensboy.analysis import estimate_lut_accuracy

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    report = estimate_lut_accuracy(lut, model, interpolations="bicubic")

    assert report.interpolations == ("bicubic",)
    assert set(report.max_angular_error_mdeg) == {"bicubic"}
    assert set(report.median_angular_error_mdeg) == {"bicubic"}
    assert np.isfinite(report.max_angular_error_mdeg["bicubic"])
    assert np.isfinite(report.median_angular_error_mdeg["bicubic"])


def test_unproject_lut_analyzer_matches_loaded_and_in_memory_lut(tmp_path: Path) -> None:
    """Loaded LUTs produce the same analyzer report as in-memory LUTs."""
    from lensboy.analysis import estimate_lut_accuracy

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    dir_path = tmp_path / "opencv_lut"
    lut.save(dir_path)
    loaded = lb.UnprojectLUT.load(dir_path)

    report_before = estimate_lut_accuracy(
        lut, model, interpolations=("nearest", "bilinear", "bicubic")
    )
    report_after = estimate_lut_accuracy(
        loaded, model, interpolations=("nearest", "bilinear", "bicubic")
    )

    assert report_after.interpolations == report_before.interpolations
    for mode in report_before.interpolations:
        assert report_after.max_angular_error_mdeg[mode] == pytest.approx(
            report_before.max_angular_error_mdeg[mode], rel=1e-3
        )
        assert report_after.median_angular_error_mdeg[mode] == pytest.approx(
            report_before.median_angular_error_mdeg[mode], rel=1e-3
        )


def test_unproject_lut_analyzer_can_sample_dense_accuracy_grid() -> None:
    """The analyzer can compare LUT rays against exact rays on a dense sample grid."""
    from lensboy.analysis import sample_lut_accuracy

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    sample = sample_lut_accuracy(
        lut, model, interpolation="bilinear", target_sample_count=2500
    )

    expected_sample_count = sample.sample_grid_width * sample.sample_grid_height
    assert sample.interpolation == "bilinear"
    assert sample.sample_count == expected_sample_count
    assert sample.sample_pixels.shape == (expected_sample_count, 2)
    assert sample.exact_rays.shape == (expected_sample_count, 3)
    assert sample.approx_rays.shape == (expected_sample_count, 3)
    assert sample.angular_error_deg.shape == (expected_sample_count,)
    assert np.isfinite(sample.angular_error_deg).all()
    assert sample.max_angular_error_mdeg == pytest.approx(
        float(np.max(sample.angular_error_deg) * 1.0e3)
    )
    assert sample.mean_angular_error_mdeg == pytest.approx(
        float(np.mean(sample.angular_error_deg) * 1.0e3)
    )
    assert sample.median_angular_error_mdeg == pytest.approx(
        float(np.median(sample.angular_error_deg) * 1.0e3)
    )


def test_unproject_lut_analyzer_dense_accuracy_grid_is_exact_for_linear_model() -> None:
    """A linear pinhole LUT matches the exact source model on dense sampled queries."""
    from lensboy.analysis import sample_lut_accuracy

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    sample = sample_lut_accuracy(
        lut, model, interpolation="bilinear", target_sample_count=2500
    )

    np.testing.assert_allclose(sample.approx_rays, sample.exact_rays, atol=1e-5)
    assert sample.max_angular_error_mdeg < 1e-2


def test_unproject_lut_grid_stride_can_be_fractional() -> None:
    """The stored sample spacing reflects the actual grid spacing."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))

    stride_x, stride_y = lut.grid_stride_xy
    assert stride_x == pytest.approx(16.0 / 5.0)
    assert stride_y == pytest.approx(12.0 / 4.0)


def test_unproject_lut_bounds_modes_match_expected_behavior() -> None:
    """Strict, clamp, and extrapolate modes behave safely and predictably."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(5, 4))
    pixels = np.array(
        [
            [-1.5, -0.5],
            [0.0, 0.0],
            [model.image_width - 1, model.image_height - 1],
            [model.image_width + 1.0, 3.5],
        ]
    )

    with pytest.raises(ValueError, match="outside the LUT domain"):
        lut.normalize_points(pixels, bounds="strict")

    strict_rays, valid_mask = lut.normalize_points(
        pixels,
        bounds="strict",
        return_valid_mask=True,
    )
    np.testing.assert_array_equal(valid_mask, np.array([False, True, True, False]))
    assert np.isnan(strict_rays[0, 0]) and np.isnan(strict_rays[0, 1])
    assert np.isnan(strict_rays[3, 0]) and np.isnan(strict_rays[3, 1])

    clamped_pixels = np.column_stack(
        [
            np.clip(pixels[:, 0], 0.0, model.image_width - 1),
            np.clip(pixels[:, 1], 0.0, model.image_height - 1),
        ]
    )
    clamp_expected = model.normalize_points(clamped_pixels)
    clamp_actual = lut.normalize_points(pixels, interpolation="bilinear", bounds="clamp")
    np.testing.assert_allclose(clamp_actual, clamp_expected, atol=1e-5)

    extrap_expected = model.normalize_points(pixels)
    extrap_actual = lut.normalize_points(
        pixels,
        interpolation="bilinear",
        bounds="extrapolate",
    )
    np.testing.assert_allclose(extrap_actual, extrap_expected, atol=1e-5)


def test_unproject_lut_bicubic_falls_back_to_bilinear_without_full_stencil() -> None:
    """Bicubic uses bilinear when a full 4x4 support region is unavailable."""
    model = _load_opencv_model()

    small_lut = model.get_unproject_lut(grid_size_wh=(3, 3))
    sample_pixels = np.array(
        [
            [10.0, 20.0],
            [512.25, 256.75],
            [model.image_width - 10.0, model.image_height - 20.0],
        ]
    )
    np.testing.assert_allclose(
        small_lut.normalize_points(sample_pixels, interpolation="bicubic"),
        small_lut.normalize_points(sample_pixels, interpolation="bilinear"),
        atol=1e-5,
    )

    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    boundary_pixels = np.array(
        [
            [0.0, 0.0],
            [0.0, model.image_height * 0.5],
            [model.image_width - 1.0, model.image_height * 0.5],
            [model.image_width * 0.5, 0.0],
            [model.image_width * 0.5, model.image_height - 1.0],
            [-10.0, model.image_height * 0.25],
        ]
    )
    np.testing.assert_allclose(
        lut.normalize_points(boundary_pixels, interpolation="bicubic", bounds="clamp"),
        lut.normalize_points(boundary_pixels, interpolation="bilinear", bounds="clamp"),
        atol=1e-5,
    )


def test_unproject_lut_analyzer_report_is_finite_for_nonlinear_model() -> None:
    """A nonlinear model produces finite observed angular error summaries."""
    from lensboy.analysis import estimate_lut_accuracy

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    report = estimate_lut_accuracy(lut, model)

    assert report.interpolations == ("bilinear",)
    assert np.isfinite(report.max_angular_error_mdeg["bilinear"])
    assert np.isfinite(report.median_angular_error_mdeg["bilinear"])
    assert (
        report.median_angular_error_mdeg["bilinear"]
        <= report.max_angular_error_mdeg["bilinear"]
    )

    xs = np.linspace(0.0, model.image_width - 1, 35)
    ys = np.linspace(0.0, model.image_height - 1, 27)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    exact = model.normalize_points(pixels)
    approx = lut.normalize_points(pixels, interpolation="bilinear")
    assert isinstance(approx, np.ndarray)
    dense_error = _query_error_deg(exact, approx)
    assert dense_error <= (report.max_angular_error_mdeg["bilinear"] / 1.0e3) + 1.0


def test_unproject_lut_rejects_unsupported_lensboy_version(tmp_path: Path) -> None:
    """Loading validates the metadata lensboy-version."""
    import json as _json

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    dir_path = tmp_path / "bad_version_lut"
    lut.save(dir_path)

    metadata_path = dir_path / "metadata.json"
    metadata = _json.loads(metadata_path.read_text())
    metadata["lensboy-version"] = "2.9.0"
    metadata_path.write_text(_json.dumps(metadata))

    with pytest.raises(ValueError, match="incompatible version"):
        lb.UnprojectLUT.load(dir_path)


def test_unproject_lut_rejects_nonfinite_grid_values(tmp_path: Path) -> None:
    """The dataclass rejects non-finite xy_grid values on load."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    dir_path = tmp_path / "nonfinite_lut"
    lut.save(dir_path)

    grid_path = dir_path / "xy_grid.npy"
    xy_grid = np.load(grid_path)
    xy_grid[0, 0, 0] = np.nan
    np.save(grid_path, xy_grid, allow_pickle=False)

    with pytest.raises(ValueError, match="finite"):
        lb.UnprojectLUT.load(dir_path)


def test_unproject_lut_analyzer_can_save_and_load_error_heatmaps(tmp_path: Path) -> None:
    """Analyzer heatmaps can be saved and loaded without the LUT header."""
    from lensboy.analysis import UnprojectLUTErrorHeatmap, compute_lut_error_heatmap

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap_path = tmp_path / "bilinear_error_heatmaps.npz"

    heatmap = compute_lut_error_heatmap(lut, model, interpolation="bilinear")
    heatmap.save(heatmap_path)
    loaded = UnprojectLUTErrorHeatmap.load(heatmap_path)

    assert loaded.interpolation == "bilinear"
    assert loaded.max_angular_error_deg.shape == (4, 5)
    assert loaded.error_direction_xy.shape == (4, 5, 2)
    assert loaded.error_delta_xy.shape == (4, 5, 2)
    assert loaded.peak_pixel_xy.shape == (4, 5, 2)
    assert np.max(loaded.max_angular_error_deg) < 2e-6


def test_plot_unproject_lut_error_heatmap_supports_angular_units(tmp_path: Path) -> None:
    """The heatmap plot helper rescales angular error into the requested units."""
    import matplotlib

    from lensboy.analysis import (
        compute_lut_error_heatmap,
        plot_unproject_lut_error_heatmap,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap = compute_lut_error_heatmap(lut, model, interpolation="bilinear")
    expected_mdeg = heatmap.max_angular_error_deg * 1.0e3

    fig = plot_unproject_lut_error_heatmap(
        heatmap,
        angular_unit="mdeg",
        show_directions=False,
        return_figure=True,
    )
    assert fig is not None
    actual = fig.axes[0].images[0].get_array()
    assert actual is not None
    np.testing.assert_allclose(actual, expected_mdeg)
    assert (
        fig.axes[0].get_title() == "Per-cell max error heatmap (bilinear) [milli degrees]"
    )
    assert fig.axes[1].get_ylabel() == "max angular error [milli degrees]"
    assert len(fig.axes[0].collections) == 0
    plt.close(fig)


def test_plot_unproject_lut_error_heatmap_accepts_figsize(tmp_path: Path) -> None:
    """The heatmap plot helper forwards the requested figure size."""
    import matplotlib

    from lensboy.analysis import (
        compute_lut_error_heatmap,
        plot_unproject_lut_error_heatmap,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap_path = tmp_path / "bilinear_error_heatmaps_figsize.npz"
    heatmap = compute_lut_error_heatmap(lut, model, interpolation="bilinear")
    heatmap.save(heatmap_path)

    fig = plot_unproject_lut_error_heatmap(
        heatmap_path,
        figsize=(7.8, 5.3),
        return_figure=True,
    )
    assert fig is not None
    np.testing.assert_allclose(fig.get_size_inches(), np.array([7.8, 5.3]))
    assert fig.axes[0].get_aspect() == 1.0
    plt.close(fig)


def test_unproject_lut_cpp_smoke(tmp_path: Path) -> None:
    """The standalone C++ loader can compile, load, and query a LUT."""
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("No C++ compiler available.")

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    lut_path = tmp_path / "cpp_round_trip_lut"
    lut.save(lut_path)
    loaded = lb.UnprojectLUT.load(lut_path)

    program_path = tmp_path / "smoke.cpp"
    binary_path = tmp_path / "smoke"
    program_path.write_text(
        textwrap.dedent(
            """
            #include "unproject_lut.hpp"

            #include <iomanip>
            #include <iostream>

            int main(int argc, char** argv) {
                auto const lut = lensboy::UnprojectLUT::load(argv[1]);

                auto const q0 = lut.query(
                    4.25,
                    3.5,
                    lensboy::InterpolationMode::kBilinear,
                    lensboy::BoundsMode::kStrict
                );
                auto const q1 = lut.query(
                    -1.0,
                    2.0,
                    lensboy::InterpolationMode::kBilinear,
                    lensboy::BoundsMode::kStrict
                );
                auto const q2 = lut.query(
                    -1.0,
                    2.0,
                    lensboy::InterpolationMode::kBilinear,
                    lensboy::BoundsMode::kExtrapolate,
                    false
                );

                std::cout << std::setprecision(17);
                std::cout << q0.valid << " " << q0.ray[0] << " " << q0.ray[1] << " "
                          << q0.ray[2] << "\\n";
                std::cout << q1.valid << "\\n";
                std::cout << q2.valid << " " << q2.ray[0] << " " << q2.ray[1] << " "
                          << q2.ray[2] << "\\n";
                auto const q3 = lut.query(
                    0.0,
                    2.0,
                    lensboy::InterpolationMode::kBicubic,
                    lensboy::BoundsMode::kStrict
                );
                std::cout << q3.valid << " " << q3.ray[0] << " " << q3.ray[1] << " "
                          << q3.ray[2] << "\\n";
                return 0;
            }
            """
        )
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            str(program_path),
            str(CPP_RUNTIME_DIR / "unproject_lut.cpp"),
            "-I",
            str(CPP_RUNTIME_DIR),
            "-O2",
            "-o",
            str(binary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0

    run_result = subprocess.run(
        [str(binary_path), str(lut_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = run_result.stdout.strip().splitlines()
    assert len(lines) == 4

    q0_tokens = lines[0].split()
    assert q0_tokens[0] == "1"
    python_q0 = loaded.normalize_points(
        np.array([[4.25, 3.5]]),
        interpolation="bilinear",
        bounds="strict",
    )[0]
    python_q0 = python_q0 / np.linalg.norm(python_q0)
    np.testing.assert_allclose(
        np.array([float(q0_tokens[1]), float(q0_tokens[2]), float(q0_tokens[3])]),
        python_q0,
        atol=1e-5,
    )

    assert lines[1] == "0"

    q2_tokens = lines[2].split()
    assert q2_tokens[0] == "1"
    python_q2 = loaded.normalize_points(
        np.array([[-1.0, 2.0]]),
        interpolation="bilinear",
        bounds="extrapolate",
    )[0]
    np.testing.assert_allclose(
        np.array([float(q2_tokens[1]), float(q2_tokens[2]), float(q2_tokens[3])]),
        python_q2,
        atol=1e-5,
    )

    q3_tokens = lines[3].split()
    assert q3_tokens[0] == "1"
    python_q3 = loaded.normalize_points(
        np.array([[0.0, 2.0]]),
        interpolation="bilinear",
        bounds="strict",
    )[0]
    python_q3 = python_q3 / np.linalg.norm(python_q3)
    np.testing.assert_allclose(
        np.array([float(q3_tokens[1]), float(q3_tokens[2]), float(q3_tokens[3])]),
        python_q3,
        atol=5e-4,
    )
