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


def _per_point_angular_error_deg(
    reference: np.ndarray,
    approx: np.ndarray,
) -> np.ndarray:
    """Per-point angular error (deg) between two batches of rays of shape (N, 3).

    Uses atan2(‖cross‖, dot) instead of acos(dot/norms) so the answer keeps
    full precision near zero — acos catastrophically loses digits when its
    argument is close to 1.
    """
    cross = np.cross(reference, approx)
    cross_norm = np.linalg.norm(cross, axis=1)
    dot = np.einsum("ij,ij->i", reference, approx)
    return np.rad2deg(np.arctan2(cross_norm, dot))


def _query_error_deg(
    reference: np.ndarray,
    approx: np.ndarray,
) -> float:
    return float(np.max(_per_point_angular_error_deg(reference, approx)))


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
    actual, _ = loaded.normalize_points(pixels, interpolation="bilinear")
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

    approx, _ = loaded.normalize_points(sample_pixels, interpolation="bilinear")
    max_abs_error = float(np.max(np.abs(approx[:, :2] - expected[:, :2])))
    assert max_abs_error < 1e-6


def test_unproject_lut_heatmap_bilinear_beats_nearest() -> None:
    """Bilinear interpolation has a smaller worst-cell error than nearest."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))

    max_errors = {
        mode: float(
            compute_lut_error_heatmap(
                lut, model, interpolation=mode
            ).max_angular_error_deg.max()
        )
        for mode in ("nearest", "bilinear", "bicubic")
    }

    for value in max_errors.values():
        assert np.isfinite(value)
    assert max_errors["bilinear"] < max_errors["nearest"]


def test_unproject_lut_heatmap_matches_loaded_and_in_memory_lut(tmp_path: Path) -> None:
    """Loaded LUTs produce the same heatmap max as in-memory LUTs."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    dir_path = tmp_path / "opencv_lut"
    lut.save(dir_path)
    loaded = lb.UnprojectLUT.load(dir_path)

    for mode in ("nearest", "bilinear", "bicubic"):
        before = float(
            compute_lut_error_heatmap(
                lut, model, interpolation=mode
            ).max_angular_error_deg.max()
        )
        after = float(
            compute_lut_error_heatmap(
                loaded, model, interpolation=mode
            ).max_angular_error_deg.max()
        )
        assert after == pytest.approx(before, rel=1e-3)


def test_unproject_lut_grid_stride_can_be_fractional() -> None:
    """The stored sample spacing reflects the actual grid spacing."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))

    stride_x, stride_y = lut.grid_stride_xy
    assert stride_x == pytest.approx(16.0 / 5.0)
    assert stride_y == pytest.approx(12.0 / 4.0)


def test_unproject_lut_out_of_bounds_returns_nan_and_mask() -> None:
    """Out-of-domain pixels get NaN rays and False in the valid mask."""
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

    rays, valid_mask = lut.normalize_points(pixels)
    np.testing.assert_array_equal(valid_mask, np.array([False, True, True, False]))
    assert np.isnan(rays[0, 0]) and np.isnan(rays[0, 1])
    assert np.isnan(rays[3, 0]) and np.isnan(rays[3, 1])


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
        small_lut.normalize_points(sample_pixels, interpolation="bicubic")[0],
        small_lut.normalize_points(sample_pixels, interpolation="bilinear")[0],
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
        ]
    )
    np.testing.assert_allclose(
        lut.normalize_points(boundary_pixels, interpolation="bicubic")[0],
        lut.normalize_points(boundary_pixels, interpolation="bilinear")[0],
        atol=1e-5,
    )


def test_unproject_lut_heatmap_bounds_dense_error_for_nonlinear_model() -> None:
    """Dense-sampled error stays under the heatmap's worst-cell max."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    heatmap = compute_lut_error_heatmap(lut, model, interpolation="bilinear")
    max_error_deg = float(heatmap.max_angular_error_deg.max())
    assert np.isfinite(max_error_deg)

    xs = np.linspace(0.0, model.image_width - 1, 35)
    ys = np.linspace(0.0, model.image_height - 1, 27)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    exact = model.normalize_points(pixels)
    approx, _ = lut.normalize_points(pixels, interpolation="bilinear")
    dense_error = _query_error_deg(exact, approx)
    assert dense_error <= max_error_deg + 1.0


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


def _heatmap_test_lut(model: lb.OpenCV) -> lb.UnprojectLUT:
    """Realistic LUT used by the heatmap correctness tests."""
    return model.get_unproject_lut(pixel_stride=32.0)


def test_heatmap_peak_pixel_reproduces_reported_error() -> None:
    """The reported per-cell peak pixel re-evaluates to the reported angular error."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = _heatmap_test_lut(model)

    for mode in ("nearest", "bilinear", "bicubic"):
        heatmap = compute_lut_error_heatmap(lut, model, interpolation=mode)
        peaks = heatmap.peak_pixel_xy.reshape(-1, 2)
        exact_rays = model.normalize_points(peaks)
        approx_rays, _ = lut.normalize_points(peaks, interpolation=mode)
        actual_deg = _per_point_angular_error_deg(exact_rays, approx_rays)
        expected_deg = heatmap.max_angular_error_deg.reshape(-1)
        np.testing.assert_allclose(actual_deg, expected_deg, rtol=1e-7, atol=1e-9)


def test_heatmap_peak_pixel_lies_inside_its_cell() -> None:
    """The optimiser respects the ReLU cell-membership constraint."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = _heatmap_test_lut(model)

    for mode in ("nearest", "bilinear", "bicubic"):
        heatmap = compute_lut_error_heatmap(lut, model, interpolation=mode)
        x_edges = heatmap.cell_x_edges
        y_edges = heatmap.cell_y_edges
        peak_x = heatmap.peak_pixel_xy[..., 0]
        peak_y = heatmap.peak_pixel_xy[..., 1]
        x_lo = x_edges[:-1][None, :]
        x_hi = x_edges[1:][None, :]
        y_lo = y_edges[:-1][:, None]
        y_hi = y_edges[1:][:, None]
        eps = 1e-9
        assert np.all(peak_x >= x_lo - eps)
        assert np.all(peak_x <= x_hi + eps)
        assert np.all(peak_y >= y_lo - eps)
        assert np.all(peak_y <= y_hi + eps)


def test_heatmap_error_delta_matches_approx_minus_exact_at_peak() -> None:
    """error_delta_xy equals (approx_xy − exact_xy) at the reported peak pixel."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = _heatmap_test_lut(model)

    for mode in ("nearest", "bilinear", "bicubic"):
        heatmap = compute_lut_error_heatmap(lut, model, interpolation=mode)
        peaks = heatmap.peak_pixel_xy.reshape(-1, 2)
        exact_xy = model.normalize_points(peaks)[:, :2]
        approx_xy, _ = lut.normalize_points(peaks, interpolation=mode)
        expected_delta = (approx_xy[:, :2] - exact_xy).reshape(
            heatmap.error_delta_xy.shape
        )
        np.testing.assert_allclose(
            heatmap.error_delta_xy, expected_delta, rtol=1e-7, atol=1e-9
        )


def test_heatmap_per_cell_max_matches_dense_brute_force() -> None:
    """Per-cell optimiser maxima match a dense brute-force search inside each cell.

    For a realistic LUT and a non-trivially distorted model, brute-force a
    sample of cells with a dense grid of pixels and compare the per-cell
    angular-error max to what the optimiser reported. The optimiser must
    reach at least the brute-force max, and on a 25x25 brute grid the gap
    to the true peak should be very small.
    """
    from lensboy.analysis import compute_lut_error_heatmap

    model = _load_opencv_model()
    lut = _heatmap_test_lut(model)
    samples_per_axis = 25
    n_cells_to_check = 50
    interpolation = "bilinear"

    heatmap = compute_lut_error_heatmap(lut, model, interpolation=interpolation)
    x_edges = heatmap.cell_x_edges
    y_edges = heatmap.cell_y_edges
    height, width = heatmap.max_angular_error_deg.shape

    rng = np.random.default_rng(0)
    sample_iy = rng.integers(0, height, size=n_cells_to_check)
    sample_ix = rng.integers(0, width, size=n_cells_to_check)

    for iy, ix in zip(sample_iy, sample_ix, strict=True):
        xs = np.linspace(x_edges[ix], x_edges[ix + 1], samples_per_axis)
        ys = np.linspace(y_edges[iy], y_edges[iy + 1], samples_per_axis)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        pixels = np.column_stack([gx.ravel(), gy.ravel()])

        exact_rays = model.normalize_points(pixels)
        approx_rays, _ = lut.normalize_points(pixels, interpolation=interpolation)
        brute_max_deg = _query_error_deg(exact_rays, approx_rays)
        heatmap_max_deg = float(heatmap.max_angular_error_deg[iy, ix])

        # Optimiser must reach at least the brute-force max (within tight
        # numerical slop). With a 25x25 brute grid the gap to the true peak
        # is small in absolute terms.
        assert heatmap_max_deg >= brute_max_deg - 1e-9, (
            f"cell ({iy},{ix}): heatmap={heatmap_max_deg}  brute={brute_max_deg}"
        )
        assert heatmap_max_deg <= brute_max_deg + 1e-5, (
            f"cell ({iy},{ix}): heatmap={heatmap_max_deg}  brute={brute_max_deg}"
        )


def test_heatmap_is_essentially_zero_for_linear_pinhole() -> None:
    """Bilinear and bicubic interpolation match a linear pinhole exactly."""
    from lensboy.analysis import compute_lut_error_heatmap

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))

    for mode in ("bilinear", "bicubic"):
        heatmap = compute_lut_error_heatmap(lut, model, interpolation=mode)
        assert np.max(heatmap.max_angular_error_deg) < 1e-6


def test_heatmap_plot_supports_angular_units() -> None:
    """The heatmap plot method rescales angular error into the requested units."""
    import matplotlib

    from lensboy.analysis import compute_lut_error_heatmap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap = compute_lut_error_heatmap(lut, model, interpolation="bilinear")
    expected_mdeg = heatmap.max_angular_error_deg * 1.0e3

    fig = heatmap.plot(
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


def test_heatmap_plot_accepts_figsize() -> None:
    """The heatmap plot method forwards the requested figure size."""
    import matplotlib

    from lensboy.analysis import compute_lut_error_heatmap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap = compute_lut_error_heatmap(lut, model, interpolation="bilinear")

    fig = heatmap.plot(
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
                    lensboy::InterpolationMode::kBilinear
                );
                auto const q1 = lut.query(
                    -1.0,
                    2.0,
                    lensboy::InterpolationMode::kBilinear
                );
                auto const q2 = lut.query(
                    0.0,
                    2.0,
                    lensboy::InterpolationMode::kBicubic
                );

                std::cout << std::setprecision(17);
                std::cout << q0.valid << " " << q0.ray[0] << " " << q0.ray[1] << " "
                          << q0.ray[2] << "\\n";
                std::cout << q1.valid << "\\n";
                std::cout << q2.valid << " " << q2.ray[0] << " " << q2.ray[1] << " "
                          << q2.ray[2] << "\\n";
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
    assert len(lines) == 3

    q0_tokens = lines[0].split()
    assert q0_tokens[0] == "1"
    python_q0, _ = loaded.normalize_points(
        np.array([[4.25, 3.5]]),
        interpolation="bilinear",
    )
    python_q0 = python_q0[0]
    python_q0 = python_q0 / np.linalg.norm(python_q0)
    np.testing.assert_allclose(
        np.array([float(q0_tokens[1]), float(q0_tokens[2]), float(q0_tokens[3])]),
        python_q0,
        atol=1e-5,
    )

    assert lines[1] == "0"

    q2_tokens = lines[2].split()
    assert q2_tokens[0] == "1"
    python_q2, _ = loaded.normalize_points(
        np.array([[0.0, 2.0]]),
        interpolation="bicubic",
    )
    python_q2 = python_q2[0]
    python_q2 = python_q2 / np.linalg.norm(python_q2)
    np.testing.assert_allclose(
        np.array([float(q2_tokens[1]), float(q2_tokens[2]), float(q2_tokens[3])]),
        python_q2,
        atol=5e-4,
    )
