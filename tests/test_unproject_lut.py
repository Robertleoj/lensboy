"""Tests for UnprojectLUT creation, serialization, querying, and C++ loading."""

from __future__ import annotations

import hashlib
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


def _make_linear_pinhole_model() -> lb.PinholeRemapped:
    return lb.PinholeRemapped(
        image_width=17,
        image_height=13,
        fx=23.0,
        fy=19.0,
        cx=8.0,
        cy=6.0,
        map_x=np.zeros((13, 17), dtype=np.float32),
        map_y=np.zeros((13, 17), dtype=np.float32),
        input_image_width=17,
        input_image_height=13,
    )


def _load_opencv_model() -> lb.OpenCV:
    return lb.OpenCV.load(DATA_DIR / "opencv.json")


def _load_spline_model() -> lb.PinholeSplined:
    return lb.PinholeSplined.load(DATA_DIR / "spline.json")


def _load_remapped_model() -> lb.PinholeRemapped:
    return _load_spline_model().get_pinhole_model()


def _header_and_payload(file_path: Path) -> tuple[str, bytes]:
    data = file_path.read_bytes()
    marker = b"END_HEADER\n"
    index = data.index(marker) + len(marker)
    return data[:index].decode("ascii"), data[index:]


def _rewrite_header_field(
    file_path: Path,
    key: str,
    new_value: str,
    *,
    recompute_payload_offset: bool = True,
) -> None:
    header, payload = _header_and_payload(file_path)
    lines = header.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            lines[i] = f"{key}: {new_value}"
            break
    else:
        raise AssertionError(f"Missing header key {key!r}.")

    if recompute_payload_offset:
        payload_offset_index = None
        for i, line in enumerate(lines):
            if line.startswith("payload_offset_bytes:"):
                payload_offset_index = i
                break
        if payload_offset_index is not None:
            payload_offset_text = "0"
            while True:
                lines[payload_offset_index] = (
                    f"payload_offset_bytes: {payload_offset_text}"
                )
                header_text = "\n".join(lines) + "\n"
                next_payload_offset_text = str(len(header_text.encode("ascii")))
                if next_payload_offset_text == payload_offset_text:
                    break
                payload_offset_text = next_payload_offset_text

    file_path.write_bytes(("\n".join(lines) + "\n").encode("ascii") + payload)


def _append_header_field(file_path: Path, key: str, value: str) -> None:
    header, payload = _header_and_payload(file_path)
    lines = header.splitlines()
    end_header_index = lines.index("END_HEADER")
    lines.insert(end_header_index, f"{key}: {value}")

    payload_offset_index = lines.index(
        next(line for line in lines if line.startswith("payload_offset_bytes:"))
    )
    payload_offset_text = "0"
    while True:
        lines[payload_offset_index] = f"payload_offset_bytes: {payload_offset_text}"
        header_text = "\n".join(lines) + "\n"
        next_payload_offset_text = str(len(header_text.encode("ascii")))
        if next_payload_offset_text == payload_offset_text:
            break
        payload_offset_text = next_payload_offset_text

    file_path.write_bytes(("\n".join(lines) + "\n").encode("ascii") + payload)


def _parse_header_text(header: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in header.splitlines():
        if line == "END_HEADER":
            continue
        key, value = line.split(": ", 1)
        parsed[key] = value
    return parsed


def _random_pixels(
    model: lb.PinholeRemapped,
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


def test_unproject_lut_header_is_human_readable(tmp_path: Path) -> None:
    """The text header surfaces key metadata near the top of the file."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(5, 4))

    file_path = tmp_path / "linear.unproject_LUT"
    lut.save(file_path)

    header, _ = _header_and_payload(file_path)
    header_fields = _parse_header_text(header)
    header_lines = header.splitlines()
    first_lines = header_lines[:16]

    assert header_fields["format_version"] == "1"
    assert first_lines[0] == "format: lensboy_unproject_LUT"
    assert first_lines[1].startswith("payload_offset_bytes: ")
    assert first_lines[2] == "format_version: 1"
    assert first_lines[3].startswith("lensboy_version: ")
    assert first_lines[4] == "source_model_type: pinhole_remapped"
    assert first_lines[5].startswith("source_model_spec_json_sha256: ")
    assert first_lines[6] == "image_size_wh: 17, 13"
    assert first_lines[7] == "grid_size_wh: 5, 4"
    assert first_lines[8] == "grid_extents_xy: 0, 16, 0, 12"
    assert first_lines[9] == "grid_stride_xy: 4, 4"
    assert first_lines[10] == "storage_encoding: float64_xy"
    assert first_lines[11] == "default_interpolation: bilinear"
    assert first_lines[12] == "default_bounds: strict"
    assert first_lines[13] == "payload_layout: row_major_interleaved_xy"
    assert first_lines[14] == "payload_endianness: little"
    assert first_lines[15].startswith("source_model_spec_json: ")
    assert not any(
        line.startswith("supported_interpolations:") for line in header.splitlines()
    )
    assert not any(line.startswith("supported_bounds:") for line in header.splitlines())
    assert not any(
        line.startswith("estimated_max_angular_error_") for line in header.splitlines()
    )
    assert not any(
        line.startswith("estimated_median_angular_error_") for line in header.splitlines()
    )
    assert not any(line.startswith("error_report_") for line in header.splitlines())
    assert any(line.startswith("source_model_spec_json:") for line in header.splitlines())
    assert int(header_fields["payload_offset_bytes"]) == len(header.encode("ascii"))
    assert (
        header_fields["source_model_spec_json_sha256"]
        == hashlib.sha256(
            header_fields["source_model_spec_json"].encode("ascii")
        ).hexdigest()
    )


def test_unproject_lut_exposes_serialized_file_metadata(tmp_path: Path) -> None:
    """The runtime object exposes its serialized header and size information."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(5, 4))
    file_path = tmp_path / "linear.unproject_LUT"

    lut.save(file_path)
    loaded = lb.UnprojectLUT.load(file_path)
    header, payload = _header_and_payload(file_path)

    assert loaded.header_text == header
    assert loaded.payload_offset_bytes == len(header.encode("ascii"))
    assert loaded.payload_bytes == len(payload)
    assert loaded.total_bytes == file_path.stat().st_size
    assert loaded.header_preview() == header.rstrip("\n")
    assert loaded.header_preview(12) == "\n".join(header.splitlines()[:12])


@pytest.mark.parametrize(
    ("factory", "expected_model_type"),
    [
        (_load_opencv_model, "opencv"),
        (_load_spline_model, "pinhole_splined"),
        (_load_remapped_model, "pinhole_remapped"),
    ],
)
def test_unproject_lut_round_trip_for_camera_models(
    tmp_path: Path,
    factory,
    expected_model_type: str,
) -> None:
    """OpenCV, spline, and remapped models survive LUT save/load round trips."""
    model = factory()
    lut = model.get_unproject_lut(
        grid_size_wh=(11, 9),
        storage_encoding="float64_xy",
    )

    file_path = tmp_path / f"{expected_model_type}.unproject_LUT"
    lut.save(file_path)
    loaded = lb.UnprojectLUT.load(file_path)

    assert loaded.source_model_type == expected_model_type
    assert loaded.grid_size_wh == (11, 9)
    assert loaded.storage_encoding == "float64_xy"
    assert loaded.source_model_spec is not None
    assert loaded.source_model_spec["type"] == expected_model_type
    assert loaded.source_model_spec_json_sha256 is not None

    x_coords = np.linspace(0.0, model.image_width - 1, loaded.grid_width)
    y_coords = np.linspace(0.0, model.image_height - 1, loaded.grid_height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    expected = model.normalize_points(pixels)
    actual = loaded.normalize_points(pixels, interpolation="bilinear")
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_linear_pinhole_lut_encodings(tmp_path: Path) -> None:
    """Linear pinhole data isolates storage-encoding error from interpolation error."""
    model = _make_linear_pinhole_model()
    sample_pixels = _random_pixels(model, seed=4)
    expected = model.normalize_points(sample_pixels)
    max_abs_errors: dict[str, float] = {}

    for encoding in ("float64_xy", "float32_xy", "float16_xy"):
        lut = model.get_unproject_lut(
            grid_size_wh=(6, 5),
            storage_encoding=encoding,
        )
        file_path = tmp_path / f"{encoding}.unproject_LUT"
        lut.save(file_path)
        loaded = lb.UnprojectLUT.load(file_path)

        approx = loaded.normalize_points(sample_pixels, interpolation="bilinear")
        max_abs_errors[encoding] = float(np.max(np.abs(approx[:, :2] - expected[:, :2])))

    assert max_abs_errors["float64_xy"] < 1e-12
    assert max_abs_errors["float32_xy"] < 1e-6
    assert max_abs_errors["float16_xy"] < 1e-3
    assert max_abs_errors["float64_xy"] <= max_abs_errors["float32_xy"] + 1e-15
    assert max_abs_errors["float32_xy"] <= max_abs_errors["float16_xy"] + 1e-15


def test_unproject_lut_analyzer_can_report_multiple_interpolations() -> None:
    """The analyzer can report multiple interpolation modes at once."""
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    report = UnprojectLUTAnalyzer(lut).estimate_accuracy(
        interpolations=("nearest", "bilinear", "bicubic")
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
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    report = UnprojectLUTAnalyzer(lut).estimate_accuracy(interpolations="bicubic")

    assert report.interpolations == ("bicubic",)
    assert set(report.max_angular_error_mdeg) == {"bicubic"}
    assert set(report.median_angular_error_mdeg) == {"bicubic"}
    assert np.isfinite(report.max_angular_error_mdeg["bicubic"])
    assert np.isfinite(report.median_angular_error_mdeg["bicubic"])


def test_unproject_lut_analyzer_matches_loaded_and_in_memory_lut(tmp_path: Path) -> None:
    """Loaded LUTs produce the same analyzer report as in-memory LUTs."""
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    file_path = tmp_path / "opencv.unproject_LUT"
    lut.save(file_path)
    loaded = lb.UnprojectLUT.load(file_path)

    report_before = UnprojectLUTAnalyzer(lut).estimate_accuracy(
        interpolations=("nearest", "bilinear", "bicubic")
    )
    report_after = UnprojectLUTAnalyzer(loaded).estimate_accuracy(
        interpolations=("nearest", "bilinear", "bicubic")
    )

    assert report_after.interpolations == report_before.interpolations
    for mode in report_before.interpolations:
        assert report_after.max_angular_error_mdeg[mode] == pytest.approx(
            report_before.max_angular_error_mdeg[mode]
        )
        assert report_after.median_angular_error_mdeg[mode] == pytest.approx(
            report_before.median_angular_error_mdeg[mode]
        )


def test_unproject_lut_analyzer_can_sample_dense_accuracy_grid() -> None:
    """The analyzer can compare LUT rays against exact rays on a dense sample grid."""
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    sample = UnprojectLUTAnalyzer(lut).sample_accuracy_grid(
        interpolation="bilinear",
        target_sample_count=2500,
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
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    sample = UnprojectLUTAnalyzer(lut).sample_accuracy_grid(
        interpolation="bilinear",
        target_sample_count=2500,
    )

    np.testing.assert_allclose(sample.approx_rays, sample.exact_rays, atol=1e-12)
    assert sample.max_angular_error_mdeg < 1e-2


def test_unproject_lut_parallel_grid_build_matches_serial() -> None:
    """Parallel LUT grid sampling matches the serial build exactly."""
    model = _make_linear_pinhole_model()
    serial_lut = model.get_unproject_lut(
        grid_size_wh=(17, 13),
        num_workers=1,
    )
    parallel_lut = model.get_unproject_lut(
        grid_size_wh=(17, 13),
        num_workers=2,
    )

    np.testing.assert_allclose(parallel_lut.xy_grid, serial_lut.xy_grid, atol=0.0)
    assert parallel_lut.grid_size_wh == serial_lut.grid_size_wh
    assert parallel_lut.grid_extents_xy == serial_lut.grid_extents_xy


def test_unproject_lut_analyzer_is_stable_across_num_workers() -> None:
    """Changing LUT build workers does not affect analyzer results."""
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _load_opencv_model()
    serial_lut = model.get_unproject_lut(
        grid_size_wh=(7, 6),
        num_workers=1,
    )
    parallel_lut = model.get_unproject_lut(
        grid_size_wh=(7, 6),
        num_workers=2,
    )
    serial_report = UnprojectLUTAnalyzer(serial_lut).estimate_accuracy(
        interpolations=("nearest", "bilinear", "bicubic")
    )
    parallel_report = UnprojectLUTAnalyzer(parallel_lut).estimate_accuracy(
        interpolations=("nearest", "bilinear", "bicubic")
    )

    for mode in ("nearest", "bilinear", "bicubic"):
        assert parallel_report.max_angular_error_mdeg[mode] == pytest.approx(
            serial_report.max_angular_error_mdeg[mode]
        )
        assert parallel_report.median_angular_error_mdeg[mode] == pytest.approx(
            serial_report.median_angular_error_mdeg[mode]
        )


def test_unproject_lut_grid_stride_can_be_fractional() -> None:
    """The stored sample spacing reflects the actual grid spacing and may be fractional."""
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
    np.testing.assert_allclose(clamp_actual, clamp_expected, atol=1e-12)

    extrap_expected = model.normalize_points(pixels)
    extrap_actual = lut.normalize_points(
        pixels,
        interpolation="bilinear",
        bounds="extrapolate",
    )
    np.testing.assert_allclose(extrap_actual, extrap_expected, atol=1e-12)


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
        atol=1e-12,
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
        atol=1e-12,
    )


def test_unproject_lut_analyzer_report_is_finite_for_nonlinear_model() -> None:
    """A nonlinear model produces finite observed angular error summaries."""
    from lensboy.analysis import UnprojectLUTAnalyzer

    model = _load_opencv_model()
    lut = model.get_unproject_lut(grid_size_wh=(7, 6))
    report = UnprojectLUTAnalyzer(lut).estimate_accuracy()

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
    dense_error = _query_error_deg(exact, approx)
    assert dense_error <= (report.max_angular_error_mdeg["bilinear"] / 1.0e3) + 1.0


def test_unproject_lut_analyzer_requires_source_model_spec() -> None:
    """Analyzer methods fail clearly when the LUT lacks an exact source model spec."""
    from lensboy.analysis import UnprojectLUTAnalyzer

    lut = lb.UnprojectLUT(
        image_width=5,
        image_height=4,
        grid_width=5,
        grid_height=4,
        grid_x_min=0.0,
        grid_x_max=4.0,
        grid_y_min=0.0,
        grid_y_max=3.0,
        storage_encoding="float64_xy",
        xy_grid=np.zeros((4, 5, 2), dtype=np.float64),
        source_model_type=None,
        source_model_spec=None,
    )

    with pytest.raises(ValueError, match="source_model_spec_json"):
        UnprojectLUTAnalyzer(lut).estimate_accuracy()
    with pytest.raises(ValueError, match="source_model_spec_json"):
        UnprojectLUTAnalyzer(lut).sample_accuracy_grid()


def test_unproject_lut_rejects_wrong_suffix(tmp_path: Path) -> None:
    """Saving requires the `.unproject_LUT` suffix."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))

    with pytest.raises(ValueError, match=".unproject_LUT"):
        lut.save(tmp_path / "invalid.bin")


def test_unproject_lut_rejects_missing_header_marker(tmp_path: Path) -> None:
    """Loading fails cleanly when the text header is incomplete."""
    file_path = tmp_path / "broken.unproject_LUT"
    file_path.write_bytes(b"format: lensboy_unproject_LUT\n")

    with pytest.raises(ValueError, match="END_HEADER"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_rejects_invalid_header_line(tmp_path: Path) -> None:
    """Loading fails when a header line does not follow `key: value`."""
    file_path = tmp_path / "broken.unproject_LUT"
    file_path.write_bytes(b"format lensboy_unproject_LUT\nEND_HEADER\n")

    with pytest.raises(ValueError, match="key: value"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_rejects_bad_payload_offset(tmp_path: Path) -> None:
    """Loading validates the declared payload byte offset."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    file_path = tmp_path / "bad_offset.unproject_LUT"
    lut.save(file_path)

    _rewrite_header_field(
        file_path,
        "payload_offset_bytes",
        "1",
        recompute_payload_offset=False,
    )

    with pytest.raises(ValueError, match="payload_offset_bytes"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_rejects_bad_source_model_spec_hash(tmp_path: Path) -> None:
    """Loading validates the source-model spec hash."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    file_path = tmp_path / "bad_hash.unproject_LUT"
    lut.save(file_path)

    _rewrite_header_field(file_path, "source_model_spec_json_sha256", "deadbeef")

    with pytest.raises(ValueError, match="source_model_spec_json_sha256"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_rejects_legacy_error_report_header_fields(tmp_path: Path) -> None:
    """Loading rejects the older mixed runtime-and-analysis header shape."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    file_path = tmp_path / "legacy_error_report.unproject_LUT"
    lut.save(file_path)

    _append_header_field(
        file_path,
        "estimated_max_angular_error_mdeg_bilinear",
        "12.5",
    )

    with pytest.raises(ValueError, match="legacy error-report header fields"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_rejects_unsupported_storage_encoding(tmp_path: Path) -> None:
    """Loading validates the declared storage encoding."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    file_path = tmp_path / "encoding.unproject_LUT"
    lut.save(file_path)

    _rewrite_header_field(file_path, "storage_encoding", "float128_xy")

    with pytest.raises(ValueError, match="Unsupported storage encoding"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_rejects_truncated_payload(tmp_path: Path) -> None:
    """Loading validates the binary payload length."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    file_path = tmp_path / "truncated.unproject_LUT"
    lut.save(file_path)

    header, payload = _header_and_payload(file_path)
    file_path.write_bytes(header.encode("ascii") + payload[:-3])

    with pytest.raises(ValueError, match="Unexpected payload size"):
        lb.UnprojectLUT.load(file_path)


@pytest.mark.parametrize("replacement", [np.nan, np.inf])
def test_unproject_lut_rejects_nonfinite_payload_values(
    tmp_path: Path,
    replacement: float,
) -> None:
    """Loading rejects NaN and infinity in the cached payload."""
    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(4, 4))
    file_path = tmp_path / "nonfinite.unproject_LUT"
    lut.save(file_path)

    header, payload = _header_and_payload(file_path)
    payload_array = np.frombuffer(payload, dtype="<f8").copy()
    payload_array[0] = replacement
    file_path.write_bytes(header.encode("ascii") + payload_array.tobytes())

    with pytest.raises(ValueError, match="finite"):
        lb.UnprojectLUT.load(file_path)


def test_unproject_lut_analyzer_can_save_and_load_error_heatmaps(tmp_path: Path) -> None:
    """Analyzer-generated heatmaps can be saved and loaded without using the LUT header."""
    from lensboy.analysis import UnprojectLUTAnalyzer, UnprojectLUTErrorHeatmap

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap_path = tmp_path / "bilinear_error_heatmaps.npz"
    analyzer = UnprojectLUTAnalyzer(lut)

    heatmap = analyzer.compute_error_heatmap(interpolation="bilinear")
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

    from lensboy.analysis import UnprojectLUTAnalyzer, plot_unproject_lut_error_heatmap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    heatmap = UnprojectLUTAnalyzer(lut).compute_error_heatmap(interpolation="bilinear")
    expected_mdeg = heatmap.max_angular_error_deg * 1.0e3

    fig = plot_unproject_lut_error_heatmap(
        heatmap,
        angular_unit="mdeg",
        show_directions=False,
        return_figure=True,
    )
    assert fig is not None
    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), expected_mdeg)
    assert (
        fig.axes[0].get_title() == "Per-cell max error heatmap (bilinear) [milli degrees]"
    )
    assert fig.axes[1].get_ylabel() == "max angular error [milli degrees]"
    assert len(fig.axes[0].collections) == 0
    plt.close(fig)


def test_plot_unproject_lut_error_heatmap_accepts_figsize(tmp_path: Path) -> None:
    """The heatmap plot helper forwards the requested figure size."""
    import matplotlib

    from lensboy.analysis import UnprojectLUTAnalyzer, plot_unproject_lut_error_heatmap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _make_linear_pinhole_model()
    lut = model.get_unproject_lut(grid_size_wh=(6, 5))
    analyzer = UnprojectLUTAnalyzer(lut)
    heatmap_path = tmp_path / "bilinear_error_heatmaps_figsize.npz"
    analyzer.save_error_heatmap(heatmap_path, interpolation="bilinear")

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
    lut = model.get_unproject_lut(
        grid_size_wh=(6, 5),
        storage_encoding="float16_xy",
    )
    lut_path = tmp_path / "cpp_round_trip.unproject_LUT"
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
        atol=1e-12,
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
        atol=1e-12,
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
