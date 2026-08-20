"""Round-trip serialization tests for camera models."""

from pathlib import Path

import numpy as np

import lensboy as lb

DATA_DIR = Path(__file__).parent.parent / "data/test_datasets"


def test_spline_serialization_round_trip(tmp_path: Path) -> None:
    """PinholeSplined survives a save/load cycle with identical projections."""
    original = lb.PinholeSplined.load(DATA_DIR / "spline.json")
    original.save(tmp_path / "spline.json")
    loaded = lb.PinholeSplined.load(tmp_path / "spline.json")

    pts = np.array([[100.0, 200.0], [1500.0, 1000.0], [3000.0, 2000.0]])
    bearings = original.normalize_points(pts)

    np.testing.assert_array_equal(
        original.project_points(bearings),
        loaded.project_points(bearings),
    )
    np.testing.assert_array_equal(
        original.normalize_points(pts),
        loaded.normalize_points(pts),
    )


def test_pinhole_remapped_serialization_round_trip(tmp_path: Path) -> None:
    """PinholeRemapped survives a save/load cycle with identical remap tables."""
    spline_model = lb.PinholeSplined.load(DATA_DIR / "spline.json")
    original = spline_model.get_pinhole_model()

    original.save(tmp_path / "pinhole")
    loaded = lb.PinholeRemapped.load(tmp_path / "pinhole")

    np.testing.assert_array_equal(original.map_x, loaded.map_x)
    np.testing.assert_array_equal(original.map_y, loaded.map_y)

    pts = np.array([[100.0, 200.0], [1500.0, 1000.0], [3000.0, 2000.0]])
    np.testing.assert_array_equal(
        original.project_points(original.normalize_points(pts)),
        loaded.project_points(loaded.normalize_points(pts)),
    )


def test_opencv_serialization_round_trip(tmp_path: Path) -> None:
    """OpenCV model survives a save/load cycle with identical projections."""
    original = lb.OpenCV.load(DATA_DIR / "opencv.json")
    original.save(tmp_path / "opencv.json")
    loaded = lb.OpenCV.load(tmp_path / "opencv.json")

    pts = np.array([[100.0, 200.0], [1500.0, 1000.0], [3000.0, 2000.0]])
    bearings = original.normalize_points(pts)

    np.testing.assert_array_equal(
        original.project_points(bearings),
        loaded.project_points(bearings),
    )
    np.testing.assert_array_equal(
        original.normalize_points(pts),
        loaded.normalize_points(pts),
    )


def test_stereographic_opencv_serialization_round_trip(tmp_path: Path) -> None:
    """StereographicOpenCV survives a save/load cycle."""
    original = lb.StereographicOpenCV(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=301.0,
        cx=320.0,
        cy=240.0,
        distortion_coeffs=np.array([0.01, -0.001, 0.0, 0.0, 0.0001]),
    )
    original.save(tmp_path / "stereo_opencv.json")
    loaded = lb.StereographicOpenCV.load(tmp_path / "stereo_opencv.json")

    rays = np.array([[0.0, 0.0, 1.0], [0.5, 0.1, 0.8], [0.8, -0.1, -0.2]])
    pts = original.project_points(rays)

    np.testing.assert_array_equal(original.project_points(rays), loaded.project_points(rays))
    np.testing.assert_array_equal(
        original.normalize_points(pts),
        loaded.normalize_points(pts),
    )


def test_stereographic_spline_serialization_round_trip(tmp_path: Path) -> None:
    """StereographicSplined survives a save/load cycle."""
    dx = np.zeros((8, 12), dtype=np.float64)
    dy = np.zeros((8, 12), dtype=np.float64)
    dx[3, 4] = 0.001
    dy[4, 5] = -0.001
    original = lb.StereographicSplined(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=301.0,
        cx=320.0,
        cy=240.0,
        dx_grid=dx,
        dy_grid=dy,
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=220.0,
        fov_deg_y=180.0,
    )
    original.save(tmp_path / "stereo_spline.json")
    loaded = lb.StereographicSplined.load(tmp_path / "stereo_spline.json")

    rays = np.array([[0.0, 0.0, 1.0], [0.5, 0.1, 0.8], [0.8, -0.1, -0.2]])
    pts = original.project_points(rays)

    np.testing.assert_array_equal(original.project_points(rays), loaded.project_points(rays))
    np.testing.assert_array_equal(
        original.normalize_points(pts),
        loaded.normalize_points(pts),
    )
