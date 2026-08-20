"""Round-trip tests: project then normalize, check we recover bearing vectors."""

from pathlib import Path

import numpy as np

from lensboy.camera_models.opencv import OpenCV
from lensboy.camera_models.pinhole_remapped import PinholeRemapped
from lensboy.camera_models.pinhole_splined import PinholeSplined
from lensboy.camera_models.stereographic_opencv import (
    StereographicOpenCV,
)
from lensboy.camera_models.stereographic_splined import StereographicSplined

DATA = Path(__file__).parent.parent / "data/test_datasets"


def _random_points_in_cam(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5, 0.5, (n, 2))
    z = rng.uniform(1.0, 5.0, (n, 1))
    return np.hstack([xy * z, z])


def test_pinhole_remapped_roundtrip() -> None:
    model = PinholeRemapped(
        image_width=640,
        image_height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        map_x=np.zeros((480, 640), dtype=np.float32),
        map_y=np.zeros((480, 640), dtype=np.float32),
        input_image_width=640,
        input_image_height=480,
    )
    points = _random_points_in_cam()
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / points[:, 2:3]
    np.testing.assert_allclose(normalized, expected, atol=1e-10)


def test_opencv_roundtrip() -> None:
    model = OpenCV.load(DATA / "opencv.json")
    points = _random_points_in_cam()
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / points[:, 2:3]
    np.testing.assert_allclose(normalized, expected, atol=1e-6)


def test_spline_roundtrip() -> None:
    model = PinholeSplined.load(DATA / "spline.json")
    points = _random_points_in_cam()
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / points[:, 2:3]
    np.testing.assert_allclose(normalized, expected, atol=1e-6)


def test_stereographic_opencv_roundtrip_unit_rays() -> None:
    model = StereographicOpenCV(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        distortion_coeffs=np.zeros(14, dtype=np.float64),
    )
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 1.0],
            [1.0, -0.1, -0.2],
        ],
        dtype=np.float64,
    )
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(normalized, expected, atol=1e-10)


def test_stereographic_splined_roundtrip_unit_rays() -> None:
    model = StereographicSplined(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        dx_grid=np.zeros((8, 12), dtype=np.float64),
        dy_grid=np.zeros((8, 12), dtype=np.float64),
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=220.0,
        fov_deg_y=180.0,
    )
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 1.0],
            [1.0, -0.1, -0.2],
        ],
        dtype=np.float64,
    )
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(normalized, expected, atol=1e-10)
