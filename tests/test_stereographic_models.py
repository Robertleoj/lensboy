"""Focused tests for stereographic camera model public contracts."""

import numpy as np
import pytest

import lensboy as lb
from lensboy.camera_models.opencv import K1, K2, K3, P1, P2, S1, S2, S3, S4


def _stereographic_xy(rays: np.ndarray) -> np.ndarray:
    unit = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    denom = 1.0 + unit[:, 2]
    return np.column_stack([unit[:, 0] * 2.0 / denom, unit[:, 1] * 2.0 / denom])


def _distort_opencv_xy(xy: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = coeffs
    assert tx == 0.0
    assert ty == 0.0

    radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = radial_num / radial_den

    x_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    x_prism = s1 * r2 + s2 * r4
    y_prism = s3 * r2 + s4 * r4
    return np.column_stack(
        [
            x * radial + x_tangential + x_prism,
            y * radial + y_tangential + y_prism,
        ]
    )


def _make_stereographic_opencv(coeffs: np.ndarray | None = None) -> lb.StereographicOpenCV:
    if coeffs is None:
        coeffs = np.zeros(14, dtype=np.float64)
    return lb.StereographicOpenCV(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=310.0,
        cx=321.0,
        cy=239.0,
        distortion_coeffs=coeffs,
    )


def _make_stereographic_splined() -> lb.StereographicSplined:
    dx = np.zeros((5, 6), dtype=np.float64)
    dy = np.zeros((5, 6), dtype=np.float64)
    dx[2, 3] = 0.0002
    dy[1, 4] = -0.0001
    return lb.StereographicSplined(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=310.0,
        cx=321.0,
        cy=239.0,
        dx_grid=dx,
        dy_grid=dy,
        num_knots_x=6,
        num_knots_y=5,
        fov_deg_x=170.0,
        fov_deg_y=140.0,
    )


def test_stereographic_opencv_config_mask_and_initial_value() -> None:
    """Config masks map to optimizer params and seed intrinsics exactly."""
    config = lb.StereographicOpenCVConfig(
        image_height=480,
        image_width=640,
        initial_focal_length=275.0,
        included_distortion_coefficients=lb.StereographicOpenCVConfig.STANDARD,
    )

    expected = np.zeros(18, dtype=bool)
    expected[:4] = True
    expected[4 + np.array([K1, K2, P1, P2, K3])] = True
    np.testing.assert_array_equal(config.optimize_mask(), expected)

    model = config.get_initial_value()
    assert model.image_height == 480
    assert model.image_width == 640
    assert model.fx == 275.0
    assert model.fy == 275.0
    assert model.cx == 320.0
    assert model.cy == 240.0
    np.testing.assert_array_equal(model.distortion_coeffs, np.zeros(14))


def test_stereographic_opencv_config_rejects_bad_masks() -> None:
    """Config rejects masks that would not match the 14 OpenCV coefficients."""
    with pytest.raises(AssertionError, match="Expected \\(14,\\) mask"):
        lb.StereographicOpenCVConfig(
            image_height=480,
            image_width=640,
            included_distortion_coefficients=np.ones(13, dtype=bool),
        )

    with pytest.raises(AssertionError, match="Expected bool dtype"):
        lb.StereographicOpenCVConfig(
            image_height=480,
            image_width=640,
            included_distortion_coefficients=np.ones(14, dtype=np.int32),
        )


def test_stereographic_opencv_get_initial_value_requires_focal_length() -> None:
    """Config cannot create a seed model without an explicit focal length."""
    config = lb.StereographicOpenCVConfig(image_height=480, image_width=640)
    with pytest.raises(ValueError, match="initial_focal_length"):
        config.get_initial_value()


def test_stereographic_opencv_project_points_matches_opencv_plane_distortion() -> None:
    """Projection applies OpenCV distortion to stereographic coordinates."""
    coeffs = np.zeros(14, dtype=np.float64)
    coeffs[[K1, K2, P1, P2, K3, S1, S2, S3, S4]] = [
        0.03,
        -0.004,
        0.001,
        -0.0007,
        0.0002,
        0.0003,
        -0.00004,
        -0.0002,
        0.00005,
    ]
    model = _make_stereographic_opencv(coeffs)
    rays = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.3, -0.2, 1.0],
            [0.8, 0.15, 0.7],
            [0.9, -0.1, -0.25],
        ],
        dtype=np.float64,
    )

    distorted = _distort_opencv_xy(_stereographic_xy(rays), coeffs)
    expected = np.column_stack(
        [
            model.fx * distorted[:, 0] + model.cx,
            model.fy * distorted[:, 1] + model.cy,
        ]
    )

    np.testing.assert_allclose(model.project_points(rays), expected, atol=1e-12)


def test_stereographic_opencv_roundtrip_with_nonzero_distortion() -> None:
    """Distorted stereographic projection normalizes back to unit rays."""
    coeffs = np.zeros(14, dtype=np.float64)
    coeffs[[K1, K2, P1, P2, K3]] = [0.02, -0.001, 0.0005, -0.0003, 0.0001]
    model = _make_stereographic_opencv(coeffs)
    rays = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.3, -0.2, 1.0],
            [0.8, 0.15, 0.7],
            [0.9, -0.1, -0.25],
        ],
        dtype=np.float64,
    )

    pixels = model.project_points(rays)
    actual = model.normalize_points(pixels)
    expected = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_stereographic_opencv_intrinsics_repr_json_and_padding() -> None:
    """Public model helpers expose stable values and names."""
    model = lb.StereographicOpenCV(
        image_width=640,
        image_height=480,
        fx=300.0,
        fy=310.0,
        cx=321.0,
        cy=239.0,
        distortion_coeffs=np.array([0.01, -0.001]),
    )

    np.testing.assert_array_equal(
        model.K(),
        np.array([[300.0, 0.0, 321.0], [0.0, 310.0, 239.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        model.distortion_coeffs,
        np.array([0.01, -0.001, *([0.0] * 12)]),
    )
    assert "StereographicOpenCV" in repr(model)
    assert "Kannala" not in repr(model)
    assert model.to_json()["type"] == "stereographic_opencv"
    np.testing.assert_array_equal(
        lb.StereographicOpenCV.from_json(model.to_json()).K(),
        model.K(),
    )


def test_stereographic_opencv_rejects_too_many_coefficients() -> None:
    """The OpenCV stereographic model has exactly the OpenCV 14-coeff surface."""
    with pytest.raises(AssertionError, match="at most 14"):
        lb.StereographicOpenCV(
            image_width=640,
            image_height=480,
            fx=300.0,
            fy=310.0,
            cx=321.0,
            cy=239.0,
            distortion_coeffs=np.zeros(15),
        )


def test_stereographic_splined_intrinsics_repr_and_json() -> None:
    """Spline model helpers expose stable values and serialization contracts."""
    model = _make_stereographic_splined()

    np.testing.assert_array_equal(
        model.K(),
        np.array([[300.0, 0.0, 321.0], [0.0, 310.0, 239.0], [0.0, 0.0, 1.0]]),
    )
    assert "StereographicSplined" in repr(model)
    assert "knots=6x5" in repr(model)
    data = model.to_json()
    assert data["type"] == "stereographic_splined"
    loaded = lb.StereographicSplined.from_json(data)
    np.testing.assert_array_equal(loaded.dx_grid, model.dx_grid)
    np.testing.assert_array_equal(loaded.dy_grid, model.dy_grid)
    assert loaded.fov_deg_x == model.fov_deg_x
    assert loaded.fov_deg_y == model.fov_deg_y


def test_stereographic_splined_rejects_bad_grid_arrays() -> None:
    """Spline grids must be 2D floating arrays before they cross into C++."""
    with pytest.raises(AssertionError, match="Expected 2D dx_grid"):
        lb.StereographicSplined(
            image_width=640,
            image_height=480,
            fx=300.0,
            fy=310.0,
            cx=321.0,
            cy=239.0,
            dx_grid=np.zeros(5, dtype=np.float64),
            dy_grid=np.zeros((5, 6), dtype=np.float64),
            num_knots_x=6,
            num_knots_y=5,
            fov_deg_x=170.0,
            fov_deg_y=140.0,
        )

    with pytest.raises(AssertionError, match="floating dtype for dy_grid"):
        lb.StereographicSplined(
            image_width=640,
            image_height=480,
            fx=300.0,
            fy=310.0,
            cx=321.0,
            cy=239.0,
            dx_grid=np.zeros((5, 6), dtype=np.float64),
            dy_grid=np.zeros((5, 6), dtype=np.int32),
            num_knots_x=6,
            num_knots_y=5,
            fov_deg_x=170.0,
            fov_deg_y=140.0,
        )


@pytest.mark.parametrize(
    "model",
    [
        _make_stereographic_opencv(
            np.array([0.01, -0.001, 0.0002, -0.0003, 0.00005])
        ),
        _make_stereographic_splined(),
    ],
)
def test_stereographic_models_get_unproject_lut_matches_direct_normalize(
    model: lb.CameraModel,
) -> None:
    """LUT generation works for unit-bearing stereographic models."""
    lut = model.get_unproject_lut(grid_size_wh=(17, 13))
    x_coords = np.linspace(0.0, model.image_width - 1, lut.grid_width)
    y_coords = np.linspace(0.0, model.image_height - 1, lut.grid_height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    actual, valid = lut.normalize_points(pixels, interpolation="bilinear")
    expected = model.normalize_points(pixels)

    assert np.all(valid)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
