"""Focused tests for calibration orchestration helpers."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lensboy.calibration import calibrate
from lensboy.calibration.type_defs import Frame
from lensboy.camera_models.pinhole_splined import PinholeSplinedConfig
from lensboy.camera_models.stereographic_opencv import StereographicOpenCV


def test_automatic_spline_fov_uses_raw_stereographic_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic spline FOV comes from a distortion-free stereographic fit."""
    frames = [
        Frame(
            target_point_indices=np.array([0, 1, 2, 3]),
            detected_points_in_image=np.zeros((4, 2)),
        )
        for _ in range(3)
    ]
    target_points = np.zeros((4, 3))
    config = PinholeSplinedConfig(
        image_height=480,
        image_width=640,
        num_knots_x=10,
        num_knots_y=8,
        initial_focal_length=300.0,
    )
    fitted_model = StereographicOpenCV(
        image_height=480,
        image_width=640,
        fx=310.0,
        fy=305.0,
        cx=321.0,
        cy=239.0,
        distortion_coeffs=np.zeros(14),
    )
    fit_call: dict[str, Any] = {}

    def fake_calibrate(
        fit_target_points: np.ndarray,
        fit_frames: list[Frame],
        fit_config: object,
        outlier_threshold_stddevs: float | None,
        estimate_target_warp: bool,
    ) -> object:
        fit_call["target_points"] = fit_target_points
        fit_call["frames"] = fit_frames
        fit_call["config"] = fit_config
        fit_call["outlier_threshold_stddevs"] = outlier_threshold_stddevs
        fit_call["estimate_target_warp"] = estimate_target_warp
        return SimpleNamespace(camera_model=fitted_model)

    monkeypatch.setattr(calibrate, "_calibrate_stereographic_opencv", fake_calibrate)
    monkeypatch.setattr(
        calibrate,
        "_compute_spline_grid_fov_from_unit_ray_model",
        lambda model: (123.0, 98.0),
    )

    fov = calibrate._estimate_spline_fov(
        frames,
        [0, 2],
        target_points,
        config,
    )

    fit_config = fit_call["config"]
    assert fov == (123.0, 98.0)
    assert fit_call["target_points"] is target_points
    fitted_frames = fit_call["frames"]
    assert fitted_frames[0] is frames[0]
    assert fitted_frames[1] is frames[2]
    assert fit_call["outlier_threshold_stddevs"] is None
    assert fit_call["estimate_target_warp"] is False
    assert fit_config.initial_focal_length == 300.0
    assert not np.any(fit_config.included_distortion_coefficients)


def test_spline_grid_fov_uses_stereographic_axis_extents() -> None:
    """Spline-grid FOV avoids angular cross-coupling at image corners."""
    model = StereographicOpenCV(
        image_height=480,
        image_width=640,
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        distortion_coeffs=np.zeros(14),
    )

    fov_x, fov_y = calibrate._compute_spline_grid_fov_from_unit_ray_model(model)

    expected_x = np.degrees(4.0 * np.arctan(0.8 / 2.0))
    expected_y = np.degrees(4.0 * np.arctan(0.6 / 2.0))
    assert fov_x == pytest.approx(expected_x)
    assert fov_y == pytest.approx(expected_y)
