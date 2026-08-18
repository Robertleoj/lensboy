"""Integration tests using a real charuco dataset and synthetic data."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import lensboy as lb
from lensboy.geometry.pose import Pose

DATASET_PATH = Path(__file__).parent.parent / "data/test_datasets/wide_angle_charuco.npz"
CUBISM_DATASET_PATH = Path(__file__).parent.parent / "data/test_datasets/cubism.json"
WIDE_ANGLE_DATASET_PATH = (
    Path(__file__).parent.parent / "data/test_datasets/wide_angle_test.json"
)


def load_test_dataset() -> tuple[np.ndarray, list[lb.Frame], int, int]:
    """Load the pre-extracted charuco test dataset.

    Returns:
        target_points: 3D target coordinates, shape (N, 3).
        frames: Per-image detection frames.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
    """
    data = np.load(DATASET_PATH)
    target_points = data["target_points"]
    image_height = int(data["image_height"])
    image_width = int(data["image_width"])
    num_frames = int(data["num_frames"])

    frames = [
        lb.Frame(
            target_point_indices=data[f"frame_{i}_indices"],
            detected_points_in_image=data[f"frame_{i}_detections"],
        )
        for i in range(num_frames)
    ]

    return target_points, frames, image_height, image_width


def load_cubism_dataset() -> tuple[np.ndarray, list[lb.Frame], int, int]:
    """Load the cubism JSON test dataset.

    Returns:
        target_points: 3D target coordinates, shape (N, 3).
        frames: Per-image detection frames.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
    """
    data = json.loads(CUBISM_DATASET_PATH.read_text())
    target_points = np.asarray(data["target_points"], dtype=np.float64)

    image_size = data["image_size"]
    image_height = int(image_size["height"])
    image_width = int(image_size["width"])

    frames = [
        lb.Frame(
            target_point_indices=np.asarray(
                detection["target_point_ids"], dtype=np.int32
            ),
            detected_points_in_image=np.asarray(detection["pixels"], dtype=np.float64),
        )
        for detection in data["detections"]
    ]

    return target_points, frames, image_height, image_width


def load_wide_angle_dataset() -> tuple[np.ndarray, list[lb.Frame], int, int]:
    """Load the wide-angle JSON dataset.

    Returns:
        target_points: 3D target coordinates, shape (N, 3).
        frames: Per-image detection frames.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
    """
    data = json.loads(WIDE_ANGLE_DATASET_PATH.read_text())
    image_width = int(data["imageDimensions"]["width"])
    image_height = int(data["imageDimensions"]["height"])

    id_to_index: dict[str, int] = {}
    target_points = []
    for idx, point in enumerate(data["targetPoints"]):
        id_to_index[point["id"]] = idx
        position = point["positionMm"]
        target_points.append([position["x"], position["y"], position["z"]])

    frames = []
    for sample in data["samples"]:
        indices = []
        pixels = []
        for detection in sample["detections"]:
            point_idx = id_to_index.get(detection["id"])
            if point_idx is None:
                continue
            pixel = detection["pixel"]
            indices.append(point_idx)
            pixels.append([pixel["x"], pixel["y"]])
        frames.append(
            lb.Frame(
                target_point_indices=np.asarray(indices, dtype=np.int32),
                detected_points_in_image=np.asarray(pixels, dtype=np.float64),
            )
        )

    return np.asarray(target_points, dtype=np.float64), frames, image_height, image_width


def test_opencv_full14() -> None:
    """Calibrate an OpenCV model with all 14 distortion coefficients."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = (result.num_outliers() / result.num_detections()) * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.2, f"Too many outliers: {outlier_pct:.1f}%"

    _check_frame_projections(result, target_points, frames)


def test_cubism_opencv_full14() -> None:
    """Calibrate an OpenCV model from the cubism dataset."""
    target_points, frames, img_h, img_w = load_cubism_dataset()

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    _check_cubism_calibration_quality(result, max_sigma=0.25, max_outlier_pct=1.0)
    _check_frame_projections(result, target_points, frames)


def test_opencv_full14_explicit_focal_length() -> None:
    """Calibrate with an explicit initial focal length guess."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        initial_focal_length=1000,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = (result.num_outliers() / result.num_detections()) * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.2, f"Too many outliers: {outlier_pct:.1f}%"


def test_opencv_calibrates_from_initial_model() -> None:
    """Calibrate an OpenCV model starting from a previous calibrated model."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    baseline = lb.calibrate_camera(target_points, frames, camera_model_config=config)
    initial_model = replace(
        baseline.camera_model,
        fx=baseline.camera_model.fx * 1.05,
        fy=baseline.camera_model.fy * 0.95,
        cx=baseline.camera_model.cx + 8.0,
        cy=baseline.camera_model.cy - 6.0,
        distortion_coeffs=baseline.camera_model.distortion_coeffs
        + np.array(
            [
                0.02,
                -0.01,
                0.001,
                -0.001,
                0.005,
                -0.002,
                0.001,
                -0.001,
                0.0005,
                -0.0005,
                0.0005,
                -0.0005,
                0.0002,
                -0.0002,
            ],
            dtype=np.float64,
        ),
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
        initial_camera_model=initial_model,
    )

    sigma = result.residual_sigma_map()
    baseline_sigma = baseline.residual_sigma_map()
    outlier_pct = (result.num_outliers() / result.num_detections()) * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert sigma <= baseline_sigma + 0.01
    assert outlier_pct < 1.2, f"Too many outliers: {outlier_pct:.1f}%"


def test_spline_30x20() -> None:
    """Calibrate a 30x20 spline model."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.PinholeSplinedConfig(
        img_h,
        img_w,
        num_knots_x=30,
        num_knots_y=20,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100

    assert sigma < 0.09, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.6, f"Too many outliers: {outlier_pct:.1f}%"

    _check_frame_projections(result, target_points, frames)


def test_cubism_spline_30x20() -> None:
    """Calibrate a 30x20 spline model from the cubism dataset."""
    target_points, frames, img_h, img_w = load_cubism_dataset()

    config = lb.PinholeSplinedConfig(
        img_h,
        img_w,
        num_knots_x=30,
        num_knots_y=20,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    _check_cubism_calibration_quality(result, max_sigma=0.25, max_outlier_pct=1.0)
    _check_frame_projections(result, target_points, frames)


def test_spline_calibrates_from_initial_model() -> None:
    """Calibrate a spline model starting from a previous calibrated model."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.PinholeSplinedConfig(
        img_h,
        img_w,
        num_knots_x=30,
        num_knots_y=20,
    )
    baseline = lb.calibrate_camera(target_points, frames, camera_model_config=config)
    initial_model = replace(
        baseline.camera_model,
        dx_grid=baseline.camera_model.dx_grid
        + np.linspace(
            -0.001,
            0.001,
            baseline.camera_model.dx_grid.size,
            dtype=np.float64,
        ).reshape(baseline.camera_model.dx_grid.shape),
        dy_grid=baseline.camera_model.dy_grid
        + np.linspace(
            0.001,
            -0.001,
            baseline.camera_model.dy_grid.size,
            dtype=np.float64,
        ).reshape(baseline.camera_model.dy_grid.shape),
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
        initial_camera_model=initial_model,
    )

    sigma = result.residual_sigma_map()
    baseline_sigma = baseline.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100

    assert sigma < 0.09, f"Residual sigma too high: {sigma:.3f}px"
    assert sigma <= baseline_sigma + 0.01
    assert outlier_pct < 1.6, f"Too many outliers: {outlier_pct:.1f}%"


def test_opencv_all_outliers_in_one_frame() -> None:
    """Calibration succeeds when one frame has all its points corrupted."""
    target_points, frames, img_h, img_w = load_test_dataset()

    # Corrupt frame 0 by shifting every detection by a large random offset
    rng = np.random.default_rng(42)
    n_corrupted = len(frames[0])
    r = rng.uniform(25, 40, size=n_corrupted)
    theta = rng.uniform(0, 2 * np.pi, size=n_corrupted)
    offsets = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    corrupted_frame = lb.Frame(
        target_point_indices=frames[0].target_point_indices,
        detected_points_in_image=frames[0].detected_points_in_image + offsets,
    )
    frames_with_corruption = [corrupted_frame] + frames[1:]

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(
        target_points, frames_with_corruption, camera_model_config=config
    )

    # The corrupted frame should be fully rejected (no valid pose/diagnostics)
    assert result.frame_diagnostics[0] is None, (
        "Expected corrupted frame to be fully rejected"
    )
    assert result.cameras_from_target[0] is None, (
        "Expected corrupted frame to have no pose"
    )

    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100
    extra_outlier_pct = n_corrupted / result.num_detections() * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.2 + extra_outlier_pct, f"Too many outliers: {outlier_pct:.1f}%"


def test_opencv_distortion_mask() -> None:
    """Unselected distortion coefficients remain zero after calibration."""
    target_points, frames, img_h, img_w = load_test_dataset()

    masks = {
        "NONE": lb.OpenCVConfig.NONE,
        "STANDARD": lb.OpenCVConfig.STANDARD,
        "RADIAL_6": lb.OpenCVConfig.RADIAL_6,
        "TANGENTIAL": lb.OpenCVConfig.TANGENTIAL,
        "THIN_PRISM": lb.OpenCVConfig.THIN_PRISM,
    }

    for name, mask in masks.items():
        config = lb.OpenCVConfig(
            image_height=img_h,
            image_width=img_w,
            included_distortion_coefficients=mask,
        )
        result = lb.calibrate_camera(target_points, frames, camera_model_config=config)
        coeffs = result.camera_model.distortion_coeffs

        disabled = ~mask
        assert np.all(coeffs[disabled] == 0), (
            f"{name}: expected zeros at disabled indices {np.where(disabled)[0]}, "
            f"got {coeffs[disabled]}"
        )


def test_initial_opencv_model_must_match_config() -> None:
    """Reject an initial OpenCV model with parameters outside the config."""
    target_points = np.zeros((4, 3), dtype=float)
    config = lb.OpenCVConfig(
        image_height=480,
        image_width=640,
        included_distortion_coefficients=lb.OpenCVConfig.NONE,
    )
    initial_model = lb.OpenCV(
        image_height=480,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        distortion_coeffs=np.array([0.1]),
    )

    with pytest.raises(ValueError, match="disabled by the config"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=config,
            initial_camera_model=initial_model,
        )


def test_initial_spline_model_must_match_config() -> None:
    """Reject an initial spline model that the config could not produce."""
    target_points = np.zeros((4, 3), dtype=float)
    config = lb.PinholeSplinedConfig(
        image_height=480,
        image_width=640,
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_xy=(90.0, 70.0),
    )
    initial_model = lb.PinholeSplined(
        image_height=480,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        dx_grid=np.zeros((8, 12), dtype=float),
        dy_grid=np.zeros((8, 12), dtype=float),
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=91.0,
        fov_deg_y=70.0,
    )

    with pytest.raises(ValueError, match="fov_deg_x"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=config,
            initial_camera_model=initial_model,
        )


def test_synthetic_stereographic_opencv() -> None:
    """Calibrate a stereographic OpenCV model from synthetic observations."""
    rng = np.random.default_rng(123)
    ground_truth = lb.StereographicOpenCV(
        image_width=640,
        image_height=480,
        fx=320.0,
        fy=318.0,
        cx=321.0,
        cy=239.0,
        distortion_coeffs=np.array([0.02, -0.001, 0.0001, -0.0002, 0.00005]),
    )
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=50)

    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    config = lb.StereographicOpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=lb.StereographicOpenCVConfig.STANDARD,
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
        estimate_target_warp=False,
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.15, f"Residual sigma too high: {sigma:.3f}px"
    _check_frame_projections(result, target_points, frames)


def test_synthetic_stereographic_splined() -> None:
    """Calibrate a stereographic spline model from synthetic observations."""
    rng = np.random.default_rng(321)
    ground_truth = lb.StereographicSplined(
        image_width=640,
        image_height=480,
        fx=320.0,
        fy=318.0,
        cx=321.0,
        cy=239.0,
        dx_grid=np.zeros((8, 12), dtype=np.float64),
        dy_grid=np.zeros((8, 12), dtype=np.float64),
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=150.0,
        fov_deg_y=120.0,
    )
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=50)

    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    config = lb.StereographicSplinedConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_xy=(150.0, 120.0),
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
        estimate_target_warp=False,
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.15, f"Residual sigma too high: {sigma:.3f}px"
    _check_frame_projections(result, target_points, frames)


def test_wide_angle_spline_models_have_matching_ray_fields() -> None:
    """Stereographic and pinhole splines agree on the supported wide-angle field."""
    from lensboy.analysis.differencing import compute_projection_diff

    target_points, frames, img_h, img_w = load_wide_angle_dataset()
    pinhole_result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=lb.PinholeSplinedConfig(
            image_height=img_h,
            image_width=img_w,
            num_knots_x=24,
            num_knots_y=16,
        ),
    )
    stereographic_result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=lb.StereographicSplinedConfig(
            image_height=img_h,
            image_width=img_w,
            num_knots_x=24,
            num_knots_y=16,
        ),
    )

    pixels, _, diff, _, _ = compute_projection_diff(
        pinhole_result.camera_model,
        stereographic_result.camera_model,
        radius=826.0,
        grid_density=120,
    )
    center = np.array([(img_w - 1) / 2.0, (img_h - 1) / 2.0])
    radii = np.linalg.norm(pixels - center, axis=1)
    diff_norm = np.linalg.norm(diff, axis=1)
    supported = (radii <= 826.0) & np.isfinite(diff_norm)

    assert np.median(diff_norm[supported]) < 0.02
    assert np.percentile(diff_norm[supported], 95) < 0.04


def test_initial_stereographic_models_must_match_config() -> None:
    """Reject initial stereographic models that the configs could not produce."""
    target_points = np.zeros((4, 3), dtype=float)
    opencv_config = lb.StereographicOpenCVConfig(
        image_height=480,
        image_width=640,
        included_distortion_coefficients=lb.StereographicOpenCVConfig.STANDARD,
    )
    disabled_coeffs = np.zeros(14, dtype=np.float64)
    disabled_coeffs[5] = 0.01
    opencv_initial = lb.StereographicOpenCV(
        image_height=480,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        distortion_coeffs=disabled_coeffs,
    )
    with pytest.raises(ValueError, match="disabled"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=opencv_config,
            initial_camera_model=opencv_initial,
        )

    opencv_wrong_size = lb.StereographicOpenCV(
        image_height=481,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        distortion_coeffs=np.zeros(14, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="image_height"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=opencv_config,
            initial_camera_model=opencv_wrong_size,
        )

    spline_config = lb.StereographicSplinedConfig(
        image_height=480,
        image_width=640,
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_xy=(120.0, 100.0),
    )
    spline_initial = lb.StereographicSplined(
        image_height=480,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        dx_grid=np.zeros((8, 12), dtype=np.float64),
        dy_grid=np.zeros((8, 12), dtype=np.float64),
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=121.0,
        fov_deg_y=100.0,
    )
    with pytest.raises(ValueError, match="fov_deg_x"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=spline_config,
            initial_camera_model=spline_initial,
        )

    spline_wrong_size = lb.StereographicSplined(
        image_height=481,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        dx_grid=np.zeros((8, 12), dtype=np.float64),
        dy_grid=np.zeros((8, 12), dtype=np.float64),
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=120.0,
        fov_deg_y=100.0,
    )
    with pytest.raises(ValueError, match="image_height"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=spline_config,
            initial_camera_model=spline_wrong_size,
        )

    spline_wrong_knots = lb.StereographicSplined(
        image_height=480,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        dx_grid=np.zeros((8, 13), dtype=np.float64),
        dy_grid=np.zeros((8, 13), dtype=np.float64),
        num_knots_x=13,
        num_knots_y=8,
        fov_deg_x=120.0,
        fov_deg_y=100.0,
    )
    with pytest.raises(ValueError, match="num_knots_x"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=spline_config,
            initial_camera_model=spline_wrong_knots,
        )

    spline_wrong_grid_shape = lb.StereographicSplined(
        image_height=480,
        image_width=640,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        dx_grid=np.zeros((7, 12), dtype=np.float64),
        dy_grid=np.zeros((7, 12), dtype=np.float64),
        num_knots_x=12,
        num_knots_y=8,
        fov_deg_x=120.0,
        fov_deg_y=100.0,
    )
    with pytest.raises(ValueError, match="dx_grid"):
        lb.calibrate_camera(
            target_points,
            [],
            camera_model_config=spline_config,
            initial_camera_model=spline_wrong_grid_shape,
        )


def _check_frame_projections(
    result: lb.CalibrationResult,
    target_points: np.ndarray,
    frames: list[lb.Frame],
) -> None:
    model = result.camera_model

    for i, frame in enumerate(frames):
        pose = result.cameras_from_target[i]
        fi = result.frame_diagnostics[i]
        if pose is None or fi is None:
            continue

        points_in_target = target_points[frame.target_point_indices]
        if result.target_warp is not None:
            points_in_target = result.target_warp.warp_target(points_in_target)

        points_in_cam = pose.apply(points_in_target)
        projected = model.project_points(points_in_cam)

        np.testing.assert_allclose(
            projected, fi.projected_points, atol=1e-6, err_msg=f"Frame {i}"
        )

        expected_residuals = projected - frame.detected_points_in_image
        np.testing.assert_allclose(
            expected_residuals, fi.residuals, atol=1e-6, err_msg=f"Frame {i}"
        )


def _check_cubism_calibration_quality(
    result: lb.CalibrationResult,
    *,
    max_sigma: float,
    max_outlier_pct: float,
) -> None:
    solved_frames = sum(
        frame_diagnostics is not None for frame_diagnostics in result.frame_diagnostics
    )
    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100

    assert solved_frames == len(result.frames), (
        f"Expected all cubism frames to solve, got {solved_frames}/{len(result.frames)}"
    )
    assert sigma < max_sigma, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < max_outlier_pct, f"Too many outliers: {outlier_pct:.1f}%"


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_planar_grid(cols: int = 12, rows: int = 9, spacing: float = 30.0) -> np.ndarray:
    """Create a planar calibration grid centred at the origin.

    Returns:
        Grid points, shape (cols * rows, 3) with z=0.
    """
    xs = np.arange(cols) * spacing - (cols - 1) * spacing / 2
    ys = np.arange(rows) * spacing - (rows - 1) * spacing / 2
    gx, gy = np.meshgrid(xs, ys)
    gz = np.zeros_like(gx)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _generate_synthetic_frames(
    rng: np.random.Generator,
    model: lb.OpenCV,
    target_points: np.ndarray,
    num_frames: int = 40,
    noise_sigma: float = 0.1,
    margin: int = 20,
) -> list[lb.Frame]:
    """Project a target grid through a known camera model to create frames.

    Args:
        rng: Numpy random generator.
        model: Ground-truth camera model.
        target_points: 3D target coordinates, shape (N, 3).
        num_frames: Number of synthetic views to generate.
        noise_sigma: Gaussian noise added to pixel detections in pixels.
        margin: Minimum distance from image border for accepted detections in pixels.

    Returns:
        Synthetic detection frames.
    """
    all_indices = np.arange(len(target_points))
    frames: list[lb.Frame] = []

    # Scale camera distances to the target's spatial extent
    centroid = target_points.mean(axis=0)
    target_radius = np.linalg.norm(target_points - centroid, axis=1).max()
    base_dist = max(
        target_radius * model.fx / (min(model.image_width, model.image_height) / 2), 100.0
    )

    for _ in range(num_frames):
        rotvec = rng.normal(scale=0.2, size=3)
        tz = rng.uniform(base_dist * 0.8, base_dist * 2.5)
        tx = rng.normal(scale=base_dist * 0.1)
        ty = rng.normal(scale=base_dist * 0.1)
        pose = lb.Pose.from_rotvec_trans(rotvec=rotvec, trans=np.array([tx, ty, tz]))

        points_in_cam = pose.apply(target_points)
        projected = model.project_points(points_in_cam)

        in_bounds = (
            (projected[:, 0] >= margin)
            & (projected[:, 0] < model.image_width - margin)
            & (projected[:, 1] >= margin)
            & (projected[:, 1] < model.image_height - margin)
            & (points_in_cam[:, 2] > 0)
        )
        if in_bounds.sum() < 10:
            continue

        noise = rng.normal(scale=noise_sigma, size=(in_bounds.sum(), 2))
        detected = projected[in_bounds] + noise
        frames.append(
            lb.Frame(
                target_point_indices=all_indices[in_bounds],
                detected_points_in_image=detected,
            )
        )

    return frames


# ---------------------------------------------------------------------------
# Synthetic focal-length estimation stress tests
# ---------------------------------------------------------------------------

_SYNTHETIC_MODELS = [
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=500.0,
            fy=500.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([-0.3, 0.1, 0.0, 0.0, 0.0]),
        ),
        id="wide_fov_1920x1080_f500",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=1800.0,
            fy=1800.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([0.1, -0.05, 0.001, -0.001, 0.0]),
        ),
        id="narrow_fov_1920x1080_f1800",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=4000,
            image_height=3000,
            fx=3500.0,
            fy=3500.0,
            cx=2000.0,
            cy=1500.0,
            distortion_coeffs=np.array([0.05, 0.01, 0.0, 0.0, -0.002]),
        ),
        id="telephoto_4000x3000_f3500",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=640,
            image_height=480,
            fx=300.0,
            fy=300.0,
            cx=320.0,
            cy=240.0,
            distortion_coeffs=np.array([-0.4, 0.15, 0.0, 0.0, -0.02]),
        ),
        id="low_res_fisheye_640x480_f300",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=3088,
            image_height=2064,
            fx=1350.0,
            fy=1350.0,
            cx=1544.0,
            cy=1032.0,
            distortion_coeffs=np.array(
                [
                    1.5,
                    0.4,
                    -0.0001,
                    0.0,
                    0.008,
                    1.8,
                    0.78,
                    0.06,
                    0.0,
                    0.0,
                    0.0002,
                    0.0,
                    0.0005,
                    -0.0003,
                ]
            ),
        ),
        id="full14_distortion_3088x2064_f1350",
    ),
]


@pytest.mark.parametrize("ground_truth", _SYNTHETIC_MODELS)
def test_synthetic_auto_focal_length(ground_truth: lb.OpenCV) -> None:
    """Calibrate synthetic data without an initial focal length guess."""
    rng = np.random.default_rng(123)
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=80)

    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=lb.OpenCVConfig.STANDARD,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    print(f"focal error pct {f_err_pct}")
    assert f_err_pct < 0.05, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"


# ---------------------------------------------------------------------------
# Exotic target generators
# ---------------------------------------------------------------------------


def _apply_random_similarity_transform(
    rng: np.random.Generator, points: np.ndarray
) -> np.ndarray:
    """Apply a random rotation, translation, and uniform scale to target points.

    Args:
        rng: Numpy random generator.
        points: 3D points, shape (N, 3).

    Returns:
        Transformed points, shape (N, 3).
    """
    scale = rng.uniform(0.01, 1000.0)
    rotvec = rng.normal(scale=3, size=3)
    # Keep translation modest so the target stays visible from the synthetic cameras
    trans = rng.normal(scale=30.0, size=3)
    pose = lb.Pose.from_rotvec_trans(rotvec=rotvec, trans=trans)
    return pose.apply(points * scale)


def _make_random_planar(
    rng: np.random.Generator, n_points: int = 100, extent: float = 150.0
) -> np.ndarray:
    """Create randomly scattered points on the z=0 plane.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        extent: Half-width of the square region.

    Returns:
        Points, shape (n_points, 3) with z=0.
    """
    xy = rng.uniform(-extent, extent, size=(n_points, 2))
    z = np.zeros((n_points, 1))
    return np.hstack([xy, z])


def _make_ball(
    rng: np.random.Generator, n_points: int = 100, radius: float = 80.0
) -> np.ndarray:
    """Create points uniformly distributed inside a 3D ball.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        radius: Ball radius.

    Returns:
        Points, shape (n_points, 3).
    """
    # Rejection sampling for uniform distribution in a ball
    points = []
    while len(points) < n_points:
        candidates = rng.uniform(-radius, radius, size=(n_points * 2, 3))
        inside = np.linalg.norm(candidates, axis=1) <= radius
        points.extend(candidates[inside].tolist())
    return np.array(points[:n_points])


def _make_two_intersecting_planes(
    rng: np.random.Generator,
    n_per_plane: int = 50,
    extent: float = 120.0,
    angle_deg: float = 30.0,
) -> np.ndarray:
    """Create points on two planes intersecting along the y-axis.

    Args:
        rng: Numpy random generator.
        n_per_plane: Number of points per plane.
        extent: Half-width of each plane.
        angle_deg: Half-angle between the two planes.

    Returns:
        Points, shape (2 * n_per_plane, 3).
    """
    angle = np.radians(angle_deg)

    # Plane 1: tilted by +angle around y-axis
    xy1 = rng.uniform(-extent, extent, size=(n_per_plane, 2))
    plane1 = np.column_stack(
        [xy1[:, 0] * np.cos(angle), xy1[:, 1], xy1[:, 0] * np.sin(angle)]
    )

    # Plane 2: tilted by -angle around y-axis
    xy2 = rng.uniform(-extent, extent, size=(n_per_plane, 2))
    plane2 = np.column_stack(
        [xy2[:, 0] * np.cos(-angle), xy2[:, 1], xy2[:, 0] * np.sin(-angle)]
    )

    return np.vstack([plane1, plane2])


def _make_hemisphere(
    rng: np.random.Generator, n_points: int = 100, radius: float = 100.0
) -> np.ndarray:
    """Create points on the surface of a hemisphere (z >= 0).

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        radius: Hemisphere radius.

    Returns:
        Points, shape (n_points, 3).
    """
    # Uniform sampling on a sphere via normal distribution, then take z >= 0
    points = []
    while len(points) < n_points:
        raw = rng.normal(size=(n_points * 3, 3))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        raw *= radius
        upper = raw[raw[:, 2] >= 0]
        points.extend(upper.tolist())
    return np.array(points[:n_points])


def _make_cylinder(
    rng: np.random.Generator,
    n_points: int = 100,
    radius: float = 60.0,
    height: float = 200.0,
) -> np.ndarray:
    """Create points on the surface of a cylinder aligned with the y-axis.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        radius: Cylinder radius.
        height: Cylinder height.

    Returns:
        Points, shape (n_points, 3).
    """
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    y = rng.uniform(-height / 2, height / 2, size=n_points)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def _make_random_3d_cluster(
    rng: np.random.Generator, n_points: int = 100, extent: float = 100.0
) -> np.ndarray:
    """Create randomly scattered points in a 3D box.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        extent: Half-width of the box along each axis.

    Returns:
        Points, shape (n_points, 3).
    """
    return rng.uniform(-extent, extent, size=(n_points, 3))


# ---------------------------------------------------------------------------
# Synthetic exotic-target tests
# ---------------------------------------------------------------------------

_EXOTIC_TARGET_MODELS = [
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([0.02, -0.005, 0.0, 0.0, 0.0]),
        ),
        id="minimal_distortion",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=800.0,
            fy=800.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([-0.3, 0.12, 0.001, -0.002, -0.04]),
        ),
        id="medium_distortion",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=3088,
            image_height=2064,
            fx=1354.5124985080904,
            fy=1354.3181984440832,
            cx=1514.104403863959,
            cy=1076.8896015546975,
            distortion_coeffs=np.array(
                [
                    1.721749851751697,
                    0.4929049527387092,
                    -0.00012249059620334055,
                    6.571195104303754e-05,
                    0.010826498817585985,
                    2.040924560626435,
                    0.9497700975338902,
                    0.0744348380132027,
                    -6.852182482012876e-05,
                    -8.155006688534965e-06,
                    0.00021009345380118726,
                    -4.392347849675705e-06,
                    0.0005388341393511124,
                    -0.0003861499673898091,
                ]
            ),
        ),
        id="extreme_distortion",
    ),
]

_EXOTIC_TARGETS = [
    pytest.param(_make_random_planar, id="random_planar"),
    pytest.param(_make_ball, id="ball"),
    pytest.param(_make_two_intersecting_planes, id="two_intersecting_planes"),
    pytest.param(_make_hemisphere, id="hemisphere"),
    pytest.param(_make_cylinder, id="cylinder"),
    pytest.param(_make_random_3d_cluster, id="random_3d_cluster"),
]


@pytest.mark.parametrize("ground_truth", _EXOTIC_TARGET_MODELS)
@pytest.mark.parametrize("make_target", _EXOTIC_TARGETS)
def test_synthetic_exotic_targets(
    ground_truth: lb.OpenCV,
    make_target: Callable[[np.random.Generator], np.ndarray],
) -> None:
    """Calibrate with exotic (non-grid) calibration targets."""
    rng = np.random.default_rng(777)
    target_points = make_target(rng)
    target_points = _apply_random_similarity_transform(rng, target_points)

    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=40)
    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    # Fit the same distortion terms the ground truth uses
    dist_mask = ground_truth.distortion_coeffs != 0
    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=dist_mask,
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
    )

    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    print(f"focal error pct: {f_err_pct}")
    assert f_err_pct < 0.05, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    print(f"sigma: {sigma}")
    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"

    _check_frame_projections(result, target_points, frames)


def test_pnp_failure() -> None:
    """Verify that frames failing initial PnP are recovered after optimization.

    Uses a wrong initial focal length so PnP produces bad poses initially.
    After the first optimization refines intrinsics, the retry should recover them.
    """
    ground_truth = lb.OpenCV(
        image_width=3088,
        image_height=2064,
        fx=1354.5124985080904,
        fy=1354.3181984440832,
        cx=1514.104403863959,
        cy=1076.8896015546975,
        distortion_coeffs=np.array(
            [
                1.721749851751697,
                0.4929049527387092,
                -0.00012249059620334055,
                6.571195104303754e-05,
                0.010826498817585985,
                2.040924560626435,
                0.9497700975338902,
                0.0744348380132027,
                -6.852182482012876e-05,
                -8.155006688534965e-06,
                0.00021009345380118726,
                -4.392347849675705e-06,
                0.0005388341393511124,
                -0.0003861499673898091,
            ]
        ),
    )

    rng = np.random.default_rng(999)
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=60)
    assert len(frames) >= 20, f"Too few valid frames ({len(frames)})"

    # Inject some frames with only 3 detections — too few for PnP (needs >= 4).
    # These should fail initially but be recovered after the first optimization
    # refines intrinsics and retries with distortion-aware PnP.
    sparse_frames = []
    for f in frames[:5]:
        sparse_frames.append(
            lb.Frame(
                target_point_indices=f.target_point_indices[:3],
                detected_points_in_image=f.detected_points_in_image[:3],
            )
        )
    frames_with_sparse = sparse_frames + frames

    dist_mask = ground_truth.distortion_coeffs != 0
    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=dist_mask,
    )
    result = lb.calibrate_camera(
        target_points,
        frames_with_sparse,
        camera_model_config=config,
    )

    # Output lists must match input length
    n_total = len(frames_with_sparse)
    assert len(result.cameras_from_target) == n_total
    assert len(result.frame_diagnostics) == n_total
    assert len(result.frames) == n_total

    # The first 5 frames (sparse, <4 points) should have None pose/diagnostics
    for i in range(5):
        assert result.cameras_from_target[i] is None, (
            f"Expected None pose for sparse frame {i}"
        )
        assert result.frame_diagnostics[i] is None, (
            f"Expected None diagnostics for sparse frame {i}"
        )

    # The remaining frames (full detections) should all have valid pose/diagnostics
    for i in range(5, n_total):
        assert result.cameras_from_target[i] is not None, (
            f"Expected valid pose for frame {i}"
        )
        assert result.frame_diagnostics[i] is not None, (
            f"Expected valid diagnostics for frame {i}"
        )

    # Should still converge — sparse frames are just ignored
    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    assert f_err_pct < 0.05, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"


def _generate_edge_frames(
    rng: np.random.Generator,
    model: lb.OpenCV,
    target_points: np.ndarray,
    target_width: float,
    target_height: float,
    samples_per_edge: int = 20,
    center_samples: int = 0,
    noise_sigma: float = 0.1,
) -> list[lb.Frame]:
    """Generate synthetic frames with observations at the absolute image edges.

    Positions cameras so that a specific edge of the target grid aligns with
    each image border, with random tilt rotations to compress points toward
    the border.

    Args:
        rng: Numpy random generator.
        model: Ground-truth camera model.
        target_points: 3D target coordinates, shape (N, 3).
        target_width: Physical width of the target grid.
        target_height: Physical height of the target grid.
        samples_per_edge: Number of camera poses per image edge.
        noise_sigma: Gaussian noise added to pixel detections in pixels.

    Returns:
        Synthetic detection frames with observations near image borders.
    """
    all_indices = np.arange(len(target_points))

    half_fov_x = np.radians(model.fov_deg_x) / 2
    half_fov_y = np.radians(model.fov_deg_y) / 2

    u_max = np.tan(half_fov_x)
    u_min = -u_max
    v_max = np.tan(half_fov_y)
    v_min = -v_max

    upper_right = np.array([u_max, v_min, 1.0])
    lower_right = np.array([u_max, v_max, 1.0])
    upper_left = np.array([u_min, v_min, 1.0])
    lower_left = np.array([u_min, v_max, 1.0])

    rot_bound_lower = 60
    rot_bound_higher = 100
    dist_bound_lower = 100.0
    dist_bound_higher = 500.0

    poses: list[Pose] = []

    # right edge (midpoint of right edge of grid centered at origin)
    target_from_edge = Pose.identity().tx(target_width / 2)
    for edge_interp, dist, rot in zip(
        rng.uniform(0, 1, samples_per_edge),
        rng.uniform(dist_bound_lower, dist_bound_higher, samples_per_edge),
        np.radians(rng.uniform(-rot_bound_higher, rot_bound_lower, samples_per_edge)),
    ):
        ray = upper_right * edge_interp + lower_right * (1 - edge_interp)
        camera_from_edge = Pose.from_trans(dist * ray).rx(np.pi).ry(rot)
        poses.append(camera_from_edge @ target_from_edge.inverse())

    # left edge
    target_from_edge = Pose.identity().tx(-target_width / 2)
    for edge_interp, dist, rot in zip(
        rng.uniform(0, 1, samples_per_edge),
        rng.uniform(dist_bound_lower, dist_bound_higher, samples_per_edge),
        np.radians(rng.uniform(rot_bound_lower, rot_bound_higher, samples_per_edge)),
    ):
        ray = upper_left * edge_interp + lower_left * (1 - edge_interp)
        camera_from_edge = Pose.from_trans(dist * ray).rx(np.pi).ry(rot)
        poses.append(camera_from_edge @ target_from_edge.inverse())

    # top edge
    target_from_edge = Pose.identity().ty(target_height / 2)
    for edge_interp, dist, rot in zip(
        rng.uniform(0, 1, samples_per_edge),
        rng.uniform(dist_bound_lower, dist_bound_higher, samples_per_edge),
        np.radians(rng.uniform(rot_bound_lower, rot_bound_higher, samples_per_edge)),
    ):
        ray = upper_left * edge_interp + upper_right * (1 - edge_interp)
        camera_from_edge = Pose.from_trans(dist * ray).rx(np.pi).rx(rot)
        poses.append(camera_from_edge @ target_from_edge.inverse())

    # bottom edge
    target_from_edge = Pose.identity().ty(-target_height / 2)
    for edge_interp, dist, rot in zip(
        rng.uniform(0, 1, samples_per_edge),
        rng.uniform(dist_bound_lower, dist_bound_higher, samples_per_edge),
        np.radians(rng.uniform(-rot_bound_higher, rot_bound_lower, samples_per_edge)),
    ):
        ray = lower_left * edge_interp + lower_right * (1 - edge_interp)
        camera_from_edge = Pose.from_trans(dist * ray).rx(np.pi).rx(rot)
        poses.append(camera_from_edge @ target_from_edge.inverse())

    # Center-coverage frames: target roughly centered with small perturbations
    center_dist = max(
        target_width / (2 * np.tan(half_fov_x)),
        target_height / (2 * np.tan(half_fov_y)),
    )
    for _ in range(center_samples):
        rotvec = rng.normal(scale=0.15, size=3)
        tz = rng.uniform(center_dist * 0.8, center_dist * 1.5)
        tx = rng.normal(scale=target_width * 0.05)
        ty = rng.normal(scale=target_height * 0.05)
        poses.append(Pose.from_rotvec_trans(rotvec=rotvec, trans=np.array([tx, ty, tz])))

    # Project poses into frames
    w, h = model.image_width, model.image_height
    frames: list[lb.Frame] = []
    for pose in poses:
        points_in_cam = pose.apply(target_points)
        if not (points_in_cam[:, 2] > 0).all():
            continue
        projected = model.project_points(points_in_cam)

        in_bounds = (
            (projected[:, 0] >= 0)
            & (projected[:, 0] < w)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < h)
        )
        if in_bounds.sum() < 6:
            continue

        noise = rng.normal(scale=noise_sigma, size=(in_bounds.sum(), 2))
        frames.append(
            lb.Frame(
                target_point_indices=all_indices[in_bounds],
                detected_points_in_image=projected[in_bounds] + noise,
            )
        )

    return frames


def test_spline_edge_observations() -> None:
    """Calibrate a spline model when many observations are at the absolute image edge.

    Uses the extreme distortion model and positions cameras using the model's FOV
    so that the dense target grid extends right to the image borders. The spline
    fine-tuner should handle edge observations gracefully.
    """
    ground_truth = lb.OpenCV(
        image_width=3088,
        image_height=2064,
        fx=1354.5124985080904,
        fy=1354.3181984440832,
        cx=1514.104403863959,
        cy=1076.8896015546975,
        distortion_coeffs=np.array(
            [
                1.721749851751697,
                0.4929049527387092,
                -0.00012249059620334055,
                6.571195104303754e-05,
                0.010826498817585985,
                2.040924560626435,
                0.9497700975338902,
                0.0744348380132027,
                -6.852182482012876e-05,
                -8.155006688534965e-06,
                0.00021009345380118726,
                -4.392347849675705e-06,
                0.0005388341393511124,
                -0.0003861499673898091,
            ]
        ),
    )

    rng = np.random.default_rng(42)
    target_points = _make_planar_grid(cols=30, rows=20, spacing=12.0)
    target_width = (30 - 1) * 12.0
    target_height = (20 - 1) * 12.0

    edge_frames = _generate_edge_frames(
        rng, ground_truth, target_points, target_width, target_height
    )
    center_frames = _generate_synthetic_frames(
        rng, ground_truth, target_points, num_frames=30, margin=0
    )

    frames = center_frames + edge_frames
    assert len(frames) >= 30, f"Too few valid frames ({len(frames)})"

    # Count how many detections are within 5px of the image border
    all_detections = np.vstack([f.detected_points_in_image for f in frames])
    near_edge = (
        (all_detections[:, 0] < 5)
        | (all_detections[:, 0] > ground_truth.image_width - 5)
        | (all_detections[:, 1] < 5)
        | (all_detections[:, 1] > ground_truth.image_height - 5)
    )
    print(
        f"Edge observations (<5px from border): {near_edge.sum()}/{len(all_detections)}"
    )

    config = lb.PinholeSplinedConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        num_knots_x=30,
        num_knots_y=20,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100

    print(f"sigma={sigma:.4f}px, outliers={outlier_pct:.1f}%")
    assert sigma < 0.15, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 5.0, f"Too many outliers: {outlier_pct:.1f}%"
