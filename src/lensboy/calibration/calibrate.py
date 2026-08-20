from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from timeit import default_timer
from typing import Generic, overload

import cv2
import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy._logging import disable_logs, enable_logs, log, warn
from lensboy.calibration.type_defs import (
    CalibrationResult,
    Frame,
    FrameDiagnostics,
    IntrinsicsT,
    TargetWarp,
    WarpCoordinates,
)
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig
from lensboy.camera_models.opencv import OpenCV, OpenCVConfig
from lensboy.camera_models.pinhole_splined import (
    PinholeSplined,
    PinholeSplinedConfig,
)
from lensboy.camera_models.stereographic_opencv import (
    StereographicOpenCV,
    StereographicOpenCVConfig,
)
from lensboy.camera_models.stereographic_splined import (
    StereographicSplined,
    StereographicSplinedConfig,
)
from lensboy.geometry.pose import Pose

DEFAULT_OUTLIER_THRESHOLD = 5.0
MAX_OUTLIER_FILTER_PASSES = 2
FOCAL_SWEEP_MAX_FRAMES = 20
SEED_FIT_MAX_FRAMES = 20


@dataclass
class _OptimizationBatch(Generic[IntrinsicsT]):
    """Compact, fully-valid data passed to the optimizer. No None values."""

    intrinsics: IntrinsicsT
    cameras_from_target: list[Pose]
    frames: list[Frame]
    warp_coeffs: tuple[float, float, float, float, float] | None


@dataclass
class _OptimizationState(Generic[IntrinsicsT]):
    intrinsics: IntrinsicsT
    cameras_from_target: list[Pose | None]
    frames: list[Frame]
    warp_coeffs: tuple[float, float, float, float, float] | None
    inlier_masks: list[np.ndarray | None]


@dataclass
class _PnPFrameData:
    object_points: np.ndarray
    image_points: np.ndarray


def _project_and_calculate_residuals(
    target_points: np.ndarray,
    camera_from_target: Pose,
    frame: Frame,
    model: OpenCV | PinholeSplined | StereographicOpenCV | StereographicSplined,
    target_warp: TargetWarp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    point_indices = frame.target_point_indices

    points_in_target = target_points[point_indices]
    if target_warp is not None:
        points_in_target = target_warp.warp_target(points_in_target)
    points_in_camera = camera_from_target.apply(points_in_target)

    projected_points_in_image = model.project_points(points_in_camera)

    residuals = projected_points_in_image - frame.detected_points_in_image

    return projected_points_in_image, residuals


def _mad_sigma_1d(x: np.ndarray) -> float:
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # 1.4826 = 1 / Phi^{-1}(0.75)  (MAD->sigma for 1D normal)
    return float(1.4826 * mad)


def _robust_sigma_xy(residuals: list[np.ndarray]) -> float:
    R = np.concatenate(residuals, axis=0)  # (M,2)
    sx = _mad_sigma_1d(R[:, 0])
    sy = _mad_sigma_1d(R[:, 1])
    # combine to one sigma (assume roughly same scale)
    return float(np.sqrt(0.5 * (sx * sx + sy * sy)))


def _filter_outliers(
    residuals: list[np.ndarray],
    k: float,
    sigma_floor_px: float = 0.05,  # prevents collapse
) -> list[np.ndarray]:
    """Compute per-frame inlier masks based on residual norms.

    Returns:
        List of boolean arrays, one per frame, True for inliers.
    """
    sigma = max(_robust_sigma_xy(residuals), sigma_floor_px)
    gate = k * sigma

    out = []
    for res in residuals:
        mask = np.linalg.norm(res, axis=1) <= gate
        out.append(mask)

    return out


def _apply_mask(frame: Frame, mask: np.ndarray | None) -> Frame:
    """Apply an inlier mask to a frame, keeping only inlier points."""
    if mask is None:
        return frame
    return Frame(
        target_point_indices=frame.target_point_indices[mask],
        detected_points_in_image=frame.detected_points_in_image[mask],
    )


def _opencv_calibrate_inner(
    batch: _OptimizationBatch[OpenCV],
    config: OpenCVConfig,
    target_points: np.ndarray,
    warp_coordinates: WarpCoordinates | None = None,
) -> _OptimizationBatch[OpenCV]:
    params = batch.intrinsics._params()
    mask = config.optimize_mask()
    intrinsics_param_optimize_mask = mask.tolist()
    warp_coordinates_cpp = None
    if warp_coordinates is not None:
        warp_coordinates_cpp = warp_coordinates._to_cpp()
    warp_coeffs_initial = [0.0] * 5
    if batch.warp_coeffs is not None:
        warp_coeffs_initial = list(batch.warp_coeffs)

    result = lbb.calibrate_opencv(
        intrinsics_initial_value=params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_target=[p._to_cpp() for p in batch.cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in batch.frames],
        warp_coordinates=warp_coordinates_cpp,
        warp_coeffs_initial=warp_coeffs_initial,
    )

    out_coeffs: tuple[float, float, float, float, float] | None = None
    if warp_coordinates is not None:
        arr = np.array(result["warp_coeffs"])
        out_coeffs = (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
            float(arr[3]),
            float(arr[4]),
        )

    return _OptimizationBatch(
        intrinsics=batch.intrinsics._with_params(result["intrinsics"]),
        cameras_from_target=[
            Pose._from_cpp(np.array(a)) for a in result["cameras_from_target"]
        ],
        frames=batch.frames,
        warp_coeffs=out_coeffs,
    )


def _stereographic_opencv_calibrate_inner(
    batch: _OptimizationBatch[StereographicOpenCV],
    config: StereographicOpenCVConfig,
    target_points: np.ndarray,
    warp_coordinates: WarpCoordinates | None = None,
) -> _OptimizationBatch[StereographicOpenCV]:
    params = batch.intrinsics._params()
    mask = config.optimize_mask()
    warp_coordinates_cpp = None
    if warp_coordinates is not None:
        warp_coordinates_cpp = warp_coordinates._to_cpp()
    warp_coeffs_initial = [0.0] * 5
    if batch.warp_coeffs is not None:
        warp_coeffs_initial = list(batch.warp_coeffs)

    result = lbb.calibrate_stereographic_opencv(
        intrinsics_initial_value=params,
        intrinsics_param_optimize_mask=mask.tolist(),
        cameras_from_target=[p._to_cpp() for p in batch.cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in batch.frames],
        warp_coordinates=warp_coordinates_cpp,
        warp_coeffs_initial=warp_coeffs_initial,
    )

    out_coeffs: tuple[float, float, float, float, float] | None = None
    if warp_coordinates is not None:
        arr = np.array(result["warp_coeffs"])
        out_coeffs = (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
            float(arr[3]),
            float(arr[4]),
        )

    return _OptimizationBatch(
        intrinsics=batch.intrinsics._with_params(result["intrinsics"]),
        cameras_from_target=[
            Pose._from_cpp(np.array(a)) for a in result["cameras_from_target"]
        ],
        frames=batch.frames,
        warp_coeffs=out_coeffs,
    )


def _compute_frame_diagnostics(
    intrinsics: OpenCV | PinholeSplined | StereographicOpenCV | StereographicSplined,
    cameras_from_target: list[Pose | None],
    frames: list[Frame],
    target_points: np.ndarray,
    inlier_masks: list[np.ndarray | None],
    target_warp: TargetWarp | None = None,
) -> list[FrameDiagnostics | None]:
    frame_diagnostics: list[FrameDiagnostics | None] = []
    for i in range(len(frames)):
        pose = cameras_from_target[i]
        if pose is None:
            frame_diagnostics.append(None)
            continue

        projected, residuals = _project_and_calculate_residuals(
            target_points,
            pose,
            frames[i],
            intrinsics,
            target_warp,
        )
        mask = inlier_masks[i]
        if mask is None:
            mask = np.ones(len(frames[i]), dtype=bool)

        frame_diagnostics.append(FrameDiagnostics(projected, residuals, mask))

    return frame_diagnostics


def _compute_mean_reproj(
    state: _OptimizationState[IntrinsicsT],
    target_points: np.ndarray,
    target_warp: TargetWarp | None,
) -> tuple[float, float]:
    """Compute mean and worst inlier reprojection error."""
    norms: list[np.ndarray] = []
    for i, pose in enumerate(state.cameras_from_target):
        if pose is None:
            continue
        mask = state.inlier_masks[i]
        frame = _apply_mask(state.frames[i], mask)
        _, r = _project_and_calculate_residuals(
            target_points,
            pose,
            frame,
            state.intrinsics,
            target_warp,
        )
        norms.append(np.linalg.norm(r, axis=1))
    all_norms = np.concatenate(norms)
    return float(np.mean(all_norms)), float(np.max(all_norms))


def _run_with_outlier_filtering(
    optimize_fn: Callable[
        [_OptimizationBatch[IntrinsicsT]], _OptimizationBatch[IntrinsicsT]
    ],
    initial_state: _OptimizationState[IntrinsicsT],
    target_points: np.ndarray,
    outlier_threshold_stddevs: float | None,
    warp_coordinates: WarpCoordinates | None = None,
    label: str = "Optimization",
) -> _OptimizationState[IntrinsicsT]:
    state = initial_state
    total_observations = sum(len(f) for f in state.frames)
    pass_num = 0

    for iteration in range(MAX_OUTLIER_FILTER_PASSES + 1):
        active_indices = [
            i for i, p in enumerate(state.cameras_from_target) if p is not None
        ]
        if not active_indices:
            raise ValueError(
                "All frames have been excluded; calibration cannot continue."
            )

        opt_frames = [
            _apply_mask(state.frames[i], state.inlier_masks[i]) for i in active_indices
        ]
        opt_poses: list[Pose] = [
            state.cameras_from_target[i]  # type: ignore[misc]
            for i in active_indices
        ]

        pass_num += 1
        start = default_timer()
        optimized = optimize_fn(
            _OptimizationBatch(
                intrinsics=state.intrinsics,
                cameras_from_target=opt_poses,
                frames=opt_frames,
                warp_coeffs=state.warp_coeffs,
            )
        )

        # Scatter optimized poses back
        for idx, pose in zip(active_indices, optimized.cameras_from_target):
            state.cameras_from_target[idx] = pose
        state.intrinsics = optimized.intrinsics
        state.warp_coeffs = optimized.warp_coeffs

        curr_target_warp = None
        if warp_coordinates is not None and state.warp_coeffs is not None:
            curr_target_warp = TargetWarp(warp_coordinates, state.warp_coeffs)

        elapsed = default_timer() - start
        residuals = []
        inlier_norms = []
        for i in active_indices:
            pose = state.cameras_from_target[i]
            assert pose is not None
            _, r = _project_and_calculate_residuals(
                target_points,
                pose,
                state.frames[i],
                state.intrinsics,
                curr_target_warp,
            )
            residuals.append(r)

            mask = state.inlier_masks[i]
            if mask is None:
                inlier_norms.append(np.linalg.norm(r, axis=1))
            else:
                inlier_norms.append(np.linalg.norm(r[mask], axis=1))

        all_inlier_norms = np.concatenate(inlier_norms)
        mean_reproj = float(np.mean(all_inlier_norms))
        worst_reproj = float(np.max(all_inlier_norms))
        log(
            f"{label} pass {pass_num}: {elapsed:.1f}s "
            f"(mean reproj={mean_reproj:.3f}px, worst={worst_reproj:.3f}px)"
        )

        if outlier_threshold_stddevs is None or iteration == MAX_OUTLIER_FILTER_PASSES:
            break

        new_active_masks = _filter_outliers(residuals, outlier_threshold_stddevs)

        changed = False
        for idx, new_mask in zip(active_indices, new_active_masks):
            old_mask = state.inlier_masks[idx]
            assert old_mask is not None

            if not new_mask.any():
                state.cameras_from_target[idx] = None
                state.inlier_masks[idx] = None
                changed = True
            elif not np.array_equal(old_mask, new_mask):
                state.inlier_masks[idx] = new_mask
                changed = True

        if not changed:
            break

        total_remaining = sum(int(m.sum()) for m in state.inlier_masks if m is not None)
        total_outliers = total_observations - total_remaining
        pct = total_outliers / total_observations * 100
        log(
            f"Outlier filtering: {total_outliers}/{total_observations}"
            f" ({pct:.1f}%) — re-optimizing..."
        )

    return state


def _solve_pnp_all_frames(
    K: np.ndarray,
    target_points: np.ndarray,
    frames: list[Frame],
    dist_coeffs: np.ndarray | None = None,
) -> tuple[list[Pose], list[bool], float]:
    """Run solvePnP for all frames.

    Every frame gets a pose — failed frames get an identity pose and are
    flagged so the caller can mask them out.

    Args:
        K: Camera intrinsics matrix, shape (3, 3).
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.
        dist_coeffs: Distortion coefficients for PnP. Zeros if None.

    Returns:
        Tuple of (poses, solved mask, mean squared error per point).
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    poses: list[Pose] = []
    solved: list[bool] = []
    total_squared_error = 0.0
    total_points = 0

    identity = Pose.from_rotvec_trans(rotvec=np.zeros(3), trans=np.array([0.0, 0.0, 1.0]))

    for frame in frames:
        obj_pts = target_points[frame.target_point_indices].astype(np.float64)
        img_pts = frame.detected_points_in_image.astype(np.float64)

        if len(obj_pts) >= 4:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs)
            if success:
                poses.append(
                    Pose.from_rotvec_trans(rotvec=rvec.flatten(), trans=tvec.flatten())
                )
                projected = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)[
                    0
                ].reshape(-1, 2)
                total_squared_error += float(np.sum((projected - img_pts) ** 2))
                total_points += len(obj_pts)
                solved.append(True)
                continue

        poses.append(identity)
        solved.append(False)

    mean_error = float("inf")
    if total_points > 0:
        mean_error = total_squared_error / total_points
    return poses, solved, mean_error


def _stereographic_pixels_to_normalized_xy(
    pixel_coords: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject centered stereographic pixels to normalized pinhole coordinates.

    Args:
        pixel_coords: Image coordinates, shape (N, 2).
        focal_length: Centered stereographic focal length.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        Normalized coordinates, shape (N, 2).
    """
    sx = (pixel_coords[:, 0] - cx) / focal_length
    sy = (pixel_coords[:, 1] - cy) / focal_length
    r_s_sq = sx * sx + sy * sy
    scale = 1.0 / (1.0 - 0.25 * r_s_sq)
    return np.column_stack([sx * scale, sy * scale])


def _project_stereographic_points(
    points_in_camera: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project camera-frame points with a centered stereographic model.

    Args:
        points_in_camera: Camera-frame target points, shape (N, 3).
        focal_length: Centered stereographic focal length.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        Image coordinates, shape (N, 2).
    """
    ray_norm = np.linalg.norm(points_in_camera, axis=1)
    denominator = ray_norm + points_in_camera[:, 2]
    sx = 2.0 * points_in_camera[:, 0] / denominator
    sy = 2.0 * points_in_camera[:, 1] / denominator
    return np.column_stack([focal_length * sx + cx, focal_length * sy + cy])


def _stereographic_front_mask(
    pixel_coords: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Select well-conditioned front-hemisphere points for pinhole PnP.

    Args:
        pixel_coords: Image coordinates, shape (N, 2).
        focal_length: Centered stereographic focal length.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        Boolean selection mask, shape (N,).
    """
    sx = (pixel_coords[:, 0] - cx) / focal_length
    sy = (pixel_coords[:, 1] - cy) / focal_length
    return sx * sx + sy * sy < 3.6


def _prepare_pnp_frame_data(
    target_points: np.ndarray,
    frames: list[Frame],
) -> list[_PnPFrameData]:
    """Gather contiguous object and image arrays for repeated PnP solves.

    Args:
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.

    Returns:
        Per-frame object points and image detections.
    """
    return [
        _PnPFrameData(
            object_points=np.ascontiguousarray(
                target_points[frame.target_point_indices],
                dtype=np.float64,
            ),
            image_points=np.ascontiguousarray(
                frame.detected_points_in_image,
                dtype=np.float64,
            ),
        )
        for frame in frames
    ]


def _score_stereographic_focal_length(
    focal_length: float,
    pnp_frames: list[_PnPFrameData],
    cx: float,
    cy: float,
) -> float:
    """Score a centered stereographic focal length with PnP reprojection error.

    Args:
        focal_length: Centered stereographic focal length.
        pnp_frames: Prepared frame data.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        Mean squared reprojection error per solved target point.
    """
    normalized_K = np.eye(3, dtype=np.float64)
    zero_distortion = np.zeros(4, dtype=np.float64)
    total_squared_error = 0.0
    total_points = 0

    for frame_data in pnp_frames:
        obj_pts = frame_data.object_points
        img_pts = frame_data.image_points
        if len(obj_pts) < 4:
            continue

        pnp_mask = _stereographic_front_mask(
            img_pts,
            focal_length,
            cx,
            cy,
        )
        if np.count_nonzero(pnp_mask) < 4:
            continue
        normalized_xy = _stereographic_pixels_to_normalized_xy(
            img_pts[pnp_mask], focal_length, cx, cy
        )
        success, rvec, tvec = cv2.solvePnP(
            obj_pts[pnp_mask],
            normalized_xy,
            normalized_K,
            zero_distortion,
        )
        if not success:
            continue

        rotmat = cv2.Rodrigues(rvec)[0]
        points_in_cam = obj_pts @ rotmat.T + tvec.reshape(1, 3)
        projected = _project_stereographic_points(points_in_cam, focal_length, cx, cy)
        total_squared_error += float(np.sum((projected - img_pts) ** 2))
        total_points += len(obj_pts)

    if total_points == 0:
        return float("inf")
    return total_squared_error / total_points


def _solve_pnp_all_frames_stereographic(
    focal_length: float,
    target_points: np.ndarray,
    frames: list[Frame],
    cx: float,
    cy: float,
) -> tuple[list[Pose], list[bool], float]:
    """Run solvePnP with detections unprojected by a stereographic model.

    Args:
        focal_length: Centered stereographic focal length.
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        Tuple of (poses, solved mask, mean squared error per point).
    """
    normalized_K = np.eye(3, dtype=np.float64)
    zero_distortion = np.zeros(4, dtype=np.float64)

    poses: list[Pose] = []
    solved: list[bool] = []
    total_squared_error = 0.0
    total_points = 0

    for frame in frames:
        obj_pts = np.ascontiguousarray(
            target_points[frame.target_point_indices], dtype=np.float64
        )
        img_pts = np.ascontiguousarray(
            frame.detected_points_in_image,
            dtype=np.float64,
        )

        if len(obj_pts) < 4:
            poses.append(Pose.identity())
            solved.append(False)
            continue

        pnp_mask = _stereographic_front_mask(
            img_pts,
            focal_length,
            cx,
            cy,
        )
        if np.count_nonzero(pnp_mask) < 4:
            poses.append(Pose.identity())
            solved.append(False)
            continue
        normalized_xy = _stereographic_pixels_to_normalized_xy(
            img_pts[pnp_mask], focal_length, cx, cy
        )
        success, rvec, tvec = cv2.solvePnP(
            obj_pts[pnp_mask],
            normalized_xy,
            normalized_K,
            zero_distortion,
        )
        if not success:
            poses.append(Pose.identity())
            solved.append(False)
            continue

        camera_from_target = Pose.from_rotvec_trans(
            rotvec=rvec.flatten(), trans=tvec.flatten()
        )
        poses.append(camera_from_target)
        points_in_cam = camera_from_target.apply(obj_pts)
        projected = _project_stereographic_points(points_in_cam, focal_length, cx, cy)
        total_squared_error += float(np.sum((projected - img_pts) ** 2))
        total_points += len(obj_pts)
        solved.append(True)

    mean_error = float("inf")
    if total_points > 0:
        mean_error = total_squared_error / total_points
    return poses, solved, mean_error


def _matching_opencv_model_from_stereographic(
    config: OpenCVConfig,
    focal_length: float,
) -> OpenCV:
    """Fit an OpenCV seed model that approximates centered stereographic projection.

    Args:
        config: Camera model configuration.
        focal_length: Centered stereographic focal length.

    Returns:
        OpenCV camera model initialized from the fitted backend parameters.
    """
    matching_fn = getattr(lbb, "get_matching_stereographic_opencv_model")
    out = matching_fn(
        config.image_width,
        config.image_height,
        float(focal_length),
        config.included_distortion_coefficients.tolist(),
    )
    params = np.asarray(out["intrinsics"], dtype=np.float64)
    return OpenCV(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=float(params[0]),
        fy=float(params[1]),
        cx=float(params[2]),
        cy=float(params[3]),
        distortion_coeffs=params[4:],
    )


def _solve_pnp_all_frames_with_model(
    model: OpenCV | PinholeSplined | StereographicOpenCV | StereographicSplined,
    target_points: np.ndarray,
    frames: list[Frame],
) -> tuple[list[Pose], list[bool], float]:
    """Estimate camera-from-target transforms with normalized detections.

    Args:
        model: Camera model used to normalize detected image points.
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.

    Returns:
        Tuple of camera-from-target transforms, per-frame solved mask, and mean
        squared reprojection error.
    """
    normalized_K = np.eye(3, dtype=np.float64)
    zero_distortion = np.zeros(4, dtype=np.float64)

    cameras_from_target: list[Pose] = []
    solved: list[bool] = []
    total_squared_error = 0.0
    total_points = 0

    for frame in frames:
        obj_pts = np.ascontiguousarray(
            target_points[frame.target_point_indices], dtype=np.float64
        )
        normalized_points_in_camera = model.normalize_points(
            frame.detected_points_in_image
        )
        pnp_mask = normalized_points_in_camera[:, 2] > 0.0
        if np.count_nonzero(pnp_mask) < 4:
            cameras_from_target.append(Pose.identity())
            solved.append(False)
            continue

        pnp_rays = normalized_points_in_camera[pnp_mask]
        z = pnp_rays[:, 2:3]
        normalized_xy = np.ascontiguousarray(
            pnp_rays[:, :2] / z,
            dtype=np.float64,
        )

        if len(obj_pts) < 4:
            cameras_from_target.append(Pose.identity())
            solved.append(False)
            continue

        success, rvec, tvec = cv2.solvePnP(
            obj_pts[pnp_mask],
            normalized_xy,
            normalized_K,
            zero_distortion,
        )
        if not success:
            cameras_from_target.append(Pose.identity())
            solved.append(False)
            continue

        camera_from_target = Pose.from_rotvec_trans(
            rotvec=rvec.flatten(), trans=tvec.flatten()
        )
        cameras_from_target.append(camera_from_target)
        points_in_cam = camera_from_target.apply(obj_pts)
        projected = model.project_points(points_in_cam)
        total_squared_error += float(
            np.sum((projected - frame.detected_points_in_image) ** 2)
        )
        total_points += len(obj_pts)
        solved.append(True)

    mean_error = float("inf")
    if total_points > 0:
        mean_error = total_squared_error / total_points
    return cameras_from_target, solved, mean_error


def _get_initial_state_with_pnp(
    config: OpenCVConfig,
    target_points: np.ndarray,
    frames: list[Frame],
    initial_camera_model: OpenCV | None = None,
) -> tuple[OpenCV, list[Pose], list[bool]]:
    """Estimate initial intrinsics and poses using PnP.

    If an initial model is supplied, uses it directly. Otherwise, uses a
    centered stereographic model for PnP, either at ``config.initial_focal_length``
    or over a log-spaced focal-length sweep. The selected stereographic model is
    then fit by a pinhole + OpenCV distortion model for the backend optimizer.

    Args:
        config: Camera model configuration.
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.
        initial_camera_model: Optional initial intrinsics to optimize from.

    Returns:
        Tuple of (initial intrinsics, poses, solved mask).
    """
    if initial_camera_model is not None:
        cameras_from_target, solved, _ = _solve_pnp_all_frames(
            initial_camera_model.K(),
            target_points,
            frames,
            initial_camera_model.distortion_coeffs,
        )
        return initial_camera_model, cameras_from_target, solved

    cx = config.image_width / 2.0
    cy = config.image_height / 2.0

    if config.initial_focal_length is not None:
        focal_length = float(config.initial_focal_length)
        poses, solved, _ = _solve_pnp_all_frames_stereographic(
            focal_length,
            target_points,
            frames,
            cx,
            cy,
        )
        intrinsics = _matching_opencv_model_from_stereographic(config, focal_length)
        return intrinsics, poses, solved

    max_dim = max(config.image_width, config.image_height)
    candidates = np.geomspace(0.2 * max_dim, 5.0 * max_dim, num=30)
    sweep_frames = _select_focal_sweep_frames(
        frames,
        config.image_width,
        config.image_height,
    )
    sweep_pnp_frames = _prepare_pnp_frame_data(target_points, sweep_frames)

    best_focal = float(candidates[0])
    best_error = float("inf")

    for f in candidates:
        error = _score_stereographic_focal_length(
            float(f),
            sweep_pnp_frames,
            cx,
            cy,
        )
        if error < best_error:
            best_error = error
            best_focal = float(f)

    log(f"Auto-estimated initial stereographic focal length: {best_focal:.1f} px")

    best_poses, best_solved, _ = _solve_pnp_all_frames_stereographic(
        best_focal,
        target_points,
        frames,
        cx,
        cy,
    )
    intrinsics = _matching_opencv_model_from_stereographic(config, best_focal)
    return intrinsics, best_poses, best_solved


def _get_initial_state_with_pnp_stereographic_opencv(
    config: StereographicOpenCVConfig,
    target_points: np.ndarray,
    frames: list[Frame],
    initial_camera_model: StereographicOpenCV | None = None,
) -> tuple[StereographicOpenCV, list[Pose], list[bool]]:
    """Estimate initial Stereographic OpenCV intrinsics and poses using stereographic PnP.

    Args:
        config: Camera model configuration.
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.
        initial_camera_model: Optional initial intrinsics to optimize from.

    Returns:
        Tuple of (initial intrinsics, poses, solved mask).
    """
    if initial_camera_model is not None:
        cameras_from_target, solved, _ = _solve_pnp_all_frames_with_model(
            initial_camera_model,
            target_points,
            frames,
        )
        return initial_camera_model, cameras_from_target, solved

    cx = config.image_width / 2.0
    cy = config.image_height / 2.0

    if config.initial_focal_length is not None:
        focal_length = float(config.initial_focal_length)
        poses, solved, _ = _solve_pnp_all_frames_stereographic(
            focal_length,
            target_points,
            frames,
            cx,
            cy,
        )
        model = StereographicOpenCV(
            image_height=config.image_height,
            image_width=config.image_width,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            distortion_coeffs=np.zeros(14, dtype=np.float64),
        )
        return model, poses, solved

    max_dim = max(config.image_width, config.image_height)
    candidates = np.geomspace(0.2 * max_dim, 5.0 * max_dim, num=30)
    sweep_frames = _select_focal_sweep_frames(
        frames,
        config.image_width,
        config.image_height,
    )
    sweep_pnp_frames = _prepare_pnp_frame_data(target_points, sweep_frames)

    best_focal = float(candidates[0])
    best_error = float("inf")

    for f in candidates:
        error = _score_stereographic_focal_length(
            float(f),
            sweep_pnp_frames,
            cx,
            cy,
        )
        if error < best_error:
            best_error = error
            best_focal = float(f)

    log(f"Auto-estimated initial stereographic focal length: {best_focal:.1f} px")

    best_poses, best_solved, _ = _solve_pnp_all_frames_stereographic(
        best_focal,
        target_points,
        frames,
        cx,
        cy,
    )
    model = StereographicOpenCV(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=best_focal,
        fy=best_focal,
        cx=cx,
        cy=cy,
        distortion_coeffs=np.zeros(14, dtype=np.float64),
    )
    return model, best_poses, best_solved


_PLANARITY_RATIO_THRESHOLD = 0.1
_RECT_FIT_RATIO_THRESHOLD = 0.85


def _make_warp_coordinates(target_points: np.ndarray) -> WarpCoordinates | None:
    centroid = target_points.mean(axis=0)
    centered = target_points - centroid
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    planarity_ratio = np.inf
    if s[1] > 1e-10:
        planarity_ratio = s[2] / s[1]
    if planarity_ratio > _PLANARITY_RATIO_THRESHOLD:
        warn(
            "Target warp can only be estimated with a planar target "
            f"(planarity ratio {planarity_ratio:.3f} > {_PLANARITY_RATIO_THRESHOLD}). "
            "Skipping warp estimation."
        )
        return None

    x_in_plane = Vt[0]
    y_in_plane = Vt[1]
    points_2d = centered @ np.column_stack([x_in_plane, y_in_plane])

    pts32 = points_2d.astype(np.float32)
    rect = cv2.minAreaRect(pts32)
    rect_w, rect_h = float(rect[1][0]), float(rect[1][1])
    rect_area = rect_w * rect_h
    hull_area = float(cv2.contourArea(cv2.convexHull(pts32)))

    use_rect = rect_area > 1e-10 and hull_area / rect_area > _RECT_FIT_RATIO_THRESHOLD

    if use_rect:
        box = cv2.boxPoints(rect).astype(float)
        e0 = box[1] - box[0]
        e1 = box[3] - box[0]
        u = e0 / np.linalg.norm(e0)
        v = e1 / np.linalg.norm(e1)
        cx2, cy2 = float(rect[0][0]), float(rect[0][1])
        x_scale = float(np.linalg.norm(e0) / 2.0)
        y_scale = float(np.linalg.norm(e1) / 2.0)
    else:
        log("Target is not rectangular; falling back to PCA for warp frame axes.")
        eigvals, eigvecs = np.linalg.eigh(np.cov(points_2d.T))
        order = np.argsort(eigvals)[::-1]
        u = eigvecs[:, order[0]]
        v = eigvecs[:, order[1]]
        proj_u = points_2d @ u
        proj_v = points_2d @ v
        cu = (proj_u.max() + proj_u.min()) / 2.0
        cv_val = (proj_v.max() + proj_v.min()) / 2.0
        center_2d = cu * u + cv_val * v
        cx2, cy2 = float(center_2d[0]), float(center_2d[1])
        x_scale = float((proj_u.max() - proj_u.min()) / 2.0)
        y_scale = float((proj_v.max() - proj_v.min()) / 2.0)

    x_hat = u[0] * x_in_plane + u[1] * y_in_plane
    y_hat = v[0] * x_in_plane + v[1] * y_in_plane
    z_hat = np.cross(x_hat, y_hat)

    center_3d = centroid + cx2 * x_in_plane + cy2 * y_in_plane
    R = np.column_stack([x_hat, y_hat, z_hat])

    return WarpCoordinates(
        target_from_warp_frame=Pose.from_rotmat_trans(rotmat=R, trans=center_3d),
        x_scale=x_scale,
        y_scale=y_scale,
    )


def _recover_failed_pnp(
    optimize_fn: Callable[
        [_OptimizationBatch[IntrinsicsT]], _OptimizationBatch[IntrinsicsT]
    ],
    solve_pnp_fn: Callable[[IntrinsicsT], tuple[list[Pose], list[bool], float]],
    initial_intrinsics: IntrinsicsT,
    initial_poses: list[Pose],
    pnp_solved: list[bool],
    target_points: np.ndarray,
    frames: list[Frame],
    image_width: int,
    image_height: int,
) -> tuple[IntrinsicsT, list[Pose | None], list[np.ndarray | None]]:
    """Fit a subsampled model and re-run PnP to recover initially failed frames.

    Args:
        optimize_fn: Optimizer callback.
        solve_pnp_fn: Callback that re-estimates every frame pose from a model.
        initial_intrinsics: Intrinsics from first PnP pass.
        initial_poses: Poses from first PnP pass (identity for failed frames).
        pnp_solved: Per-frame success mask from first PnP pass.
        target_points: 3D target points.
        frames: All calibration frames.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Updated (intrinsics, poses, inlier_masks) with recovered frames.
    """
    solved_frames = [f for f, ok in zip(frames, pnp_solved) if ok]
    solved_poses = [p for p, ok in zip(initial_poses, pnp_solved) if ok]
    covering_indices = set(
        _select_covering_frames(solved_frames, image_width, image_height)
    )
    covering_frames = [f for i, f in enumerate(solved_frames) if i in covering_indices]
    covering_poses = [p for i, p in enumerate(solved_poses) if i in covering_indices]
    log(
        f"Fitting subsampled model ({len(covering_frames)} frames)"
        f" to recover failed PnP..."
    )
    result = optimize_fn(
        _OptimizationBatch(
            intrinsics=initial_intrinsics,
            cameras_from_target=covering_poses,
            frames=covering_frames,
            warp_coeffs=None,
        )
    )
    model = result.intrinsics
    log("Re-running PnP with optimized model...")
    re_poses, re_solved, _ = solve_pnp_fn(model)
    n_re_solved = sum(re_solved)
    log(f"Re-PnP solved {n_re_solved}/{len(frames)} frames")

    poses: list[Pose | None] = []
    inlier_masks: list[np.ndarray | None] = []
    for frame, camera_from_target, ok in zip(frames, re_poses, re_solved):
        if ok:
            poses.append(camera_from_target)
            inlier_masks.append(np.ones(len(frame), dtype=bool))
            continue
        poses.append(None)
        inlier_masks.append(None)
    return model, poses, inlier_masks


def _opencv_calibrate(
    target_points: np.ndarray,
    frames: list[Frame],
    config: OpenCVConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
    initial_camera_model: OpenCV | None = None,
) -> CalibrationResult[OpenCV]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    log("Computing initial poses with PnP...")
    initial_intrinsics, initial_poses, pnp_solved = _get_initial_state_with_pnp(
        config, target_points, frames, initial_camera_model
    )

    n_solved = sum(pnp_solved)
    n_failed = len(frames) - n_solved
    log(f"PnP solved {n_solved}/{len(frames)} frames")
    if n_failed > 0:
        log(f"{n_failed} frame(s) failed PnP, excluding from optimization")

    poses: list[Pose | None] = []
    inlier_masks: list[np.ndarray | None] = []
    for frame, camera_from_target, ok in zip(frames, initial_poses, pnp_solved):
        if ok:
            poses.append(camera_from_target)
            inlier_masks.append(np.ones(len(frame), dtype=bool))
            continue
        poses.append(None)
        inlier_masks.append(None)

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(batch: _OptimizationBatch[OpenCV]) -> _OptimizationBatch[OpenCV]:
        return _opencv_calibrate_inner(batch, config, target_points, warp_coordinates)

    if n_failed > 0:
        initial_intrinsics, poses, inlier_masks = _recover_failed_pnp(
            optimize_fn,
            lambda model: _solve_pnp_all_frames(
                model.K(),
                target_points,
                frames,
                model.distortion_coeffs,
            ),
            initial_intrinsics,
            initial_poses,
            pnp_solved,
            target_points,
            frames,
            config.image_width,
            config.image_height,
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(initial_intrinsics, poses, frames, None, inlier_masks),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
        label="OpenCV",
    )

    target_warp = None
    if warp_coordinates is not None and state.warp_coeffs is not None:
        target_warp = TargetWarp(
            warp_coordinates=warp_coordinates, object_warp=state.warp_coeffs
        )
        deflection = target_warp.max_deflection(target_points)
        log(f"Target warp max deflection: {deflection:.4f} (target units)")

    diagnostics = _compute_frame_diagnostics(
        state.intrinsics,
        state.cameras_from_target,
        state.frames,
        target_points,
        inlier_masks=state.inlier_masks,
        target_warp=target_warp,
    )

    return CalibrationResult(
        camera_model=state.intrinsics,
        cameras_from_target=state.cameras_from_target,
        frame_diagnostics=diagnostics,
        frames=list(frames),
        target_points=target_points,
        target_warp=target_warp,
    )


def _calibrate_stereographic_opencv(
    target_points: np.ndarray,
    frames: list[Frame],
    config: StereographicOpenCVConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
    initial_camera_model: StereographicOpenCV | None = None,
) -> CalibrationResult[StereographicOpenCV]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    log("Computing initial poses with stereographic PnP...")
    initial_intrinsics, initial_poses, pnp_solved = (
        _get_initial_state_with_pnp_stereographic_opencv(
            config,
            target_points,
            frames,
            initial_camera_model,
        )
    )

    n_solved = sum(pnp_solved)
    n_failed = len(frames) - n_solved
    log(f"PnP solved {n_solved}/{len(frames)} frames")
    if n_failed > 0:
        log(f"{n_failed} frame(s) failed PnP, excluding from optimization")

    poses: list[Pose | None] = []
    inlier_masks: list[np.ndarray | None] = []
    for frame, camera_from_target, ok in zip(frames, initial_poses, pnp_solved):
        if ok:
            poses.append(camera_from_target)
            inlier_masks.append(np.ones(len(frame), dtype=bool))
            continue
        poses.append(None)
        inlier_masks.append(None)

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(
        batch: _OptimizationBatch[StereographicOpenCV],
    ) -> _OptimizationBatch[StereographicOpenCV]:
        return _stereographic_opencv_calibrate_inner(
            batch,
            config,
            target_points,
            warp_coordinates,
        )

    if n_failed > 0:
        initial_intrinsics, poses, inlier_masks = _recover_failed_pnp(
            optimize_fn,
            lambda model: _solve_pnp_all_frames_with_model(
                model,
                target_points,
                frames,
            ),
            initial_intrinsics,
            initial_poses,
            pnp_solved,
            target_points,
            frames,
            config.image_width,
            config.image_height,
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(initial_intrinsics, poses, frames, None, inlier_masks),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
        label="Stereographic OpenCV",
    )

    target_warp = None
    if warp_coordinates is not None and state.warp_coeffs is not None:
        target_warp = TargetWarp(
            warp_coordinates=warp_coordinates,
            object_warp=state.warp_coeffs,
        )
        deflection = target_warp.max_deflection(target_points)
        log(f"Target warp max deflection: {deflection:.4f} (target units)")

    diagnostics = _compute_frame_diagnostics(
        state.intrinsics,
        state.cameras_from_target,
        state.frames,
        target_points,
        inlier_masks=state.inlier_masks,
        target_warp=target_warp,
    )

    return CalibrationResult(
        camera_model=state.intrinsics,
        cameras_from_target=state.cameras_from_target,
        frame_diagnostics=diagnostics,
        frames=list(frames),
        target_points=target_points,
        target_warp=target_warp,
    )


def _pinhole_splined_refine_inner(
    batch: _OptimizationBatch[PinholeSplined],
    config: PinholeSplinedConfig,
    target_points: np.ndarray,
    warp_coordinates: WarpCoordinates | None,
) -> _OptimizationBatch[PinholeSplined]:
    warp_coordinates_cpp = None
    if warp_coordinates is not None:
        warp_coordinates_cpp = warp_coordinates._to_cpp()
    warp_coeffs_initial = [0.0] * 5
    if batch.warp_coeffs is not None:
        warp_coeffs_initial = list(batch.warp_coeffs)

    fine_tune_result = lbb.fine_tune_pinhole_splined(
        model_config=_pinhole_splined_cpp_optimization_config(batch.intrinsics, config),
        intrinsics_parameters=batch.intrinsics._cpp_params(),
        cameras_from_target=[pose._to_cpp() for pose in batch.cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in batch.frames],
        warp_coordinates=warp_coordinates_cpp,
        warp_coeffs_initial=warp_coeffs_initial,
    )

    out_coeffs: tuple[float, float, float, float, float] | None = None
    if warp_coordinates is not None:
        arr = np.array(fine_tune_result["warp_coeffs"])
        out_coeffs = (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
            float(arr[3]),
            float(arr[4]),
        )

    return _OptimizationBatch(
        intrinsics=replace(
            batch.intrinsics,
            dx_grid=fine_tune_result["dx_grid"],
            dy_grid=fine_tune_result["dy_grid"],
        ),
        cameras_from_target=[
            Pose._from_cpp(np.array(a)) for a in fine_tune_result["cameras_from_target"]
        ],
        frames=batch.frames,
        warp_coeffs=out_coeffs,
    )


def _stereographic_splined_refine_inner(
    batch: _OptimizationBatch[StereographicSplined],
    config: StereographicSplinedConfig,
    target_points: np.ndarray,
    warp_coordinates: WarpCoordinates | None,
) -> _OptimizationBatch[StereographicSplined]:
    warp_coordinates_cpp = None
    if warp_coordinates is not None:
        warp_coordinates_cpp = warp_coordinates._to_cpp()
    warp_coeffs_initial = [0.0] * 5
    if batch.warp_coeffs is not None:
        warp_coeffs_initial = list(batch.warp_coeffs)

    fine_tune_result = lbb.fine_tune_stereographic_splined(
        model_config=_stereographic_splined_cpp_optimization_config(
            batch.intrinsics,
            config,
        ),
        intrinsics_parameters=batch.intrinsics._cpp_params(),
        cameras_from_target=[pose._to_cpp() for pose in batch.cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in batch.frames],
        warp_coordinates=warp_coordinates_cpp,
        warp_coeffs_initial=warp_coeffs_initial,
    )

    out_coeffs: tuple[float, float, float, float, float] | None = None
    if warp_coordinates is not None:
        arr = np.array(fine_tune_result["warp_coeffs"])
        out_coeffs = (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
            float(arr[3]),
            float(arr[4]),
        )

    return _OptimizationBatch(
        intrinsics=replace(
            batch.intrinsics,
            dx_grid=fine_tune_result["dx_grid"],
            dy_grid=fine_tune_result["dy_grid"],
        ),
        cameras_from_target=[
            Pose._from_cpp(np.array(a)) for a in fine_tune_result["cameras_from_target"]
        ],
        frames=batch.frames,
        warp_coeffs=out_coeffs,
    )


def _pinhole_splined_cpp_optimization_config(
    model: PinholeSplined,
    config: PinholeSplinedConfig,
) -> lbb.PinholeSplinedOptimizationConfig:
    """Build the C++ spline optimizer config from model geometry and fit config.

    Args:
        model: Camera model providing the fitted FOV.
        config: Calibration config providing optimizer settings.

    Returns:
        C++ spline config for the optimizer.
    """
    return lbb.PinholeSplinedOptimizationConfig(
        config.image_width,
        config.image_height,
        model.fov_deg_x,
        model.fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
        config.smoothness_lambda,
    )


def _stereographic_splined_cpp_optimization_config(
    model: StereographicSplined,
    config: StereographicSplinedConfig,
) -> lbb.StereographicSplinedOptimizationConfig:
    """Build the C++ stereographic spline optimizer config.

    Args:
        model: Camera model providing the fitted FOV.
        config: Calibration config providing optimizer settings.

    Returns:
        C++ spline config for the optimizer.
    """
    return lbb.StereographicSplinedOptimizationConfig(
        config.image_width,
        config.image_height,
        model.fov_deg_x,
        model.fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
        config.smoothness_lambda,
    )


def _select_covering_frames(
    frames: list[Frame],
    image_width: int,
    image_height: int,
    max_frames: int = 30,
    cell_fraction: float = 0.02,
) -> list[int]:
    """Select a subset of frames that maximizes spatial coverage.

    Greedily picks the frame covering the most uncovered image cells until
    max_frames is reached or all cells are covered.

    Args:
        frames: Input calibration frames.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        max_frames: Maximum number of frames to select.
        cell_fraction: Cell size as a fraction of the smaller image dimension.

    Returns:
        Indices of selected frames ordered by coverage contribution.
    """
    cell_size = cell_fraction * min(image_width, image_height)
    nx = int(np.ceil(image_width / cell_size))

    frame_cells: list[set[int]] = []
    for frame in frames:
        pts = frame.detected_points_in_image
        cx = np.clip((pts[:, 0] / cell_size).astype(int), 0, nx - 1)
        cy = (pts[:, 1] / cell_size).astype(int)
        frame_cells.append(set((cy * nx + cx).tolist()))

    covered: set[int] = set()
    selected: list[int] = []
    remaining = set(range(len(frames)))

    for _ in range(min(max_frames, len(frames))):
        best_idx = -1
        best_new = -1
        for i in remaining:
            new_count = len(frame_cells[i] - covered)
            if new_count > best_new:
                best_new = new_count
                best_idx = i
        if best_idx < 0 or best_new == 0:
            break
        covered |= frame_cells[best_idx]
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _select_focal_sweep_frames(
    frames: list[Frame],
    image_width: int,
    image_height: int,
    max_frames: int = FOCAL_SWEEP_MAX_FRAMES,
) -> list[Frame]:
    """Select frames used for initial focal-length sweeps.

    Args:
        frames: Input calibration frames.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        max_frames: Maximum number of frames to keep.

    Returns:
        Coverage-selected frames, or all frames if the input is already small.
    """
    if len(frames) <= max_frames:
        return frames

    sub_indices = _select_covering_frames(
        frames,
        image_width,
        image_height,
        max_frames=max_frames,
    )
    return [frames[i] for i in sub_indices]


def _compute_spline_grid_fov_from_unit_ray_model(
    model: StereographicOpenCV,
    max_fov_deg: float = 175.0,
) -> tuple[float, float]:
    """Compute spline-grid FOV from image-border stereographic coordinates.

    Args:
        model: Fitted stereographic camera model.
        max_fov_deg: Maximum allowed FOV.

    Returns:
        Symmetric (fov_deg_x, fov_deg_y) covering the image border, capped at
        max_fov_deg.
    """
    w, h = float(model.image_width), float(model.image_height)
    n = 80
    t = np.linspace(0, 1, n)
    edges = np.concatenate(
        [
            np.column_stack([t * w, np.zeros(n)]),
            np.column_stack([t * w, np.full(n, h)]),
            np.column_stack([np.zeros(n), t * h]),
            np.column_stack([np.full(n, w), t * h]),
        ]
    )
    rays = model.normalize_points(edges)
    denominator = 1.0 + rays[:, 2]
    stereographic_x = 2.0 * rays[:, 0] / denominator
    stereographic_y = 2.0 * rays[:, 1] / denominator
    half_x = float(np.max(np.abs(stereographic_x)))
    half_y = float(np.max(np.abs(stereographic_y)))
    fov_x = float(np.degrees(4.0 * np.arctan(half_x / 2.0)))
    fov_y = float(np.degrees(4.0 * np.arctan(half_y / 2.0)))
    return min(fov_x, max_fov_deg), min(fov_y, max_fov_deg)


def _compute_spline_fov_from_seed_coverage(
    seed_model: StereographicOpenCV,
    cameras_from_target: list[Pose],
    pnp_solved_mask: list[bool],
    target_points: np.ndarray,
    frames: list[Frame],
    max_fov_deg: float,
) -> tuple[float, float]:
    """Compute grid support from the raw seed border and observed target rays.

    Args:
        seed_model: Distortion-free stereographic seed.
        cameras_from_target: Camera-from-target transforms from seed-model PnP.
        pnp_solved_mask: Per-frame PnP success mask.
        target_points: 3D target points, shape (N, 3).
        frames: All calibration frames.
        max_fov_deg: Maximum grid FOV on either axis.

    Returns:
        Grid FOV covering both the seed image border and observed target rays.
    """
    fov_x, fov_y = _compute_spline_grid_fov_from_unit_ray_model(
        seed_model,
        max_fov_deg=max_fov_deg,
    )
    stereographic_points: list[np.ndarray] = []
    for frame, camera_from_target, solved in zip(
        frames,
        cameras_from_target,
        pnp_solved_mask,
    ):
        if not solved:
            continue
        points_in_camera = camera_from_target.apply(
            target_points[frame.target_point_indices]
        )
        rays = points_in_camera / np.linalg.norm(points_in_camera, axis=1, keepdims=True)
        denominator = 1.0 + rays[:, 2:3]
        stereographic_points.append(2.0 * rays[:, :2] / denominator)

    if not stereographic_points:
        return fov_x, fov_y

    coverage = np.concatenate(stereographic_points)
    tolerance = 1.03
    half_x = float(np.max(np.abs(coverage[:, 0]))) * tolerance
    half_y = float(np.max(np.abs(coverage[:, 1]))) * tolerance
    coverage_fov_x = float(np.degrees(4.0 * np.arctan(half_x / 2.0)))
    coverage_fov_y = float(np.degrees(4.0 * np.arctan(half_y / 2.0)))
    return (
        min(max(fov_x, coverage_fov_x), max_fov_deg),
        min(max(fov_y, coverage_fov_y), max_fov_deg),
    )


def _fit_stereographic_seed(
    target_points: np.ndarray,
    frames: list[Frame],
    config: PinholeSplinedConfig | StereographicSplinedConfig,
) -> CalibrationResult[StereographicOpenCV]:
    """Fit one distortion-free stereographic seed on coverage-selected frames.

    Args:
        target_points: 3D target points, shape (N, 3).
        frames: All calibration frames.
        config: Spline config used for the image size and initial focal length.

    Returns:
        Distortion-free stereographic calibration result.
    """
    start_time = default_timer()
    seed_indices = _select_covering_frames(
        frames,
        config.image_width,
        config.image_height,
        max_frames=SEED_FIT_MAX_FRAMES,
    )
    seed_frames = [frames[index] for index in seed_indices]
    seed_config = StereographicOpenCVConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        initial_focal_length=config.initial_focal_length,
        included_distortion_coefficients=StereographicOpenCVConfig.NONE,
    )

    disable_logs()
    try:
        result = _calibrate_stereographic_opencv(
            target_points,
            seed_frames,
            seed_config,
            None,
            estimate_target_warp=False,
        )
    finally:
        enable_logs()
    fov_x, fov_y = _compute_spline_grid_fov_from_unit_ray_model(
        result.camera_model,
        max_fov_deg=220.0,
    )
    log(
        f"Fitted distortion-free stereographic seed model: "
        f"{default_timer() - start_time:.1f}s "
        f"(FOV: {fov_x:.1f}° x {fov_y:.1f}°)"
    )
    return result


def _prepare_spline_frame_state(
    all_poses_pnp: list[Pose],
    pnp_solved_mask: list[bool],
    frames: list[Frame],
    target_points: np.ndarray,
    estimate_target_warp: bool,
) -> tuple[
    list[Pose | None],
    list[np.ndarray | None],
    WarpCoordinates | None,
]:
    """Prepare shared per-frame and target-warp spline optimization state.

    Args:
        all_poses_pnp: Camera-from-target transforms from PnP.
        pnp_solved_mask: Per-frame PnP success mask.
        frames: All calibration frames.
        target_points: 3D target points, shape (N, 3).
        estimate_target_warp: Whether the final spline fit estimates target warp.

    Returns:
        Optional poses, initial inlier masks, and optional warp coordinates.
    """
    n_solved = sum(pnp_solved_mask)
    log(f"PnP solved {n_solved}/{len(frames)} frames")

    poses: list[Pose | None] = []
    inlier_masks: list[np.ndarray | None] = []
    for frame, camera_from_target, ok in zip(frames, all_poses_pnp, pnp_solved_mask):
        if ok:
            poses.append(camera_from_target)
            inlier_masks.append(np.ones(len(frame), dtype=bool))
            continue
        poses.append(None)
        inlier_masks.append(None)

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)
    return poses, inlier_masks, warp_coordinates


def _build_initial_spline_model(
    seed_model: StereographicOpenCV,
    config: PinholeSplinedConfig,
    fov_deg_x: float,
    fov_deg_y: float,
) -> PinholeSplined:
    """Build a pinhole spline matching a distortion-free stereographic seed.

    Args:
        seed_model: Distortion-free stereographic seed.
        config: Spline config.
        fov_deg_x: Target FOV in x for the spline grid.
        fov_deg_y: Target FOV in y for the spline grid.

    Returns:
        Initial model with knots matched to stereographic projection.
    """
    cpp_config = lbb.PinholeSplinedOptimizationConfig(
        config.image_width,
        config.image_height,
        fov_deg_x,
        fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
        config.smoothness_lambda,
    )

    image_bound_x = np.tan(np.deg2rad(fov_deg_x) / 2.0) * 0.8
    image_bound_y = np.tan(np.deg2rad(fov_deg_y) / 2.0) * 0.8

    matching_fn = getattr(lbb, "get_matching_stereographic_spline_distortion_model")
    out_dict = matching_fn(
        cpp_config,
        float(image_bound_x),
        float(image_bound_y),
    )

    return PinholeSplined(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=seed_model.fx,
        fy=seed_model.fy,
        cx=seed_model.cx,
        cy=seed_model.cy,
        dx_grid=out_dict["x_knots"],
        dy_grid=out_dict["y_knots"],
        num_knots_x=config.num_knots_x,
        num_knots_y=config.num_knots_y,
        fov_deg_x=fov_deg_x,
        fov_deg_y=fov_deg_y,
    )


def _build_initial_stereographic_spline_model(
    seed_model: StereographicOpenCV,
    config: StereographicSplinedConfig,
    fov_deg_x: float,
    fov_deg_y: float,
) -> StereographicSplined:
    """Build the initial stereographic spline model from a Stereographic OpenCV seed.

    Args:
        seed_model: Seed stereographic model.
        config: Spline config.
        fov_deg_x: Target FOV in x for the spline grid.
        fov_deg_y: Target FOV in y for the spline grid.

    Returns:
        Initial StereographicSplined model with zero spline correction.
    """
    half_x = 2.0 * np.tan(np.deg2rad(fov_deg_x) / 4.0)
    half_y = 2.0 * np.tan(np.deg2rad(fov_deg_y) / 4.0)
    xs = np.linspace(-half_x, half_x, config.num_knots_x)
    ys = np.linspace(-half_y, half_y, config.num_knots_y)
    sx, sy = np.meshgrid(xs, ys)
    radius2 = sx * sx + sy * sy
    denom = 4.0 + radius2
    rays = np.column_stack(
        [
            (4.0 * sx / denom).ravel(),
            (4.0 * sy / denom).ravel(),
            ((4.0 - radius2) / denom).ravel(),
        ]
    )
    seed_pixels = seed_model.project_points(rays)
    seed_sx = ((seed_pixels[:, 0] - seed_model.cx) / seed_model.fx).reshape(sx.shape)
    seed_sy = ((seed_pixels[:, 1] - seed_model.cy) / seed_model.fy).reshape(sy.shape)

    return StereographicSplined(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=seed_model.fx,
        fy=seed_model.fy,
        cx=seed_model.cx,
        cy=seed_model.cy,
        dx_grid=np.asarray(seed_sx - sx, dtype=np.float64),
        dy_grid=np.asarray(seed_sy - sy, dtype=np.float64),
        num_knots_x=config.num_knots_x,
        num_knots_y=config.num_knots_y,
        fov_deg_x=fov_deg_x,
        fov_deg_y=fov_deg_y,
    )


def _calibrate_pinhole_splined(
    target_points: np.ndarray,
    frames: list[Frame],
    config: PinholeSplinedConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
    initial_camera_model: PinholeSplined | None = None,
) -> CalibrationResult[PinholeSplined]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )

    if initial_camera_model is None:
        seed_result = _fit_stereographic_seed(
            target_points,
            frames,
            config,
        )
        seed_model = seed_result.camera_model
        all_poses_pnp, pnp_solved_mask, _ = _solve_pnp_all_frames_with_model(
            seed_model,
            target_points,
            frames,
        )

        if config.fov_deg_xy is not None:
            fov_deg_x, fov_deg_y = config.fov_deg_xy
            log(f"Spline FOV (user-specified): {fov_deg_x:.1f}° x {fov_deg_y:.1f}°")
        else:
            fov_deg_x, fov_deg_y = _compute_spline_fov_from_seed_coverage(
                seed_model,
                all_poses_pnp,
                pnp_solved_mask,
                target_points,
                frames,
                max_fov_deg=175.0,
            )
            log(
                f"Spline FOV from seed: {fov_deg_x:.1f}° x {fov_deg_y:.1f}°"
            )

        prior_model = _build_initial_spline_model(
            seed_model,
            config,
            fov_deg_x,
            fov_deg_y,
        )

    else:
        prior_model = initial_camera_model
        log("Using user-provided initial spline model")
        seed_result = _fit_stereographic_seed(target_points, frames, config)
        all_poses_pnp, pnp_solved_mask, _ = _solve_pnp_all_frames_with_model(
            seed_result.camera_model,
            target_points,
            frames,
        )

    poses, inlier_masks, warp_coordinates = _prepare_spline_frame_state(
        all_poses_pnp,
        pnp_solved_mask,
        frames,
        target_points,
        estimate_target_warp,
    )

    def optimize_fn(
        batch: _OptimizationBatch[PinholeSplined],
    ) -> _OptimizationBatch[PinholeSplined]:
        return _pinhole_splined_refine_inner(
            batch,
            config,
            target_points,
            warp_coordinates,
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(prior_model, poses, frames, None, inlier_masks),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
        label="Spline",
    )

    target_warp = None
    if warp_coordinates is not None and state.warp_coeffs is not None:
        target_warp = TargetWarp(
            warp_coordinates=warp_coordinates, object_warp=state.warp_coeffs
        )
        deflection = target_warp.max_deflection(target_points)
        log(f"Target warp max deflection: {deflection:.4f} (target units)")

    diagnostics = _compute_frame_diagnostics(
        state.intrinsics,
        state.cameras_from_target,
        state.frames,
        target_points,
        inlier_masks=state.inlier_masks,
        target_warp=target_warp,
    )

    return CalibrationResult(
        camera_model=state.intrinsics,
        cameras_from_target=state.cameras_from_target,
        frame_diagnostics=diagnostics,
        frames=list(frames),
        target_points=target_points,
        target_warp=target_warp,
    )


def _calibrate_stereographic_splined(
    target_points: np.ndarray,
    frames: list[Frame],
    config: StereographicSplinedConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
    initial_camera_model: StereographicSplined | None = None,
) -> CalibrationResult[StereographicSplined]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )

    if initial_camera_model is None:
        seed_result = _fit_stereographic_seed(
            target_points,
            frames,
            config,
        )
        seed_model = seed_result.camera_model
        all_poses_pnp, pnp_solved_mask, _ = _solve_pnp_all_frames_with_model(
            seed_model,
            target_points,
            frames,
        )
        if config.fov_deg_xy is not None:
            fov_deg_x, fov_deg_y = config.fov_deg_xy
            log(
                f"Stereographic spline FOV (user-specified): "
                f"{fov_deg_x:.1f}° x {fov_deg_y:.1f}°"
            )
        else:
            fov_deg_x, fov_deg_y = _compute_spline_fov_from_seed_coverage(
                seed_model,
                all_poses_pnp,
                pnp_solved_mask,
                target_points,
                frames,
                max_fov_deg=220.0,
            )
            log(
                f"Stereographic spline FOV from seed: "
                f"{fov_deg_x:.1f}° x {fov_deg_y:.1f}°"
            )
        model = _build_initial_stereographic_spline_model(
            seed_model,
            config,
            fov_deg_x,
            fov_deg_y,
        )
    else:
        model = initial_camera_model
        seed_result = _fit_stereographic_seed(target_points, frames, config)
        all_poses_pnp, pnp_solved_mask, _ = _solve_pnp_all_frames_with_model(
            seed_result.camera_model,
            target_points,
            frames,
        )

    poses, inlier_masks, warp_coordinates = _prepare_spline_frame_state(
        all_poses_pnp,
        pnp_solved_mask,
        frames,
        target_points,
        estimate_target_warp,
    )
    warp_coeffs = None

    def optimize_fn(
        batch: _OptimizationBatch[StereographicSplined],
    ) -> _OptimizationBatch[StereographicSplined]:
        return _stereographic_splined_refine_inner(
            batch,
            config,
            target_points,
            warp_coordinates,
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(model, poses, frames, warp_coeffs, inlier_masks),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
        label="Stereographic spline",
    )

    target_warp = None
    if warp_coordinates is not None and state.warp_coeffs is not None:
        target_warp = TargetWarp(
            warp_coordinates=warp_coordinates,
            object_warp=state.warp_coeffs,
        )
        deflection = target_warp.max_deflection(target_points)
        log(f"Target warp max deflection: {deflection:.4f} (target units)")

    diagnostics = _compute_frame_diagnostics(
        state.intrinsics,
        state.cameras_from_target,
        frames,
        target_points,
        inlier_masks=state.inlier_masks,
        target_warp=target_warp,
    )

    return CalibrationResult(
        camera_model=state.intrinsics,
        cameras_from_target=state.cameras_from_target,
        frame_diagnostics=diagnostics,
        frames=list(frames),
        target_points=target_points,
        target_warp=target_warp,
    )


def _validate_opencv_initial_model_config(
    config: OpenCVConfig,
    initial_camera_model: OpenCV,
) -> None:
    """Validate that an OpenCV config could have produced an initial model.

    Args:
        config: Camera model configuration.
        initial_camera_model: Initial model to optimize from.
    """
    if config.image_width != initial_camera_model.image_width:
        raise ValueError(
            "Initial OpenCV model image_width is incompatible with config: "
            f"{initial_camera_model.image_width} != {config.image_width}"
        )
    if config.image_height != initial_camera_model.image_height:
        raise ValueError(
            "Initial OpenCV model image_height is incompatible with config: "
            f"{initial_camera_model.image_height} != {config.image_height}"
        )

    disabled = ~config.included_distortion_coefficients
    if not np.allclose(initial_camera_model.distortion_coeffs[disabled], 0.0):
        disabled_indices = np.where(disabled)[0].tolist()
        raise ValueError(
            "Initial OpenCV model has non-zero distortion coefficients that "
            f"are disabled by the config: {disabled_indices}"
        )


def _validate_spline_initial_model_config(
    config: PinholeSplinedConfig,
    initial_camera_model: PinholeSplined,
) -> None:
    """Validate that a spline config could have produced an initial model.

    Args:
        config: Camera model configuration.
        initial_camera_model: Initial model to optimize from.
    """
    if config.image_width != initial_camera_model.image_width:
        raise ValueError(
            "Initial spline model image_width is incompatible with config: "
            f"{initial_camera_model.image_width} != {config.image_width}"
        )
    if config.image_height != initial_camera_model.image_height:
        raise ValueError(
            "Initial spline model image_height is incompatible with config: "
            f"{initial_camera_model.image_height} != {config.image_height}"
        )

    if initial_camera_model.num_knots_x != config.num_knots_x:
        raise ValueError(
            "Initial spline num_knots_x is incompatible with config: "
            f"{initial_camera_model.num_knots_x} != {config.num_knots_x}"
        )
    if initial_camera_model.num_knots_y != config.num_knots_y:
        raise ValueError(
            "Initial spline num_knots_y is incompatible with config: "
            f"{initial_camera_model.num_knots_y} != {config.num_knots_y}"
        )

    expected_shape = (config.num_knots_y, config.num_knots_x)
    if initial_camera_model.dx_grid.shape != expected_shape:
        raise ValueError(
            "Initial spline dx_grid shape is incompatible with config: "
            f"{initial_camera_model.dx_grid.shape} != {expected_shape}"
        )
    if initial_camera_model.dy_grid.shape != expected_shape:
        raise ValueError(
            "Initial spline dy_grid shape is incompatible with config: "
            f"{initial_camera_model.dy_grid.shape} != {expected_shape}"
        )

    if config.fov_deg_xy is None:
        return

    fov_deg_x, fov_deg_y = config.fov_deg_xy
    if not np.isclose(initial_camera_model.fov_deg_x, fov_deg_x):
        raise ValueError(
            "Initial spline fov_deg_x is incompatible with config: "
            f"{initial_camera_model.fov_deg_x} != {fov_deg_x}"
        )
    if not np.isclose(initial_camera_model.fov_deg_y, fov_deg_y):
        raise ValueError(
            "Initial spline fov_deg_y is incompatible with config: "
            f"{initial_camera_model.fov_deg_y} != {fov_deg_y}"
        )


def _validate_stereographic_opencv_initial_model_config(
    config: StereographicOpenCVConfig,
    initial_camera_model: StereographicOpenCV,
) -> None:
    """Validate that a stereographic OpenCV config could have produced a model.

    Args:
        config: Camera model configuration.
        initial_camera_model: Initial model to optimize from.
    """
    if config.image_width != initial_camera_model.image_width:
        raise ValueError(
            "Initial stereographic OpenCV image_width is incompatible with config: "
            f"{initial_camera_model.image_width} != {config.image_width}"
        )
    if config.image_height != initial_camera_model.image_height:
        raise ValueError(
            "Initial stereographic OpenCV image_height is incompatible with config: "
            f"{initial_camera_model.image_height} != {config.image_height}"
        )
    disabled = ~config.included_distortion_coefficients
    if not np.allclose(initial_camera_model.distortion_coeffs[disabled], 0.0):
        disabled_indices = np.where(disabled)[0].tolist()
        raise ValueError(
            "Initial stereographic OpenCV model has non-zero distortion "
            f"coefficients that are disabled by the config: {disabled_indices}"
        )


def _validate_stereographic_spline_initial_model_config(
    config: StereographicSplinedConfig,
    initial_camera_model: StereographicSplined,
) -> None:
    """Validate that a stereographic spline config could have produced a model.

    Args:
        config: Camera model configuration.
        initial_camera_model: Initial model to optimize from.
    """
    if config.image_width != initial_camera_model.image_width:
        raise ValueError(
            "Initial stereographic spline image_width is incompatible with config: "
            f"{initial_camera_model.image_width} != {config.image_width}"
        )
    if config.image_height != initial_camera_model.image_height:
        raise ValueError(
            "Initial stereographic spline image_height is incompatible with config: "
            f"{initial_camera_model.image_height} != {config.image_height}"
        )
    if initial_camera_model.num_knots_x != config.num_knots_x:
        raise ValueError(
            "Initial stereographic spline num_knots_x is incompatible with config: "
            f"{initial_camera_model.num_knots_x} != {config.num_knots_x}"
        )
    if initial_camera_model.num_knots_y != config.num_knots_y:
        raise ValueError(
            "Initial stereographic spline num_knots_y is incompatible with config: "
            f"{initial_camera_model.num_knots_y} != {config.num_knots_y}"
        )

    expected_shape = (config.num_knots_y, config.num_knots_x)
    if initial_camera_model.dx_grid.shape != expected_shape:
        raise ValueError(
            "Initial stereographic spline dx_grid shape is incompatible with config: "
            f"{initial_camera_model.dx_grid.shape} != {expected_shape}"
        )
    if initial_camera_model.dy_grid.shape != expected_shape:
        raise ValueError(
            "Initial stereographic spline dy_grid shape is incompatible with config: "
            f"{initial_camera_model.dy_grid.shape} != {expected_shape}"
        )

    if config.fov_deg_xy is None:
        return

    fov_deg_x, fov_deg_y = config.fov_deg_xy
    if not np.isclose(initial_camera_model.fov_deg_x, fov_deg_x):
        raise ValueError(
            "Initial stereographic spline fov_deg_x is incompatible with config: "
            f"{initial_camera_model.fov_deg_x} != {fov_deg_x}"
        )
    if not np.isclose(initial_camera_model.fov_deg_y, fov_deg_y):
        raise ValueError(
            "Initial stereographic spline fov_deg_y is incompatible with config: "
            f"{initial_camera_model.fov_deg_y} != {fov_deg_y}"
        )


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: PinholeSplinedConfig,
    *,
    initial_camera_model: PinholeSplined | None = None,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[PinholeSplined]: ...


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: OpenCVConfig,
    *,
    initial_camera_model: OpenCV | None = None,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[OpenCV]: ...


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: StereographicOpenCVConfig,
    *,
    initial_camera_model: StereographicOpenCV | None = None,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[StereographicOpenCV]: ...


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: StereographicSplinedConfig,
    *,
    initial_camera_model: StereographicSplined | None = None,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[StereographicSplined]: ...


def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: CameraModelConfig,
    *,
    initial_camera_model: CameraModel | None = None,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult:
    """Calibrate a camera from a set of per-image frames.

    Target warp estimation requires a planar target; it will be skipped
    automatically if the target points are not sufficiently coplanar.

    Args:
        target_points: 3D target point coordinates, shape (N, 3).
        frames: Per-image frames, one per calibration image.
        camera_model_config: Specifies the camera model to fit.
        initial_camera_model: Optional initial model to optimize from.
        estimate_target_warp: Whether to estimate a Legendre-polynomial warp
            of the target to account for slight non-planarity.
        outlier_threshold_stddevs: Sigma threshold for outlier rejection.
            Pass None to disable.

    Returns:
        Calibration result containing the optimised model and per-image diagnostics.
    """
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    if isinstance(camera_model_config, PinholeSplinedConfig):
        if initial_camera_model is not None:
            if not isinstance(initial_camera_model, PinholeSplined):
                raise TypeError(
                    "initial_camera_model must be a PinholeSplined when "
                    "camera_model_config is a PinholeSplinedConfig"
                )
            _validate_spline_initial_model_config(
                camera_model_config, initial_camera_model
            )
        return _calibrate_pinhole_splined(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
            initial_camera_model,
        )

    if isinstance(camera_model_config, OpenCVConfig):
        if initial_camera_model is not None:
            if not isinstance(initial_camera_model, OpenCV):
                raise TypeError(
                    "initial_camera_model must be an OpenCV when "
                    "camera_model_config is an OpenCVConfig"
                )
            _validate_opencv_initial_model_config(
                camera_model_config, initial_camera_model
            )
        return _opencv_calibrate(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
            initial_camera_model,
        )

    if isinstance(camera_model_config, StereographicOpenCVConfig):
        if initial_camera_model is not None:
            if not isinstance(initial_camera_model, StereographicOpenCV):
                raise TypeError(
                    "initial_camera_model must be a StereographicOpenCV when "
                    "camera_model_config is a StereographicOpenCVConfig"
                )
            _validate_stereographic_opencv_initial_model_config(
                camera_model_config,
                initial_camera_model,
            )
        return _calibrate_stereographic_opencv(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
            initial_camera_model,
        )

    if isinstance(camera_model_config, StereographicSplinedConfig):
        if initial_camera_model is not None:
            if not isinstance(initial_camera_model, StereographicSplined):
                raise TypeError(
                    "initial_camera_model must be a StereographicSplined when "
                    "camera_model_config is a StereographicSplinedConfig"
                )
            _validate_stereographic_spline_initial_model_config(
                camera_model_config,
                initial_camera_model,
            )
        return _calibrate_stereographic_splined(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
            initial_camera_model,
        )

    raise RuntimeError("Invalid config")
