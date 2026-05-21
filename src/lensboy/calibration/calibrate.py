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
from lensboy.camera_models.base_model import CameraModelConfig
from lensboy.camera_models.opencv import OpenCV, OpenCVConfig
from lensboy.camera_models.pinhole_splined import (
    PinholeSplined,
    PinholeSplinedConfig,
)
from lensboy.geometry.pose import Pose

DEFAULT_OUTLIER_THRESHOLD = 5.0
MAX_OUTLIER_FILTER_PASSES = 2


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


def _project_and_calculate_residuals(
    target_points: np.ndarray,
    camera_from_target: Pose,
    frame: Frame,
    model: OpenCV | PinholeSplined,
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

    result = lbb.calibrate_opencv(
        intrinsics_initial_value=params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_target=[p._to_cpp() for p in batch.cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in batch.frames],
        warp_coordinates=(
            warp_coordinates._to_cpp() if warp_coordinates is not None else None
        ),
        warp_coeffs_initial=(
            list(batch.warp_coeffs) if batch.warp_coeffs is not None else [0.0] * 5
        ),
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
    intrinsics: OpenCV | PinholeSplined,
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
        mean_reproj, worst_reproj = _compute_mean_reproj(
            state,
            target_points,
            curr_target_warp,
        )
        log(
            f"{label} pass {pass_num}: {elapsed:.1f}s "
            f"(mean reproj={mean_reproj:.3f}px, worst={worst_reproj:.3f}px)"
        )

        if outlier_threshold_stddevs is None or iteration == MAX_OUTLIER_FILTER_PASSES:
            break

        # Compute residuals and update masks
        residuals = []
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

    mean_error = total_squared_error / total_points if total_points > 0 else float("inf")
    return poses, solved, mean_error


def _get_initial_state_with_pnp(
    config: OpenCVConfig,
    target_points: np.ndarray,
    frames: list[Frame],
) -> tuple[OpenCV, list[Pose], list[bool]]:
    """Estimate initial intrinsics and poses using PnP.

    If ``config.initial_focal_length`` is set, uses that value directly.
    Otherwise, sweeps log-spaced candidate focal lengths and picks the one
    with the lowest total reprojection error.

    Args:
        config: Camera model configuration.
        target_points: Calibration target 3D points, shape (N, 3).
        frames: Detected calibration frames.

    Returns:
        Tuple of (initial intrinsics, poses, solved mask).
    """
    cx = config.image_width / 2.0
    cy = config.image_height / 2.0

    if config.initial_focal_length is not None:
        intrinsics = config.get_initial_value()
        poses, solved, _ = _solve_pnp_all_frames(intrinsics.K(), target_points, frames)
        return intrinsics, poses, solved

    max_dim = max(config.image_width, config.image_height)
    candidates = np.geomspace(0.2 * max_dim, 5.0 * max_dim, num=30)

    best_focal = float(candidates[0])
    best_error = float("inf")
    best_poses: list[Pose] = []
    best_solved: list[bool] = []

    for f in candidates:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        poses, solved, error = _solve_pnp_all_frames(K, target_points, frames)
        if error < best_error:
            best_error = error
            best_focal = float(f)
            best_poses = poses
            best_solved = solved

    log(f"Auto-estimated initial focal length: {best_focal:.1f} px")

    intrinsics = OpenCV(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=best_focal,
        fy=best_focal,
        cx=cx,
        cy=cy,
        distortion_coeffs=np.zeros(14, dtype=np.float64),
    )
    return intrinsics, best_poses, best_solved


_PLANARITY_RATIO_THRESHOLD = 0.1
_RECT_FIT_RATIO_THRESHOLD = 0.85


def _make_warp_coordinates(target_points: np.ndarray) -> WarpCoordinates | None:
    centroid = target_points.mean(axis=0)
    centered = target_points - centroid
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    planarity_ratio = s[2] / s[1] if s[1] > 1e-10 else np.inf
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
    optimize_fn: Callable[[_OptimizationBatch[OpenCV]], _OptimizationBatch[OpenCV]],
    initial_intrinsics: OpenCV,
    initial_poses: list[Pose],
    pnp_solved: list[bool],
    target_points: np.ndarray,
    frames: list[Frame],
    image_width: int,
    image_height: int,
) -> tuple[OpenCV, list[Pose | None], list[np.ndarray | None]]:
    """Fit a subsampled model and re-run PnP to recover initially failed frames.

    Args:
        optimize_fn: Optimizer callback.
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
    re_poses, re_solved, _ = _solve_pnp_all_frames(
        model.K(),
        target_points,
        frames,
        model.distortion_coeffs,
    )
    n_re_solved = sum(re_solved)
    log(f"Re-PnP solved {n_re_solved}/{len(frames)} frames")

    poses: list[Pose | None] = [p if ok else None for p, ok in zip(re_poses, re_solved)]
    inlier_masks: list[np.ndarray | None] = [
        np.ones(len(f), dtype=bool) if ok else None for f, ok in zip(frames, re_solved)
    ]
    return model, poses, inlier_masks


def _opencv_calibrate(
    target_points: np.ndarray,
    frames: list[Frame],
    config: OpenCVConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
) -> CalibrationResult[OpenCV]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    log("Computing initial poses with PnP...")
    initial_intrinsics, initial_poses, pnp_solved = _get_initial_state_with_pnp(
        config, target_points, frames
    )

    n_solved = sum(pnp_solved)
    n_failed = len(frames) - n_solved
    log(f"PnP solved {n_solved}/{len(frames)} frames")
    if n_failed > 0:
        log(f"{n_failed} frame(s) failed PnP, excluding from optimization")

    poses: list[Pose | None] = [
        p if ok else None for p, ok in zip(initial_poses, pnp_solved)
    ]
    inlier_masks: list[np.ndarray | None] = [
        np.ones(len(f), dtype=bool) if ok else None for f, ok in zip(frames, pnp_solved)
    ]

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(batch: _OptimizationBatch[OpenCV]) -> _OptimizationBatch[OpenCV]:
        return _opencv_calibrate_inner(batch, config, target_points, warp_coordinates)

    if n_failed > 0:
        initial_intrinsics, poses, inlier_masks = _recover_failed_pnp(
            optimize_fn,
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


def _pinhole_splined_refine_inner(
    batch: _OptimizationBatch[PinholeSplined],
    target_points: np.ndarray,
    warp_coordinates: WarpCoordinates | None,
) -> _OptimizationBatch[PinholeSplined]:
    fine_tune_result = lbb.fine_tune_pinhole_splined(
        model_config=batch.intrinsics._cpp_config(),
        intrinsics_parameters=batch.intrinsics._cpp_params(),
        cameras_from_target=[pose._to_cpp() for pose in batch.cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in batch.frames],
        warp_coordinates=(
            warp_coordinates._to_cpp() if warp_coordinates is not None else None
        ),
        warp_coeffs_initial=(
            list(batch.warp_coeffs) if batch.warp_coeffs is not None else [0.0] * 5
        ),
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


def _compute_fov_from_opencv(
    opencv_model: OpenCV,
    padding_fraction: float = 0.05,
    max_fov_deg: float = 175.0,
) -> tuple[float, float]:
    """Compute the spline FOV from an OpenCV model with percentage padding.

    Args:
        opencv_model: Seed OpenCV camera model.
        padding_fraction: Fractional padding to add to each FOV axis.
        max_fov_deg: Maximum allowed FOV to avoid singularities near 180 degrees.

    Returns:
        Padded (fov_deg_x, fov_deg_y), capped at max_fov_deg.
    """
    return (
        min(opencv_model.fov_deg_x * (1 + padding_fraction), max_fov_deg),
        min(opencv_model.fov_deg_y * (1 + padding_fraction), max_fov_deg),
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


def _compute_fov_from_spline_model(
    model: PinholeSplined,
    padding_fraction: float = 0.05,
    max_fov_deg: float = 175.0,
) -> tuple[float, float]:
    """Compute FOV by unprojecting the image corners through the spline model.

    Args:
        model: Fitted spline model.
        padding_fraction: Fractional padding to add to each FOV axis.
        max_fov_deg: Maximum allowed FOV.

    Returns:
        Padded (fov_deg_x, fov_deg_y), capped at max_fov_deg.
    """
    w, h = float(model.image_width), float(model.image_height)
    n = 50
    t = np.linspace(0, 1, n)
    edges = np.concatenate(
        [
            np.column_stack([t * w, np.zeros(n)]),  # top
            np.column_stack([t * w, np.full(n, h)]),  # bottom
            np.column_stack([np.zeros(n), t * h]),  # left
            np.column_stack([np.full(n, w), t * h]),  # right
        ]
    )
    normalized = model.normalize_points(edges)
    half_x = float(np.abs(normalized[:, 0]).max())
    half_y = float(np.abs(normalized[:, 1]).max())

    fov_x = float(np.degrees(2 * np.arctan(half_x * (1 + padding_fraction))))
    fov_y = float(np.degrees(2 * np.arctan(half_y * (1 + padding_fraction))))

    return (min(fov_x, max_fov_deg), min(fov_y, max_fov_deg))


def _fit_opencv_seed(
    target_points: np.ndarray,
    frames: list[Frame],
    config: PinholeSplinedConfig,
) -> tuple[CalibrationResult[OpenCV], list[int]]:
    """Fit an OpenCV seed model on a coverage-subsampled set of frames.

    Args:
        target_points: 3D target points, shape (N, 3).
        frames: All calibration frames.
        config: Spline config (used for image size and initial focal length).

    Returns:
        Tuple of (calibration result, indices of subsampled frames).
    """
    start_time = default_timer()
    sub_indices = _select_covering_frames(frames, config.image_width, config.image_height)
    subsampled_frames = [frames[i] for i in sub_indices]

    opencv_config = OpenCVConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        initial_focal_length=config.initial_focal_length,
        included_distortion_coefficients=OpenCVConfig.FULL_14,
    )

    disable_logs()
    try:
        result = _opencv_calibrate(
            target_points,
            subsampled_frames,
            opencv_config,
            None,
            estimate_target_warp=False,
        )
    finally:
        enable_logs()
    fov_x, fov_y = _compute_fov_from_opencv(result.camera_model, padding_fraction=0.0)
    log(
        f"Fitted OpenCV seed model: {default_timer() - start_time:.1f}s "
        f"(FOV: {fov_x:.1f}° x {fov_y:.1f}°)"
    )
    return result, sub_indices


def _estimate_spline_fov(
    opencv_model: OpenCV,
    opencv_result: CalibrationResult[OpenCV],
    frames: list[Frame],
    sub_indices: list[int],
    target_points: np.ndarray,
    config: PinholeSplinedConfig,
    opencv_fov_x: float,
    opencv_fov_y: float,
) -> tuple[float, float]:
    """Estimate the spline FOV by fitting a coarse spline and unprojecting image edges.

    Args:
        opencv_model: Seed OpenCV model.
        opencv_result: Calibration result from the seed model.
        frames: All calibration frames.
        sub_indices: Indices of subsampled frames used for the seed model.
        target_points: 3D target points, shape (N, 3).
        config: Spline config.
        opencv_fov_x: OpenCV model's FOV in x (no padding).
        opencv_fov_y: OpenCV model's FOV in y (no padding).

    Returns:
        Estimated (fov_deg_x, fov_deg_y) for the full spline model.
    """
    coarse_fov_x, coarse_fov_y = _compute_fov_from_opencv(
        opencv_model, padding_fraction=0.30
    )

    coarse_nx = min(config.num_knots_x, 10)
    coarse_ny = min(config.num_knots_y, 8)
    coarse_cpp_config = lbb.PinholeSplinedConfig(
        config.image_width,
        config.image_height,
        coarse_fov_x,
        coarse_fov_y,
        coarse_nx,
        coarse_ny,
        1.0,  # strong smoothness for coarse model
    )

    coarse_image_bound_x = np.tan(np.deg2rad(opencv_fov_x) / 2.0) * 0.8
    coarse_image_bound_y = np.tan(np.deg2rad(opencv_fov_y) / 2.0) * 0.8
    coarse_out = lbb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(),
        coarse_cpp_config,
        float(coarse_image_bound_x),
        float(coarse_image_bound_y),
    )

    coarse_model = PinholeSplined(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=opencv_model.fx,
        fy=opencv_model.fy,
        cx=opencv_model.cx,
        cy=opencv_model.cy,
        dx_grid=coarse_out["x_knots"],
        dy_grid=coarse_out["y_knots"],
        num_knots_x=coarse_nx,
        num_knots_y=coarse_ny,
        fov_deg_x=coarse_fov_x,
        fov_deg_y=coarse_fov_y,
        smoothness_lambda=1.0,
    )

    all_pnp = opencv_result.cameras_from_target
    sub_frames = [frames[i] for i in sub_indices]
    solved_poses = [p for p in all_pnp if p is not None]
    solved_frames = [f for f, p in zip(sub_frames, all_pnp) if p is not None]

    start_time = default_timer()
    coarse_result = _pinhole_splined_refine_inner(
        _OptimizationBatch(
            intrinsics=coarse_model,
            cameras_from_target=solved_poses,
            frames=solved_frames,
            warp_coeffs=None,
        ),
        target_points,
        None,
    )

    fov_deg_x, fov_deg_y = _compute_fov_from_spline_model(
        coarse_result.intrinsics, padding_fraction=0.05
    )
    log(
        f"Spline FOV estimate: {default_timer() - start_time:.1f}s "
        f"({fov_deg_x:.1f}° x {fov_deg_y:.1f}°)"
    )
    return fov_deg_x, fov_deg_y


def _build_initial_spline_model(
    opencv_model: OpenCV,
    config: PinholeSplinedConfig,
    fov_deg_x: float,
    fov_deg_y: float,
    opencv_fov_x: float,
    opencv_fov_y: float,
) -> PinholeSplined:
    """Build the initial spline model by matching the OpenCV distortion.

    Args:
        opencv_model: Seed OpenCV model.
        config: Spline config.
        fov_deg_x: Target FOV in x for the spline grid.
        fov_deg_y: Target FOV in y for the spline grid.
        opencv_fov_x: OpenCV model's FOV in x (no padding).
        opencv_fov_y: OpenCV model's FOV in y (no padding).

    Returns:
        Initial PinholeSplined model with knots matched to the OpenCV distortion.
    """
    cpp_config = lbb.PinholeSplinedConfig(
        config.image_width,
        config.image_height,
        fov_deg_x,
        fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
        config.smoothness_lambda,
    )

    matching_bounds_fov_x = min(fov_deg_x, opencv_fov_x)
    matching_bounds_fov_y = min(fov_deg_y, opencv_fov_y)
    image_bound_x = np.tan(np.deg2rad(matching_bounds_fov_x) / 2.0) * 0.8
    image_bound_y = np.tan(np.deg2rad(matching_bounds_fov_y) / 2.0) * 0.8

    out_dict = lbb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(),
        cpp_config,
        float(image_bound_x),
        float(image_bound_y),
    )

    return PinholeSplined(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=opencv_model.fx,
        fy=opencv_model.fy,
        cx=opencv_model.cx,
        cy=opencv_model.cy,
        dx_grid=out_dict["x_knots"],
        dy_grid=out_dict["y_knots"],
        num_knots_x=config.num_knots_x,
        num_knots_y=config.num_knots_y,
        fov_deg_x=fov_deg_x,
        fov_deg_y=fov_deg_y,
        smoothness_lambda=config.smoothness_lambda,
    )


def _calibrate_pinhole_splined(
    target_points: np.ndarray,
    frames: list[Frame],
    config: PinholeSplinedConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
) -> CalibrationResult[PinholeSplined]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )

    # Stage 1: Fit OpenCV seed on subsampled frames
    opencv_result, sub_indices = _fit_opencv_seed(target_points, frames, config)
    opencv_model = opencv_result.camera_model

    opencv_fov_x, opencv_fov_y = _compute_fov_from_opencv(
        opencv_model, padding_fraction=0.0
    )

    # Stage 2: Determine spline FOV
    if config.fov_deg_xy is not None:
        fov_deg_x, fov_deg_y = config.fov_deg_xy
        log(f"Spline FOV (user-specified): {fov_deg_x:.1f}° x {fov_deg_y:.1f}°")
    else:
        fov_deg_x, fov_deg_y = _estimate_spline_fov(
            opencv_model,
            opencv_result,
            frames,
            sub_indices,
            target_points,
            config,
            opencv_fov_x,
            opencv_fov_y,
        )

    # Stage 3: Build and fit full spline model
    prior_model = _build_initial_spline_model(
        opencv_model,
        config,
        fov_deg_x,
        fov_deg_y,
        opencv_fov_x,
        opencv_fov_y,
    )

    # PnP for all frames using the opencv model
    all_poses_pnp, pnp_solved_mask, _ = _solve_pnp_all_frames(
        opencv_model.K(),
        target_points,
        frames,
        dist_coeffs=opencv_model.distortion_coeffs,
    )
    n_solved = sum(pnp_solved_mask)
    log(f"PnP solved {n_solved}/{len(frames)} frames")

    poses: list[Pose | None] = [
        p if ok else None for p, ok in zip(all_poses_pnp, pnp_solved_mask)
    ]
    inlier_masks: list[np.ndarray | None] = [
        np.ones(len(f), dtype=bool) if ok else None
        for f, ok in zip(frames, pnp_solved_mask)
    ]

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(
        batch: _OptimizationBatch[PinholeSplined],
    ) -> _OptimizationBatch[PinholeSplined]:
        return _pinhole_splined_refine_inner(batch, target_points, warp_coordinates)

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


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: PinholeSplinedConfig,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[PinholeSplined]: ...


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: OpenCVConfig,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[OpenCV]: ...


def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: CameraModelConfig,
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
        return _calibrate_pinhole_splined(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
        )

    if isinstance(camera_model_config, OpenCVConfig):
        return _opencv_calibrate(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
        )

    raise RuntimeError("Invalid config")
