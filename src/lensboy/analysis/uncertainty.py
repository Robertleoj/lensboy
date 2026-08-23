from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.opencv import OpenCV
from lensboy.camera_models.pinhole_splined import PinholeSplined

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from lensboy.calibration.type_defs import CalibrationResult, Frame
    from lensboy.geometry.pose import Pose


@dataclass
class ProjectionUncertainty:
    """Projection uncertainty for camera rays.

    Attributes:
        rays: Query camera-frame rays or points, shape (N, 3).
        covariances_px: Pixel-space output covariance, shape (N, 2, 2).
        trace_std_px: Scalar summary, sqrt(trace(covariance)), shape (N,).
        metadata: Diagnostic information about grouping, damping, and Hessian solves.
    """

    rays: np.ndarray
    covariances_px: np.ndarray
    trace_std_px: np.ndarray
    metadata: dict

    @property
    def stddev_px(self) -> np.ndarray:
        """Return scalar standard deviations for scalar-compatible outputs.

        Returns:
            Square root of the covariance diagonal for each query, shape (N, 2).
        """
        diag = np.diagonal(self.covariances_px, axis1=1, axis2=2)
        return np.sqrt(np.maximum(diag, 0.0))


def _active_calibration_data(
    result: CalibrationResult,
) -> tuple[list[Pose], list[Frame]]:
    cameras_from_target = []
    frames = []
    for camera_from_target, frame, diagnostics in zip(
        result.cameras_from_target,
        result.frames,
        result.frame_diagnostics,
    ):
        if camera_from_target is None or diagnostics is None:
            continue

        mask = diagnostics.inlier_mask
        if not np.any(mask):
            continue

        cameras_from_target.append(camera_from_target)
        frames.append(
            type(frame)(
                target_point_indices=frame.target_point_indices[mask],
                detected_points_in_image=frame.detected_points_in_image[mask],
            )
        )

    if not frames:
        raise ValueError("Projection uncertainty needs at least one solved inlier frame.")
    return cameras_from_target, frames


def compute_projection_uncertainty(
    result: CalibrationResult,
    rays: np.ndarray,
    *,
    damping: float = 1e-9,
    relative_eigen_floor: float = 1e-12,
    spline_smoothness_lambda: float = 1.0,
) -> ProjectionUncertainty:
    """Estimate output-space projection uncertainty for camera rays.

    Args:
        result: Calibration result containing the fitted model and inlier masks.
        rays: Query camera-frame rays or points, shape (N, 3).
        damping: Absolute eigenvalue floor used in Hessian solves.
        relative_eigen_floor: Relative eigenvalue floor used in Hessian solves.
        spline_smoothness_lambda: Smoothness prior used for spline Hessian regularization.

    Returns:
        Projection uncertainty with full 2D pixel covariance per ray.
    """
    query_rays = np.ascontiguousarray(rays, dtype=np.float64)
    if query_rays.ndim != 2 or query_rays.shape[1] != 3:
        raise ValueError(f"rays must have shape (N, 3), got {query_rays.shape}")

    cameras_from_target, frames = _active_calibration_data(result)
    warp_coordinates = None
    warp_coeffs = [0.0] * 5
    if result.target_warp is not None:
        warp_coordinates = result.target_warp.warp_coordinates._to_cpp()
        warp_coeffs = list(result.target_warp.object_warp)

    model = result.camera_model
    if isinstance(model, OpenCV):
        native = lbb.projection_uncertainty_opencv(
            intrinsics=model._params(),
            cameras_from_target=[pose._to_cpp() for pose in cameras_from_target],
            target_points=list(result.target_points),
            frames=[frame._to_cpp() for frame in frames],
            warp_coordinates=warp_coordinates,
            warp_coeffs_initial=warp_coeffs,
            query_rays=query_rays,
            damping=damping,
            relative_eigen_floor=relative_eigen_floor,
        )
    elif isinstance(model, PinholeSplined):
        native = lbb.projection_uncertainty_pinhole_splined(
            model_config=lbb.PinholeSplinedOptimizationConfig(
                model.image_width,
                model.image_height,
                model.fov_deg_x,
                model.fov_deg_y,
                model.num_knots_x,
                model.num_knots_y,
                spline_smoothness_lambda,
            ),
            intrinsics_parameters=model._cpp_params(),
            cameras_from_target=[pose._to_cpp() for pose in cameras_from_target],
            target_points=list(result.target_points),
            frames=[frame._to_cpp() for frame in frames],
            warp_coordinates=warp_coordinates,
            warp_coeffs_initial=warp_coeffs,
            query_rays=query_rays,
            damping=damping,
            relative_eigen_floor=relative_eigen_floor,
        )
    else:
        raise TypeError(f"Unsupported camera model type: {type(model)!r}")

    return ProjectionUncertainty(
        rays=query_rays,
        covariances_px=np.asarray(native["covariances_px"], dtype=np.float64),
        trace_std_px=np.asarray(native["trace_std_px"], dtype=np.float64),
        metadata=dict(native["metadata"]),
    )


def plot_projection_uncertainty(
    result: CalibrationResult,
    *,
    heatmap_max: float | None = None,
    grid_density: int = 200,
    return_figure: bool = False,
    damping: float = 1e-9,
    relative_eigen_floor: float = 1e-12,
    spline_smoothness_lambda: float = 1.0,
) -> Figure | None:
    """Plot a pixel-space heatmap of ray projection uncertainty.

    Args:
        result: Calibration result to analyse.
        heatmap_max: Color scale ceiling in pixels. Auto-scaled if None.
        grid_density: Number of grid samples along the longer image axis.
        return_figure: If True, return the figure instead of showing it.
        damping: Absolute eigenvalue floor used in Hessian solves.
        relative_eigen_floor: Relative eigenvalue floor used in Hessian solves.
        spline_smoothness_lambda: Smoothness prior used for spline Hessian regularization.

    Returns:
        The figure if requested, otherwise None.
    """
    from lensboy.analysis.plots import _plot_projection_uncertainty

    model = result.camera_model
    w, h = model.image_width, model.image_height
    aspect = w / h
    if w >= h:
        nx = grid_density
        ny = max(2, round(grid_density / aspect))
    else:
        ny = grid_density
        nx = max(2, round(grid_density * aspect))

    x = np.linspace(0, w - 1, nx)
    y = np.linspace(0, h - 1, ny)
    xx, yy = np.meshgrid(x, y)
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    rays = model.normalize_points(pixels)
    uncertainty = compute_projection_uncertainty(
        result,
        rays,
        damping=damping,
        relative_eigen_floor=relative_eigen_floor,
        spline_smoothness_lambda=spline_smoothness_lambda,
    )
    return _plot_projection_uncertainty(
        uncertainty.trace_std_px.reshape(ny, nx),
        image_width=w,
        image_height=h,
        heatmap_max=heatmap_max,
        return_figure=return_figure,
    )
