from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lensboy.camera_models.unproject_lut import UnprojectLUT

if TYPE_CHECKING:
    from lensboy.camera_models.base_model import CameraModel

_SUPPORTED_INTERPOLATIONS: tuple[str, ...] = (
    "nearest",
    "bilinear",
    "bicubic",
)
# Per-cell gradient-ascent tuning. Hidden from the user — these are tight
# enough that the optimiser converges on every cell of any reasonable LUT,
# and loosening them is never a useful knob in practice.
_OPTIMISER_MAX_ITERS = 50
_OPTIMISER_GRAD_TOL = 1.0e-12
_ANGULAR_ERROR_MDEG_SCALE = 1.0e3
_INTERP_TO_CPP_MODE: dict[str, int] = {
    "nearest": 0,
    "bilinear": 1,
    "bicubic": 2,
}


def _validate_interpolation_mode(interpolation: str) -> str:
    if interpolation not in _SUPPORTED_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported interpolation mode {interpolation!r}. "
            f"Expected one of {_SUPPORTED_INTERPOLATIONS}."
        )
    return interpolation


def _normalize_interpolations(
    interpolations: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(interpolations, str):
        raw_items = [interpolations]
    else:
        raw_items = list(interpolations)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        mode = _validate_interpolation_mode(item)
        if mode in seen:
            continue
        normalized.append(mode)
        seen.add(mode)
    if len(normalized) == 0:
        raise ValueError("interpolations must be non-empty.")
    return tuple(normalized)


@dataclass(frozen=True)
class UnprojectLUTAccuracyReport:
    """Accuracy summary for one or more LUT interpolation modes.

    Args:
        interpolations: Interpolation modes included in the report. Each entry
            is one of ``"nearest"``, ``"bilinear"``, ``"bicubic"``.
        max_angular_error_mdeg: Observed maximum angular error per interpolation mode.

    Returns:
        Immutable report describing the requested interpolation modes.
    """

    interpolations: tuple[str, ...]
    max_angular_error_mdeg: dict[str, float]


@dataclass
class UnprojectLUTErrorHeatmap:
    """Per-cell angular-error heatmap for a LUT interpolation mode.

    Stores three primary 2D arrays per cell — the pixel where the worst
    angular error sits, and the exact and LUT-approximated normalised xy
    rays at that pixel (z=1 is implied). Angular error, residual delta,
    and unit residual direction are derivable from the rays and exposed
    as properties.

    Args:
        interpolation: Interpolation mode represented by the heatmap. One of
            ``"nearest"``, ``"bilinear"``, ``"bicubic"``.
        cell_x_edges: Pixel-x boundaries between cells, shape ``(grid_width,)``.
            Cell ``ix`` spans ``[cell_x_edges[ix], cell_x_edges[ix + 1]]``.
        cell_y_edges: Pixel-y boundaries between cells, shape ``(grid_height,)``.
            Cell ``iy`` spans ``[cell_y_edges[iy], cell_y_edges[iy + 1]]``.
        peak_pixel_xy: Pixel location of the local peak error, shape ``(H, W, 2)``.
        exact_xy: Camera-model normalised xy at the peak pixel, shape
            ``(H, W, 2)``. The full ray is ``(exact_x, exact_y, 1)``.
        approx_xy: LUT-interpolated normalised xy at the peak pixel, shape
            ``(H, W, 2)``. The full ray is ``(approx_x, approx_y, 1)``.

    Returns:
        In-memory representation of a saved or computed heatmap.
    """

    interpolation: str
    cell_x_edges: np.ndarray
    cell_y_edges: np.ndarray
    peak_pixel_xy: np.ndarray
    exact_xy: np.ndarray
    approx_xy: np.ndarray

    def __post_init__(self) -> None:
        self.interpolation = _validate_interpolation_mode(self.interpolation)
        self.cell_x_edges = np.asarray(self.cell_x_edges, dtype=np.float64).copy()
        self.cell_y_edges = np.asarray(self.cell_y_edges, dtype=np.float64).copy()
        self.peak_pixel_xy = np.asarray(self.peak_pixel_xy, dtype=np.float64).copy()
        self.exact_xy = np.asarray(self.exact_xy, dtype=np.float64).copy()
        self.approx_xy = np.asarray(self.approx_xy, dtype=np.float64).copy()

        if self.peak_pixel_xy.ndim != 3 or self.peak_pixel_xy.shape[-1] != 2:
            raise ValueError(
                "peak_pixel_xy must have shape (H, W, 2), "
                f"got {self.peak_pixel_xy.shape}."
            )
        expected_shape = self.peak_pixel_xy.shape
        if self.exact_xy.shape != expected_shape:
            raise ValueError(
                f"exact_xy must have shape {expected_shape}, got {self.exact_xy.shape}."
            )
        if self.approx_xy.shape != expected_shape:
            raise ValueError(
                f"approx_xy must have shape {expected_shape}, "
                f"got {self.approx_xy.shape}."
            )

    @property
    def error_delta_xy(self) -> np.ndarray:
        """Per-cell residual ``approx_xy − exact_xy``, shape ``(H, W, 2)``."""
        return self.approx_xy - self.exact_xy

    @property
    def error_direction_xy(self) -> np.ndarray:
        """Unit-length residual direction, shape ``(H, W, 2)``.

        Zero where the residual is exactly zero.
        """
        delta = self.error_delta_xy
        delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        return np.divide(
            delta,
            delta_norm,
            out=np.zeros_like(delta),
            where=delta_norm > 0.0,
        )

    @property
    def max_angular_error_deg(self) -> np.ndarray:
        """Per-cell maximum angular error in degrees, shape ``(H, W)``.

        Computed as ``atan2(‖cross‖, dot)`` between the rays
        ``(exact_xy, 1)`` and ``(approx_xy, 1)``.
        """
        ex = self.exact_xy[..., 0]
        ey = self.exact_xy[..., 1]
        ax = self.approx_xy[..., 0]
        ay = self.approx_xy[..., 1]
        cross_x = ey - ay
        cross_y = ax - ex
        cross_z = ex * ay - ey * ax
        cross_norm = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
        dot = ex * ax + ey * ay + 1.0
        return np.rad2deg(np.arctan2(cross_norm, dot))

    def save(self, path: Path | str) -> None:
        """Save the heatmap to a compressed NumPy archive.

        Args:
            path: Output `.npz` path.

        Returns:
            None.
        """
        np.savez_compressed(
            Path(path),
            interpolation=np.array(self.interpolation),
            cell_x_edges=self.cell_x_edges,
            cell_y_edges=self.cell_y_edges,
            peak_pixel_xy=self.peak_pixel_xy,
            exact_xy=self.exact_xy,
            approx_xy=self.approx_xy,
        )

    @staticmethod
    def load(path: Path | str) -> UnprojectLUTErrorHeatmap:
        """Load a saved heatmap archive.

        Args:
            path: Path to a `.npz` archive written by ``save()``.

        Returns:
            Loaded heatmap object.
        """
        with np.load(Path(path)) as heatmap_data:
            return UnprojectLUTErrorHeatmap(
                interpolation=str(np.asarray(heatmap_data["interpolation"]).item()),
                cell_x_edges=np.asarray(heatmap_data["cell_x_edges"], dtype=np.float64),
                cell_y_edges=np.asarray(heatmap_data["cell_y_edges"], dtype=np.float64),
                peak_pixel_xy=np.asarray(
                    heatmap_data["peak_pixel_xy"], dtype=np.float64
                ),
                exact_xy=np.asarray(heatmap_data["exact_xy"], dtype=np.float64),
                approx_xy=np.asarray(heatmap_data["approx_xy"], dtype=np.float64),
            )


def estimate_lut_accuracy(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolations: str | Sequence[str] = "bilinear",
) -> UnprojectLUTAccuracyReport:
    """Estimate maximum angular interpolation error for one or more modes.

    For each LUT cell, runs a gradient-ascent maximisation in normalised
    camera-frame coordinates of ``sin²(angle between approx and exact rays)``
    with a ReLU penalty enforcing that ``project(n)`` stays inside the pixel
    cell. The reported value per mode is the worst per-cell maximum.

    Args:
        lut: Runtime LUT to analyse.
        model: The exact camera model the LUT was built from. Must be a
            :class:`PinholeSplined` or :class:`OpenCV` instance.
        interpolations: Interpolation modes to include in the report. Each
            entry must be one of ``"nearest"``, ``"bilinear"``, ``"bicubic"``.
            Pass a single string or a sequence of strings.

    Returns:
        Accuracy report for the requested interpolation modes.
    """
    normalized_interpolations = _normalize_interpolations(interpolations)
    max_errors_mdeg = _max_cell_angular_errors_mdeg(
        lut,
        model,
        interpolations=normalized_interpolations,
    )
    return UnprojectLUTAccuracyReport(
        interpolations=normalized_interpolations,
        max_angular_error_mdeg=max_errors_mdeg,
    )


def compute_lut_error_heatmap(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: str = "bilinear",
) -> UnprojectLUTErrorHeatmap:
    """Compute a per-cell error heatmap for one interpolation mode.

    Each LUT cell is maximised independently in normalised camera-frame
    coordinates with a gradient-ascent loop run in C++. The heatmap stores
    the per-cell peak angular error along with the residual vector and the
    pixel where the peak sits.

    Args:
        lut: Runtime LUT to analyse.
        model: The exact camera model the LUT was built from. Must be a
            :class:`PinholeSplined` or :class:`OpenCV` instance.
        interpolation: Interpolation mode to evaluate. One of ``"nearest"``,
            ``"bilinear"``, ``"bicubic"``.

    Returns:
        In-memory heatmap for the requested interpolation mode.
    """
    interpolation = _validate_interpolation_mode(interpolation)

    cells_result = _max_cell_errors_call(
        lut,
        model,
        interpolation=interpolation,
    )

    heatmap_width = lut.grid_width - 1
    heatmap_height = lut.grid_height - 1
    cells_grid = cells_result.reshape(heatmap_height, heatmap_width, 6)

    peak_pixel_xy = cells_grid[:, :, 0:2]
    exact_xy = cells_grid[:, :, 2:4]
    approx_xy = cells_grid[:, :, 4:6]

    x_edges = np.linspace(lut.grid_x_min, lut.grid_x_max, lut.grid_width)
    y_edges = np.linspace(lut.grid_y_min, lut.grid_y_max, lut.grid_height)

    return UnprojectLUTErrorHeatmap(
        interpolation=interpolation,
        cell_x_edges=x_edges,
        cell_y_edges=y_edges,
        peak_pixel_xy=peak_pixel_xy,
        exact_xy=exact_xy,
        approx_xy=approx_xy,
    )


def _max_cell_errors_call(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: str,
) -> np.ndarray:
    """Run the C++ per-cell maximiser for a single interpolation mode.

    Dispatches by camera-model type to the matching C++ entry point and
    returns the raw per-cell result array.

    Args:
        lut: Runtime LUT to analyse.
        model: Camera model the LUT was built from. Must be a PinholeSplined
            or OpenCV instance.
        interpolation: Interpolation mode. One of ``"nearest"``, ``"bilinear"``,
            ``"bicubic"``.

    Returns:
        Per-cell results with shape ``(num_cells, 5)``. Columns are
        ``[max_angular_error_deg, peak_pixel_x, peak_pixel_y,
        error_delta_x, error_delta_y]``. Cells are row-major over
        ``(cell_y, cell_x)``.
    """
    from lensboy import lensboy_bindings as lbb
    from lensboy.camera_models.opencv import OpenCV
    from lensboy.camera_models.pinhole_splined import PinholeSplined

    if lut.grid_width < 2 or lut.grid_height < 2:
        raise ValueError(
            "Per-cell error analysis requires a LUT with at least 2 grid "
            "samples in each dimension."
        )

    mode_int = _INTERP_TO_CPP_MODE[interpolation]
    common_kwargs = {
        "lut_xy_grid": lut.xy_grid,
        "grid_x_min": lut.grid_x_min,
        "grid_x_max": lut.grid_x_max,
        "grid_y_min": lut.grid_y_min,
        "grid_y_max": lut.grid_y_max,
        "interpolation_mode": mode_int,
        "max_iters": _OPTIMISER_MAX_ITERS,
        "grad_tol": _OPTIMISER_GRAD_TOL,
    }
    if isinstance(model, PinholeSplined):
        return lbb.max_cell_errors_pinhole_splined(
            config=model._cpp_config(),
            intrinsics=model._cpp_params(),
            **common_kwargs,
        )
    if isinstance(model, OpenCV):
        intrinsics_18 = np.concatenate(
            [
                np.array([model.fx, model.fy, model.cx, model.cy], dtype=np.float64),
                np.asarray(model.distortion_coeffs, dtype=np.float64),
            ]
        )
        return lbb.max_cell_errors_opencv(
            intrinsics=intrinsics_18,
            **common_kwargs,
        )
    raise TypeError(
        "Per-cell error analysis is only supported for PinholeSplined and "
        "OpenCV camera models."
    )


def _max_cell_angular_errors_mdeg(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolations: tuple[str, ...],
) -> dict[str, float]:
    """Maximum per-cell angular error for each interpolation mode, in mdeg.

    Args:
        lut: Runtime LUT to analyse.
        model: Camera model the LUT was built from.
        interpolations: Interpolation modes to evaluate.

    Returns:
        Mapping from interpolation mode to maximum angular error in millidegrees.
    """
    max_errors_mdeg: dict[str, float] = {}
    for mode in interpolations:
        cells_result = _max_cell_errors_call(
            lut,
            model,
            interpolation=mode,
        )
        ex = cells_result[:, 2]
        ey = cells_result[:, 3]
        ax = cells_result[:, 4]
        ay = cells_result[:, 5]
        cross_x = ey - ay
        cross_y = ax - ex
        cross_z = ex * ay - ey * ax
        cross_norm = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
        dot = ex * ax + ey * ay + 1.0
        per_cell_deg = np.rad2deg(np.arctan2(cross_norm, dot))
        max_err_deg = float(np.max(per_cell_deg))
        max_errors_mdeg[mode] = max_err_deg * _ANGULAR_ERROR_MDEG_SCALE
    return max_errors_mdeg
