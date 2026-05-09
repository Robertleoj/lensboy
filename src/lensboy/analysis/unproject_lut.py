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
_DEFAULT_MAX_ITERS = 50
_DEFAULT_GRAD_TOL = 1.0e-12
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
        max_iters: Per-cell gradient-ascent iteration cap used during the analysis.
        grad_tol: Per-cell gradient-norm convergence threshold used during the analysis.

    Returns:
        Immutable report describing the requested interpolation modes.
    """

    interpolations: tuple[str, ...]
    max_angular_error_mdeg: dict[str, float]
    max_iters: int
    grad_tol: float


@dataclass
class UnprojectLUTErrorHeatmap:
    """Per-cell angular-error heatmap for a LUT interpolation mode.

    Args:
        interpolation: Interpolation mode represented by the heatmap. One of
            ``"nearest"``, ``"bilinear"``, ``"bicubic"``.
        max_iters: Per-cell gradient-ascent iteration cap used during the analysis.
        grad_tol: Per-cell gradient-norm convergence threshold used during the analysis.
        cell_x_edges: Cell x edges, shape ``(grid_width,)``
            or ``(grid_width + 1,)``.
        cell_y_edges: Cell y edges, shape ``(grid_height,)``
            or ``(grid_height + 1,)``.
        max_angular_error_deg: Per-cell maximum angular error,
            shape ``(H, W)``.
        error_direction_xy: Unit x/y direction of the local peak
            error, shape ``(H, W, 2)``.
        error_delta_xy: Peak x/y interpolation error vector, shape ``(H, W, 2)``.
        peak_pixel_xy: Pixel location of the local peak error, shape ``(H, W, 2)``.

    Returns:
        In-memory representation of a saved or computed heatmap.
    """

    interpolation: str
    max_iters: int
    grad_tol: float
    cell_x_edges: np.ndarray
    cell_y_edges: np.ndarray
    max_angular_error_deg: np.ndarray
    error_direction_xy: np.ndarray
    error_delta_xy: np.ndarray
    peak_pixel_xy: np.ndarray

    def __post_init__(self) -> None:
        self.interpolation = _validate_interpolation_mode(self.interpolation)
        self.max_iters = int(self.max_iters)
        self.grad_tol = float(self.grad_tol)
        self.cell_x_edges = np.asarray(self.cell_x_edges, dtype=np.float64).copy()
        self.cell_y_edges = np.asarray(self.cell_y_edges, dtype=np.float64).copy()
        self.max_angular_error_deg = np.asarray(
            self.max_angular_error_deg, dtype=np.float64
        ).copy()
        self.error_direction_xy = np.asarray(
            self.error_direction_xy, dtype=np.float64
        ).copy()
        self.error_delta_xy = np.asarray(self.error_delta_xy, dtype=np.float64).copy()
        self.peak_pixel_xy = np.asarray(self.peak_pixel_xy, dtype=np.float64).copy()

        if self.max_angular_error_deg.ndim != 2:
            raise ValueError(
                "max_angular_error_deg must have shape (H, W), "
                f"got {self.max_angular_error_deg.shape}."
            )
        expected_vector_shape = (*self.max_angular_error_deg.shape, 2)
        if self.error_direction_xy.shape != expected_vector_shape:
            raise ValueError(
                "error_direction_xy must have shape (H, W, 2), "
                f"got {self.error_direction_xy.shape}."
            )
        if self.error_delta_xy.shape != expected_vector_shape:
            raise ValueError(
                "error_delta_xy must have shape (H, W, 2), "
                f"got {self.error_delta_xy.shape}."
            )
        if self.peak_pixel_xy.shape != expected_vector_shape:
            raise ValueError(
                "peak_pixel_xy must have shape (H, W, 2), "
                f"got {self.peak_pixel_xy.shape}."
            )

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
            max_iters=np.array(self.max_iters, dtype=np.int64),
            grad_tol=np.array(self.grad_tol, dtype=np.float64),
            cell_x_edges=self.cell_x_edges,
            cell_y_edges=self.cell_y_edges,
            max_angular_error_deg=self.max_angular_error_deg,
            error_direction_xy=self.error_direction_xy,
            error_delta_xy=self.error_delta_xy,
            peak_pixel_xy=self.peak_pixel_xy,
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
                max_iters=int(np.asarray(heatmap_data["max_iters"]).item()),
                grad_tol=float(np.asarray(heatmap_data["grad_tol"]).item()),
                cell_x_edges=np.asarray(heatmap_data["cell_x_edges"], dtype=np.float64),
                cell_y_edges=np.asarray(heatmap_data["cell_y_edges"], dtype=np.float64),
                max_angular_error_deg=np.asarray(
                    heatmap_data["max_angular_error_deg"], dtype=np.float64
                ),
                error_direction_xy=np.asarray(
                    heatmap_data["error_direction_xy"], dtype=np.float64
                ),
                error_delta_xy=np.asarray(
                    heatmap_data["error_delta_xy"], dtype=np.float64
                ),
                peak_pixel_xy=np.asarray(heatmap_data["peak_pixel_xy"], dtype=np.float64),
            )


def estimate_lut_accuracy(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolations: str | Sequence[str] = "bilinear",
    max_iters: int = _DEFAULT_MAX_ITERS,
    grad_tol: float = _DEFAULT_GRAD_TOL,
) -> UnprojectLUTAccuracyReport:
    """Estimate maximum angular interpolation error for one or more modes.

    For each LUT cell, runs a gradient-ascent maximisation in normalised
    camera-frame coordinates of ``‖approx_xy(project(n)) − n‖²`` with a
    ReLU penalty enforcing that ``project(n)`` stays inside the pixel cell.
    The reported value per mode is the worst per-cell maximum.

    Args:
        lut: Runtime LUT to analyse.
        model: The exact camera model the LUT was built from. Must be a
            :class:`PinholeSplined` or :class:`OpenCV` instance.
        interpolations: Interpolation modes to include in the report. Each
            entry must be one of ``"nearest"``, ``"bilinear"``, ``"bicubic"``.
            Pass a single string or a sequence of strings.
        max_iters: Per-cell gradient-ascent iteration cap.
        grad_tol: Per-cell gradient-norm convergence threshold.

    Returns:
        Accuracy report for the requested interpolation modes.
    """
    normalized_interpolations = _normalize_interpolations(interpolations)
    max_errors_mdeg = _max_cell_angular_errors_mdeg(
        lut,
        model,
        interpolations=normalized_interpolations,
        max_iters=max_iters,
        grad_tol=grad_tol,
    )
    return UnprojectLUTAccuracyReport(
        interpolations=normalized_interpolations,
        max_angular_error_mdeg=max_errors_mdeg,
        max_iters=max_iters,
        grad_tol=grad_tol,
    )


def compute_lut_error_heatmap(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: str = "bilinear",
    max_iters: int = _DEFAULT_MAX_ITERS,
    grad_tol: float = _DEFAULT_GRAD_TOL,
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
        max_iters: Per-cell gradient-ascent iteration cap.
        grad_tol: Per-cell gradient-norm convergence threshold.

    Returns:
        In-memory heatmap for the requested interpolation mode.
    """
    interpolation = _validate_interpolation_mode(interpolation)

    cells_result = _max_cell_errors_call(
        lut,
        model,
        interpolation=interpolation,
        max_iters=max_iters,
        grad_tol=grad_tol,
    )

    heatmap_width = lut.grid_width - 1
    heatmap_height = lut.grid_height - 1
    cells_grid = cells_result.reshape(heatmap_height, heatmap_width, 5)

    max_angular_error_deg = cells_grid[:, :, 0]
    peak_pixel_xy = cells_grid[:, :, 1:3]
    error_delta_xy = cells_grid[:, :, 3:5]
    delta_norm = np.linalg.norm(error_delta_xy, axis=2, keepdims=True)
    error_direction_xy = np.divide(
        error_delta_xy,
        delta_norm,
        out=np.zeros_like(error_delta_xy),
        where=delta_norm > 0.0,
    )

    x_edges = np.linspace(lut.grid_x_min, lut.grid_x_max, lut.grid_width)
    y_edges = np.linspace(lut.grid_y_min, lut.grid_y_max, lut.grid_height)

    return UnprojectLUTErrorHeatmap(
        interpolation=interpolation,
        max_iters=max_iters,
        grad_tol=grad_tol,
        cell_x_edges=x_edges,
        cell_y_edges=y_edges,
        max_angular_error_deg=max_angular_error_deg,
        error_direction_xy=error_direction_xy,
        error_delta_xy=error_delta_xy,
        peak_pixel_xy=peak_pixel_xy,
    )


def _max_cell_errors_call(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: str,
    max_iters: int,
    grad_tol: float,
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
        max_iters: Per-cell gradient-ascent iteration cap.
        grad_tol: Per-cell gradient-norm convergence threshold.

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
        "max_iters": max_iters,
        "grad_tol": grad_tol,
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
    max_iters: int,
    grad_tol: float,
) -> dict[str, float]:
    """Maximum per-cell angular error for each interpolation mode, in mdeg.

    Args:
        lut: Runtime LUT to analyse.
        model: Camera model the LUT was built from.
        interpolations: Interpolation modes to evaluate.
        max_iters: Per-cell gradient-ascent iteration cap.
        grad_tol: Per-cell gradient-norm convergence threshold.

    Returns:
        Mapping from interpolation mode to maximum angular error in millidegrees.
    """
    max_errors_mdeg: dict[str, float] = {}
    for mode in interpolations:
        cells_result = _max_cell_errors_call(
            lut,
            model,
            interpolation=mode,
            max_iters=max_iters,
            grad_tol=grad_tol,
        )
        max_err_deg = float(np.max(cells_result[:, 0]))
        max_errors_mdeg[mode] = max_err_deg * _ANGULAR_ERROR_MDEG_SCALE
    return max_errors_mdeg
