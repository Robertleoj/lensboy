from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from lensboy.camera_models.unproject_lut import InterpolationMode, UnprojectLUT

if TYPE_CHECKING:
    from lensboy.camera_models.base_model import CameraModel

_NormalizePointsFn = Callable[[np.ndarray], np.ndarray]

_SUPPORTED_INTERPOLATIONS: tuple[InterpolationMode, ...] = (
    "nearest",
    "bilinear",
    "bicubic",
)
_DEFAULT_ERROR_MAX_DEPTH = 2
_DEFAULT_ERROR_MIN_CELL_SIZE = 0.5
_ANGULAR_ERROR_MDEG_SCALE = 1.0e3


def _validate_target_sample_count(target_sample_count: int) -> int:
    resolved = int(target_sample_count)
    if resolved <= 0:
        raise ValueError("target_sample_count must be positive.")
    return resolved


def _validate_interpolation_mode(interpolation: str) -> InterpolationMode:
    if interpolation not in _SUPPORTED_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported interpolation mode {interpolation!r}. "
            f"Expected one of {_SUPPORTED_INTERPOLATIONS}."
        )
    return interpolation  # type: ignore[return-value]


def _validate_error_mode(mode: str) -> str:
    if mode != "adaptive":
        raise ValueError(f"Unsupported error mode {mode!r}.")
    return mode


def _normalize_interpolations(
    interpolations: InterpolationMode
    | tuple[InterpolationMode, ...]
    | list[InterpolationMode],
) -> tuple[InterpolationMode, ...]:
    if isinstance(interpolations, str):
        raw_items = [interpolations]
    else:
        raw_items = list(interpolations)

    normalized: list[InterpolationMode] = []
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


def _angular_error_deg_from_xy(
    reference_xy: np.ndarray,
    approx_xy: np.ndarray,
) -> np.ndarray:
    ref_x = reference_xy[:, 0]
    ref_y = reference_xy[:, 1]
    approx_x = approx_xy[:, 0]
    approx_y = approx_xy[:, 1]
    dot = ref_x * approx_x + ref_y * approx_y + 1.0
    ref_norm = np.sqrt(ref_x * ref_x + ref_y * ref_y + 1.0)
    approx_norm = np.sqrt(approx_x * approx_x + approx_y * approx_y + 1.0)
    return np.rad2deg(np.arccos(np.clip(dot / (ref_norm * approx_norm), -1.0, 1.0)))


def _dense_sample_grid(
    *,
    image_width: int,
    image_height: int,
    target_sample_count: int,
) -> tuple[int, int, np.ndarray]:
    image_aspect = image_width / image_height
    sample_grid_width = max(
        2,
        int(round(np.sqrt(target_sample_count * image_aspect))),
    )
    sample_grid_height = max(
        2,
        int(round(target_sample_count / sample_grid_width)),
    )
    xs = np.linspace(0.0, image_width - 1, sample_grid_width)
    ys = np.linspace(0.0, image_height - 1, sample_grid_height)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    sample_pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return sample_grid_width, sample_grid_height, sample_pixels


def _sample_cell_points(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x0, y1],
            [x1, y1],
            [x_mid, y0],
            [x_mid, y1],
            [x0, y_mid],
            [x1, y_mid],
            [x_mid, y_mid],
        ],
        dtype=np.float64,
    )


def _sample_cell_points_batch(
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    points = np.empty((len(x0), 9, 2), dtype=np.float64)
    points[:, 0, 0] = x0
    points[:, 0, 1] = y0
    points[:, 1, 0] = x1
    points[:, 1, 1] = y0
    points[:, 2, 0] = x0
    points[:, 2, 1] = y1
    points[:, 3, 0] = x1
    points[:, 3, 1] = y1
    points[:, 4, 0] = x_mid
    points[:, 4, 1] = y0
    points[:, 5, 0] = x_mid
    points[:, 5, 1] = y1
    points[:, 6, 0] = x0
    points[:, 6, 1] = y_mid
    points[:, 7, 0] = x1
    points[:, 7, 1] = y_mid
    points[:, 8, 0] = x_mid
    points[:, 8, 1] = y_mid
    return points


def _subdivide_cell(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> list[tuple[float, float, float, float]]:
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    return [
        (x0, x_mid, y0, y_mid),
        (x_mid, x1, y0, y_mid),
        (x0, x_mid, y_mid, y1),
        (x_mid, x1, y_mid, y1),
    ]


def _estimate_cells_error_detail_batch(
    lut: UnprojectLUT,
    normalize_points_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mode: InterpolationMode,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_points = _sample_cell_points_batch(x0, x1, y0, y1)
    flat_points = sample_points.reshape(-1, 2)
    exact_rays = normalize_points_fn(flat_points)
    exact_xy = np.asarray(exact_rays[:, :2], dtype=np.float64)
    approx_xy = lut._interpolate_xy(
        flat_points[:, 0],
        flat_points[:, 1],
        mode,
        "strict",
    )
    errors = _angular_error_deg_from_xy(exact_xy, approx_xy).reshape(len(x0), 9)
    error_delta_xy = (approx_xy - exact_xy).reshape(len(x0), 9, 2)

    best_indices = np.argmax(errors, axis=1)
    row_indices = np.arange(len(x0))
    max_errors = errors[row_indices, best_indices]
    best_delta_xy = error_delta_xy[row_indices, best_indices]
    delta_norm = np.linalg.norm(best_delta_xy, axis=1, keepdims=True)
    best_direction_xy = np.divide(
        best_delta_xy,
        delta_norm,
        out=np.zeros_like(best_delta_xy),
        where=delta_norm > 0.0,
    )
    best_peak_pixel = sample_points[row_indices, best_indices]
    interior_peak = np.max(errors[:, 4:], axis=1) > (
        np.max(errors[:, :4], axis=1) + 1e-12
    )
    sampled_errors_deg = errors.reshape(-1).copy()
    return (
        max_errors,
        best_direction_xy,
        best_delta_xy,
        best_peak_pixel,
        interior_peak,
        sampled_errors_deg,
    )


def _estimate_cell_error_detail(
    lut: UnprojectLUT,
    normalize_points_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mode: InterpolationMode,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    depth: int,
    max_depth: int,
    min_cell_size: float,
    sampled_errors_deg: list[float],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    sample_points = _sample_cell_points(x0, x1, y0, y1)
    exact_rays = normalize_points_fn(sample_points)
    exact_xy = np.asarray(exact_rays[:, :2], dtype=np.float64)
    approx_xy = lut._interpolate_xy(
        sample_points[:, 0],
        sample_points[:, 1],
        mode,
        "strict",
    )
    errors = _angular_error_deg_from_xy(exact_xy, approx_xy)
    sampled_errors_deg.extend(errors.tolist())
    error_delta_xy = approx_xy - exact_xy
    best_index = int(np.argmax(errors))
    max_error = float(errors[best_index])
    best_delta_xy = np.asarray(error_delta_xy[best_index], dtype=np.float64)
    delta_norm = float(np.linalg.norm(best_delta_xy))
    if delta_norm > 0.0:
        best_direction_xy = best_delta_xy / delta_norm
    else:
        best_direction_xy = np.zeros(2, dtype=np.float64)
    best_peak_pixel = np.asarray(sample_points[best_index], dtype=np.float64)

    interior_peak = float(np.max(errors[4:])) > float(np.max(errors[:4])) + 1e-12
    cell_width = abs(x1 - x0)
    cell_height = abs(y1 - y0)
    can_subdivide = depth < max_depth and (
        cell_width > min_cell_size or cell_height > min_cell_size
    )
    if not (interior_peak and can_subdivide):
        return max_error, best_direction_xy, best_delta_xy, best_peak_pixel

    for subcell in _subdivide_cell(x0, x1, y0, y1):
        (
            subcell_max_error,
            subcell_direction_xy,
            subcell_delta_xy,
            subcell_peak_pixel,
        ) = _estimate_cell_error_detail(
            lut,
            normalize_points_fn,
            mode=mode,
            x0=subcell[0],
            x1=subcell[1],
            y0=subcell[2],
            y1=subcell[3],
            depth=depth + 1,
            max_depth=max_depth,
            min_cell_size=min_cell_size,
            sampled_errors_deg=sampled_errors_deg,
        )
        if subcell_max_error > max_error:
            max_error = subcell_max_error
            best_direction_xy = subcell_direction_xy
            best_delta_xy = subcell_delta_xy
            best_peak_pixel = subcell_peak_pixel
    return max_error, best_direction_xy, best_delta_xy, best_peak_pixel


def _estimate_adaptive_errors_for_cell_chunk(
    lut: UnprojectLUT,
    normalize_points_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mode: InterpolationMode,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    max_depth: int,
    min_cell_size: float,
) -> tuple[float, np.ndarray]:
    (
        top_level_max_errors,
        _top_level_direction_xy,
        _top_level_delta_xy,
        _top_level_peak_pixel,
        interior_peak,
        top_level_sampled_errors_deg,
    ) = _estimate_cells_error_detail_batch(
        lut,
        normalize_points_fn,
        mode=mode,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
    )

    max_error = float(np.max(top_level_max_errors))
    sampled_error_arrays: list[np.ndarray] = [top_level_sampled_errors_deg]
    cell_widths = np.abs(x1 - x0)
    cell_heights = np.abs(y1 - y0)
    can_subdivide = (cell_widths > min_cell_size) | (cell_heights > min_cell_size)
    recurse_indices = np.flatnonzero(interior_peak & can_subdivide)

    if len(recurse_indices) == 0:
        return max_error, top_level_sampled_errors_deg

    sampled_errors_deg: list[float] = []
    for cell_index in recurse_indices:
        max_error = max(
            max_error,
            _estimate_cell_error_detail(
                lut,
                normalize_points_fn,
                mode=mode,
                x0=float(x0[cell_index]),
                x1=float(x1[cell_index]),
                y0=float(y0[cell_index]),
                y1=float(y1[cell_index]),
                depth=0,
                max_depth=max_depth,
                min_cell_size=min_cell_size,
                sampled_errors_deg=sampled_errors_deg,
            )[0],
        )

    if len(sampled_errors_deg) > 0:
        sampled_error_arrays.append(np.asarray(sampled_errors_deg, dtype=np.float64))
    return max_error, np.concatenate(sampled_error_arrays)


@dataclass(frozen=True)
class UnprojectLUTAccuracyReport:
    """Accuracy summary for one or more LUT interpolation modes.

    Args:
        interpolations: Interpolation modes included in the report.
        max_angular_error_mdeg: Observed maximum angular error per interpolation mode.
        median_angular_error_mdeg: Observed median angular error per interpolation mode.
        mode: Error-estimation mode used to build the report.
        max_depth: Maximum adaptive subdivision depth.
        min_cell_size: Minimum subcell size in pixels.

    Returns:
        Immutable report describing the requested interpolation modes.
    """

    interpolations: tuple[InterpolationMode, ...]
    max_angular_error_mdeg: dict[str, float]
    median_angular_error_mdeg: dict[str, float]
    mode: str
    max_depth: int
    min_cell_size: float


@dataclass
class UnprojectLUTSampleAccuracy:
    """Dense sampled comparison between a LUT and its exact source model.

    Args:
        interpolation: Interpolation mode used to query the LUT.
        target_sample_count: Requested approximate number of sample pixels.
        sample_grid_width: Sample-grid width.
        sample_grid_height: Sample-grid height.
        sample_pixels: Evenly spaced sample pixels with shape ``(N, 2)``.
        exact_rays: Exact source-model rays with shape ``(N, 3)``.
        approx_rays: LUT-queried rays with shape ``(N, 3)``.
        angular_error_deg: Per-sample angular error in degrees, shape ``(N,)``.

    Returns:
        In-memory sampled comparison result for one interpolation mode.
    """

    interpolation: InterpolationMode
    target_sample_count: int
    sample_grid_width: int
    sample_grid_height: int
    sample_pixels: np.ndarray
    exact_rays: np.ndarray
    approx_rays: np.ndarray
    angular_error_deg: np.ndarray

    def __post_init__(self) -> None:
        self.interpolation = _validate_interpolation_mode(self.interpolation)
        self.target_sample_count = _validate_target_sample_count(self.target_sample_count)
        self.sample_grid_width = int(self.sample_grid_width)
        self.sample_grid_height = int(self.sample_grid_height)
        self.sample_pixels = np.asarray(self.sample_pixels, dtype=np.float64).copy()
        self.exact_rays = np.asarray(self.exact_rays, dtype=np.float64).copy()
        self.approx_rays = np.asarray(self.approx_rays, dtype=np.float64).copy()
        self.angular_error_deg = np.asarray(
            self.angular_error_deg, dtype=np.float64
        ).copy()

        expected_samples = self.sample_grid_width * self.sample_grid_height
        if self.sample_grid_width < 2 or self.sample_grid_height < 2:
            raise ValueError("sample grid dimensions must both be at least 2.")
        if self.sample_pixels.shape != (expected_samples, 2):
            raise ValueError(
                "sample_pixels must have shape "
                f"({expected_samples}, 2), got {self.sample_pixels.shape}."
            )
        if self.exact_rays.shape != (expected_samples, 3):
            raise ValueError(
                "exact_rays must have shape "
                f"({expected_samples}, 3), got {self.exact_rays.shape}."
            )
        if self.approx_rays.shape != (expected_samples, 3):
            raise ValueError(
                "approx_rays must have shape "
                f"({expected_samples}, 3), got {self.approx_rays.shape}."
            )
        if self.angular_error_deg.shape != (expected_samples,):
            raise ValueError(
                "angular_error_deg must have shape "
                f"({expected_samples},), got {self.angular_error_deg.shape}."
            )

    @property
    def sample_count(self) -> int:
        """Return the number of sampled pixels.

        Returns:
            Number of sample pixels in the dense comparison grid.
        """
        return int(len(self.angular_error_deg))

    @property
    def max_angular_error_mdeg(self) -> float:
        """Return the maximum sampled angular error in milli degrees.

        Returns:
            Maximum per-sample angular error.
        """
        return float(np.max(self.angular_error_deg) * _ANGULAR_ERROR_MDEG_SCALE)

    @property
    def mean_angular_error_mdeg(self) -> float:
        """Return the mean sampled angular error in milli degrees.

        Returns:
            Mean per-sample angular error.
        """
        return float(np.mean(self.angular_error_deg) * _ANGULAR_ERROR_MDEG_SCALE)

    @property
    def median_angular_error_mdeg(self) -> float:
        """Return the median sampled angular error in milli degrees.

        Returns:
            Median per-sample angular error.
        """
        return float(np.median(self.angular_error_deg) * _ANGULAR_ERROR_MDEG_SCALE)


@dataclass
class UnprojectLUTErrorHeatmap:
    """Per-cell angular-error heatmap for a LUT interpolation mode.

    Args:
        interpolation: Interpolation mode represented by the heatmap.
        max_depth: Maximum adaptive subdivision depth.
        min_cell_size: Minimum subcell size in pixels.
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

    interpolation: InterpolationMode
    max_depth: int
    min_cell_size: float
    cell_x_edges: np.ndarray
    cell_y_edges: np.ndarray
    max_angular_error_deg: np.ndarray
    error_direction_xy: np.ndarray
    error_delta_xy: np.ndarray
    peak_pixel_xy: np.ndarray

    def __post_init__(self) -> None:
        self.interpolation = _validate_interpolation_mode(self.interpolation)
        self.max_depth = int(self.max_depth)
        self.min_cell_size = float(self.min_cell_size)
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
            max_depth=np.array(self.max_depth, dtype=np.int64),
            min_cell_size=np.array(self.min_cell_size, dtype=np.float64),
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
                interpolation=cast(
                    InterpolationMode,
                    str(np.asarray(heatmap_data["interpolation"]).item()),
                ),
                max_depth=int(np.asarray(heatmap_data["max_depth"]).item()),
                min_cell_size=float(np.asarray(heatmap_data["min_cell_size"]).item()),
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
    interpolations: InterpolationMode
    | tuple[InterpolationMode, ...]
    | list[InterpolationMode] = "bilinear",
    mode: str = "adaptive",
    max_depth: int = _DEFAULT_ERROR_MAX_DEPTH,
    min_cell_size: float = _DEFAULT_ERROR_MIN_CELL_SIZE,
) -> UnprojectLUTAccuracyReport:
    """Estimate angular interpolation accuracy for one or more modes.

    Args:
        lut: Runtime LUT to analyze.
        model: The exact camera model the LUT was built from.
        interpolations: Interpolation modes to include in the report.
        mode: Error-estimation mode. Only ``"adaptive"`` is supported.
        max_depth: Maximum adaptive subdivision depth.
        min_cell_size: Minimum subcell size in pixels.

    Returns:
        Accuracy report for the requested interpolation modes.
    """
    normalized_interpolations = _normalize_interpolations(interpolations)
    mode = _validate_error_mode(mode)
    max_errors_mdeg, median_errors_mdeg = _estimate_adaptive_errors(
        lut,
        model.normalize_points,
        interpolations=normalized_interpolations,
        max_depth=max_depth,
        min_cell_size=min_cell_size,
    )
    return UnprojectLUTAccuracyReport(
        interpolations=normalized_interpolations,
        max_angular_error_mdeg=max_errors_mdeg,
        median_angular_error_mdeg=median_errors_mdeg,
        mode=mode,
        max_depth=max_depth,
        min_cell_size=min_cell_size,
    )


def compute_lut_error_heatmap(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: InterpolationMode = "bilinear",
    mode: str = "adaptive",
    max_depth: int = _DEFAULT_ERROR_MAX_DEPTH,
    min_cell_size: float = _DEFAULT_ERROR_MIN_CELL_SIZE,
) -> UnprojectLUTErrorHeatmap:
    """Compute a per-cell error heatmap for one interpolation mode.

    Args:
        lut: Runtime LUT to analyze.
        model: The exact camera model the LUT was built from.
        interpolation: Interpolation mode to evaluate.
        mode: Error-estimation mode. Only ``"adaptive"`` is supported.
        max_depth: Maximum adaptive subdivision depth.
        min_cell_size: Minimum subcell size in pixels.

    Returns:
        In-memory heatmap for the requested interpolation mode.
    """
    interpolation = _validate_interpolation_mode(interpolation)
    _validate_error_mode(mode)
    normalize_points_fn = model.normalize_points

    x_edges = np.linspace(lut.grid_x_min, lut.grid_x_max, lut.grid_width)
    y_edges = np.linspace(lut.grid_y_min, lut.grid_y_max, lut.grid_height)
    heatmap_width = max(lut.grid_width - 1, 1)
    heatmap_height = max(lut.grid_height - 1, 1)
    max_angular_error_deg = np.zeros((heatmap_height, heatmap_width), dtype=np.float64)
    error_direction_xy = np.zeros((heatmap_height, heatmap_width, 2), dtype=np.float64)
    error_delta_xy = np.zeros((heatmap_height, heatmap_width, 2), dtype=np.float64)
    peak_pixel_xy = np.zeros((heatmap_height, heatmap_width, 2), dtype=np.float64)

    for iy in range(heatmap_height):
        y0 = y_edges[min(iy, len(y_edges) - 1)]
        y1 = y_edges[min(iy + 1, len(y_edges) - 1)]
        for ix in range(heatmap_width):
            x0 = x_edges[min(ix, len(x_edges) - 1)]
            x1 = x_edges[min(ix + 1, len(x_edges) - 1)]
            (
                max_error,
                direction_xy,
                delta_xy,
                peak_pixel,
            ) = _estimate_cell_error_detail(
                lut,
                normalize_points_fn,
                mode=interpolation,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                depth=0,
                max_depth=max_depth,
                min_cell_size=min_cell_size,
                sampled_errors_deg=[],
            )
            max_angular_error_deg[iy, ix] = max_error
            error_direction_xy[iy, ix] = direction_xy
            error_delta_xy[iy, ix] = delta_xy
            peak_pixel_xy[iy, ix] = peak_pixel

    return UnprojectLUTErrorHeatmap(
        interpolation=interpolation,
        max_depth=max_depth,
        min_cell_size=min_cell_size,
        cell_x_edges=x_edges,
        cell_y_edges=y_edges,
        max_angular_error_deg=max_angular_error_deg,
        error_direction_xy=error_direction_xy,
        error_delta_xy=error_delta_xy,
        peak_pixel_xy=peak_pixel_xy,
    )


def sample_lut_accuracy(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: InterpolationMode = "bilinear",
    target_sample_count: int = 2500,
) -> UnprojectLUTSampleAccuracy:
    """Sample LUT accuracy on an evenly spaced image grid.

    Args:
        lut: Runtime LUT to analyze.
        model: The exact camera model the LUT was built from.
        interpolation: Interpolation mode to evaluate.
        target_sample_count: Approximate number of evenly spaced sample pixels.

    Returns:
        Dense sampled comparison between the LUT and the exact model.
    """
    interpolation = _validate_interpolation_mode(interpolation)
    target_sample_count = _validate_target_sample_count(target_sample_count)
    (
        sample_grid_width,
        sample_grid_height,
        sample_pixels,
    ) = _dense_sample_grid(
        image_width=lut.image_width,
        image_height=lut.image_height,
        target_sample_count=target_sample_count,
    )
    exact_rays = model.normalize_points(sample_pixels)
    approx_rays = cast(
        np.ndarray,
        lut.normalize_points(
            sample_pixels,
            interpolation=interpolation,
            bounds="strict",
        ),
    )
    angular_error_deg = _angular_error_deg_from_xy(
        np.asarray(exact_rays[:, :2], dtype=np.float64),
        np.asarray(approx_rays[:, :2], dtype=np.float64),
    )
    return UnprojectLUTSampleAccuracy(
        interpolation=interpolation,
        target_sample_count=target_sample_count,
        sample_grid_width=sample_grid_width,
        sample_grid_height=sample_grid_height,
        sample_pixels=sample_pixels,
        exact_rays=exact_rays,
        approx_rays=approx_rays,
        angular_error_deg=angular_error_deg,
    )


def _estimate_adaptive_errors(
    lut: UnprojectLUT,
    normalize_points_fn: _NormalizePointsFn,
    *,
    interpolations: tuple[InterpolationMode, ...],
    max_depth: int,
    min_cell_size: float,
) -> tuple[dict[str, float], dict[str, float]]:
    max_errors_mdeg: dict[str, float] = {}
    median_errors_mdeg: dict[str, float] = {}

    if lut.grid_width == 1 and lut.grid_height == 1:
        sample_points = np.array(
            [[lut.grid_x_min, lut.grid_y_min]],
            dtype=np.float64,
        )
        exact_rays = normalize_points_fn(sample_points)
        exact_xy = np.asarray(exact_rays[:, :2], dtype=np.float64)
        for mode in interpolations:
            approx_xy = lut._interpolate_xy(
                sample_points[:, 0],
                sample_points[:, 1],
                mode,
                "strict",
            )
            sample_errors_deg = _angular_error_deg_from_xy(exact_xy, approx_xy)
            max_errors_mdeg[mode] = (
                float(np.max(sample_errors_deg)) * _ANGULAR_ERROR_MDEG_SCALE
            )
            median_errors_mdeg[mode] = (
                float(np.median(sample_errors_deg)) * _ANGULAR_ERROR_MDEG_SCALE
            )
        return max_errors_mdeg, median_errors_mdeg

    x_edges = np.linspace(lut.grid_x_min, lut.grid_x_max, lut.grid_width)
    y_edges = np.linspace(lut.grid_y_min, lut.grid_y_max, lut.grid_height)
    x0_cells = x_edges[: max(lut.grid_width - 1, 1)]
    x1_cells = x_edges[1:] if lut.grid_width > 1 else x_edges[:1]
    y0_cells = y_edges[: max(lut.grid_height - 1, 1)]
    y1_cells = y_edges[1:] if lut.grid_height > 1 else y_edges[:1]
    cell_x0, cell_y0 = np.meshgrid(x0_cells, y0_cells, indexing="xy")
    cell_x1, cell_y1 = np.meshgrid(x1_cells, y1_cells, indexing="xy")
    flat_x0 = cell_x0.ravel()
    flat_x1 = cell_x1.ravel()
    flat_y0 = cell_y0.ravel()
    flat_y1 = cell_y1.ravel()

    for mode in interpolations:
        max_error, sampled_errors_deg = _estimate_adaptive_errors_for_cell_chunk(
            lut,
            normalize_points_fn,
            mode=mode,
            x0=flat_x0,
            x1=flat_x1,
            y0=flat_y0,
            y1=flat_y1,
            max_depth=max_depth,
            min_cell_size=min_cell_size,
        )
        median_error = (
            float(np.median(sampled_errors_deg))
            if len(sampled_errors_deg) > 0
            else float("nan")
        )
        max_errors_mdeg[mode] = max_error * _ANGULAR_ERROR_MDEG_SCALE
        median_errors_mdeg[mode] = median_error * _ANGULAR_ERROR_MDEG_SCALE

    return max_errors_mdeg, median_errors_mdeg
