from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from lensboy.camera_models.unproject_lut import UnprojectLUT

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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
                f"approx_xy must have shape {expected_shape}, got {self.approx_xy.shape}."
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
                peak_pixel_xy=np.asarray(heatmap_data["peak_pixel_xy"], dtype=np.float64),
                exact_xy=np.asarray(heatmap_data["exact_xy"], dtype=np.float64),
                approx_xy=np.asarray(heatmap_data["approx_xy"], dtype=np.float64),
            )

    def plot(
        self,
        *,
        title: str | None = None,
        angular_unit: Literal["deg", "mdeg", "udeg", "rad", "mrad", "urad"] = "mdeg",
        vmax: float | None = None,
        show_directions: bool = True,
        arrow_grid: int = 28,
        arrow_scale: float = 0.5,
        constant_arrow_length: bool = False,
        cmap_name: str = "inferno",
        figsize: tuple[float, float] = (8.5, 6.0),
        return_figure: bool = False,
    ) -> Figure | None:
        """Plot the per-cell maximum angular error as a heatmap.

        Optionally overlays the local peak x/y interpolation error direction in
        each cell. Arrow length is proportional to per-cell error magnitude by
        default.

        Args:
            title: Plot title. Uses the heatmap's interpolation mode when omitted.
            angular_unit: Angular units for the heatmap color scale.
            vmax: Upper limit of the colorbar in the same units as ``angular_unit``.
                Cells above this clip to the top colour. ``None`` auto-fits to the
                data.
            show_directions: Whether to draw the error-direction arrows.
            arrow_grid: Approximate maximum number of arrows along the longer
                heatmap axis.
            arrow_scale: Arrow length, as a fraction of the spacing between drawn
                arrows, for the largest-error cell drawn. Other arrows scale down
                proportionally to their error magnitude (or are all this length
                when ``constant_arrow_length=True``).
            constant_arrow_length: If True, draw all arrows the same length
                regardless of per-cell error magnitude.
            cmap_name: Matplotlib colormap name for the heatmap.
            figsize: Figure size in inches as ``(width, height)``.
            return_figure: If True, return the figure instead of calling
                ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_unproject_lut_error_heatmap

        return _plot_unproject_lut_error_heatmap(
            self,
            title=title,
            angular_unit=angular_unit,
            vmax=vmax,
            show_directions=show_directions,
            arrow_grid=arrow_grid,
            arrow_scale=arrow_scale,
            constant_arrow_length=constant_arrow_length,
            cmap_name=cmap_name,
            figsize=figsize,
            return_figure=return_figure,
        )


def compute_lut_error_heatmap(
    lut: UnprojectLUT,
    model: CameraModel,
    *,
    interpolation: str = "bicubic",
) -> UnprojectLUTErrorHeatmap:
    """Compute a per-cell error heatmap for one interpolation mode.

    Each LUT cell is maximised independently in normalised camera-frame
    coordinates with a gradient-ascent loop run in C++. The heatmap stores
    the per-cell peak angular error along with the residual vector and the
    pixel where the peak sits.

    Args:
        lut: Runtime LUT to analyse.
        model: The exact camera model the LUT was built from.
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

    x_edges = np.linspace(0.0, lut.image_width - 1, lut.grid_width)
    y_edges = np.linspace(0.0, lut.image_height - 1, lut.grid_height)

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
        model: Camera model the LUT was built from.
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
        "image_width": lut.image_width,
        "image_height": lut.image_height,
        "interpolation_mode": mode_int,
        "max_iterations": _OPTIMISER_MAX_ITERS,
        "gradient_tolerance": _OPTIMISER_GRAD_TOL,
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
