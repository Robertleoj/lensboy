from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lensboy.camera_models.base_model import CameraModel
    from lensboy.camera_models.opencv import OpenCV
    from lensboy.camera_models.pinhole_splined import PinholeSplined

_METADATA_FILENAME = "metadata.json"
_XY_GRID_FILENAME = "xy_grid.npy"
_STORAGE_DTYPE = np.dtype("<f4")
SUPPORTED_INTERPOLATIONS: tuple[str, ...] = (
    "nearest",
    "bilinear",
    "bicubic",
)
SUPPORTED_BOUNDS: tuple[str, ...] = ("strict", "clamp", "extrapolate")


def _validate_interpolation_mode(interpolation: str) -> str:
    if interpolation not in SUPPORTED_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported interpolation mode {interpolation!r}. "
            f"Expected one of {SUPPORTED_INTERPOLATIONS}."
        )
    return interpolation


def _validate_bounds_mode(bounds: str) -> str:
    if bounds not in SUPPORTED_BOUNDS:
        raise ValueError(
            f"Unsupported bounds mode {bounds!r}. Expected one of {SUPPORTED_BOUNDS}."
        )
    return bounds


def _catmull_rom_weights(t: np.ndarray) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return np.stack(
        [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ],
        axis=1,
    )


def _stereographic_to_normalized(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Convert stereographic coordinates to normalized pinhole coordinates.

    Inverts the normalized_to_stereographic mapping.

    Args:
        sx: Stereographic x coordinates, shape ``(N,)``.
        sy: Stereographic y coordinates, shape ``(N,)``.

    Returns:
        Normalized coordinates, shape ``(N, 2)``.
    """
    r_s = np.sqrt(sx * sx + sy * sy + 1e-30)
    theta = 2.0 * np.arctan(r_s / 2.0)
    r_n = np.tan(theta)
    scale = r_n / r_s
    return np.column_stack([sx * scale, sy * scale])


def _compute_seed_grid(
    camera_model: OpenCV | PinholeSplined,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build a seed correspondence grid for seeded normalization.

    Creates a regular grid in stereographic space, converts to normalized
    coordinates, and forward-projects through the camera model to get pixel
    locations. Returns (pixel, normalized) correspondences suitable for
    building an initial-guess spatial index.

    Args:
        camera_model: Camera model with ``project_points`` and FOV properties.

    Returns:
        Tuple of (seed_pixels, seed_normals, seed_w, seed_h) where pixels and
        normals have shape ``(seed_w * seed_h, 2)``.
    """
    fov_x_rad = camera_model.fov_deg_x * math.pi / 180.0
    fov_y_rad = camera_model.fov_deg_y * math.pi / 180.0
    half_x = 2.0 * math.tan(fov_x_rad / 4.0)
    half_y = 2.0 * math.tan(fov_y_rad / 4.0)

    # No overscan -- the FOV-based range covers the image exactly.
    # Overscanning risks entering fold-over regions where the distortion
    # model maps multiple rays to the same pixel.

    # Seed grid: ~1 seed per 4 image pixels along the long axis
    w = camera_model.image_width
    h = camera_model.image_height
    aspect = half_x / half_y if half_y > 0 else 1.0
    seed_long = max(w, h) // 4
    if aspect >= 1.0:
        seed_w = seed_long
        seed_h = max(4, int(round(seed_long / aspect)))
    else:
        seed_h = seed_long
        seed_w = max(4, int(round(seed_long * aspect)))

    sx = np.linspace(-half_x, half_x, seed_w, dtype=np.float64)
    sy = np.linspace(-half_y, half_y, seed_h, dtype=np.float64)
    gsx, gsy = np.meshgrid(sx, sy, indexing="xy")
    flat_sx = gsx.ravel()
    flat_sy = gsy.ravel()

    normals_xy = _stereographic_to_normalized(flat_sx, flat_sy)
    rays_3d = np.column_stack([normals_xy, np.ones(len(normals_xy), dtype=np.float64)])
    seed_pixels = camera_model.project_points(rays_3d)

    # Discard seed points that project outside the image.  Points beyond
    # the valid FOV can fold back into the image under heavy distortion,
    # but the stereographic margin is tight enough that these are rare
    # and the Newton solver handles them via the pinhole fallback guess.
    w = camera_model.image_width
    h = camera_model.image_height
    outside = (
        (seed_pixels[:, 0] < -0.5)
        | (seed_pixels[:, 0] > w - 0.5)
        | (seed_pixels[:, 1] < -0.5)
        | (seed_pixels[:, 1] > h - 0.5)
        | ~np.isfinite(seed_pixels[:, 0])
    )
    seed_pixels[outside] = np.nan
    normals_xy[outside] = np.nan

    seed_normals = np.ascontiguousarray(normals_xy, dtype=np.float64)
    seed_pixels = np.ascontiguousarray(seed_pixels, dtype=np.float64)

    return seed_pixels, seed_normals, seed_w, seed_h


def _sample_xy_grid_seeded(
    camera_model: CameraModel,
    *,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Sample the unproject grid using seeded C++ normalization.

    Builds a seed correspondence grid from a stereographic-space sampling,
    then dispatches to C++ for spatial-index lookup, bilinear initial guess,
    and Newton refinement.

    Args:
        camera_model: Camera model to sample.
        x_coords: Grid x pixel coordinates, shape ``(grid_width,)``.
        y_coords: Grid y pixel coordinates, shape ``(grid_height,)``.

    Returns:
        Sampled xy grid, shape ``(grid_height, grid_width, 2)``.
    """
    from lensboy import lensboy_bindings as lbb

    grid_width = len(x_coords)
    grid_height = len(y_coords)

    # Build query pixel grid
    gx, gy = np.meshgrid(x_coords, y_coords, indexing="xy")
    query_pixels = np.ascontiguousarray(
        np.column_stack([gx.ravel(), gy.ravel()]), dtype=np.float64
    )

    seed_pixels, seed_normals, seed_w, seed_h = _compute_seed_grid(camera_model)  # type: ignore[arg-type]

    from lensboy.camera_models.opencv import OpenCV
    from lensboy.camera_models.pinhole_splined import PinholeSplined

    if isinstance(camera_model, OpenCV):
        dist = np.asarray(camera_model.distortion_coeffs, dtype=np.float64)
        if len(dist) < 14:
            dist = np.pad(dist, (0, 14 - len(dist)))
        intrinsics = np.concatenate(
            [
                np.array(
                    [camera_model.fx, camera_model.fy, camera_model.cx, camera_model.cy],
                    dtype=np.float64,
                ),
                dist[:14],
            ]
        )
        rays = lbb.seeded_normalize_opencv(
            seed_pixels, seed_normals, seed_w, seed_h, query_pixels, intrinsics
        )
    elif isinstance(camera_model, PinholeSplined):
        rays = lbb.seeded_normalize_splined(
            seed_pixels,
            seed_normals,
            seed_w,
            seed_h,
            query_pixels,
            camera_model._cpp_config(),
            camera_model._cpp_params(),
        )
    else:
        raise TypeError(
            "UnprojectLUT is only supported for OpenCV and PinholeSplined models."
        )

    xy = np.asarray(rays[:, :2], dtype=np.float64)
    return xy.reshape(grid_height, grid_width, 2)


def _normalize_pixel_stride(
    pixel_stride: float | tuple[float, float],
) -> tuple[float, float]:
    if isinstance(pixel_stride, tuple):
        stride_x = float(pixel_stride[0])
        stride_y = float(pixel_stride[1])
    else:
        stride_x = float(pixel_stride)
        stride_y = float(pixel_stride)
    if stride_x <= 0.0 or stride_y <= 0.0:
        raise ValueError("pixel_stride values must be positive.")
    return stride_x, stride_y


def _grid_size_from_stride(image_size: int, stride: float) -> int:
    if image_size <= 1:
        return 1
    return int(math.ceil((image_size - 1) / stride)) + 1


@dataclass
class UnprojectLUT:
    """Regular-grid cache of `normalize_points()` values.

    Stores the x/y components of camera-frame rays on a regular image-space grid.
    Queries interpolate those cached values and return rays of the form
    ``[x, y, 1]``.

    Args:
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        grid_width: Number of cached samples along x.
        grid_height: Number of cached samples along y.
        grid_x_min: Minimum pixel x covered by the LUT.
        grid_x_max: Maximum pixel x covered by the LUT.
        grid_y_min: Minimum pixel y covered by the LUT.
        grid_y_max: Maximum pixel y covered by the LUT.
        xy_grid: Cached x/y ray components with shape ``(grid_height, grid_width, 2)``.
        lensboy_version: Package version that produced the LUT.

    Returns:
        A runtime lookup table that can save, load, and query cached unprojection rays.
    """

    image_width: int
    image_height: int
    grid_width: int
    grid_height: int
    grid_x_min: float
    grid_x_max: float
    grid_y_min: float
    grid_y_max: float
    xy_grid: np.ndarray
    lensboy_version: str = field(default_factory=lambda: _package_version("lensboy"))
    _grid_scale_x: float = field(init=False, repr=False)
    _grid_scale_y: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive.")
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("grid dimensions must be positive.")
        if self.grid_x_max < self.grid_x_min or self.grid_y_max < self.grid_y_min:
            raise ValueError("grid extents must be ordered from min to max.")

        grid = np.asarray(self.xy_grid, dtype=np.float64)
        if grid.shape != (self.grid_height, self.grid_width, 2):
            raise ValueError(
                "xy_grid must have shape "
                f"({self.grid_height}, {self.grid_width}, 2), got {grid.shape}."
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError("xy_grid must contain only finite values.")
        self.xy_grid = np.ascontiguousarray(grid)

        self._grid_scale_x = self._compute_grid_scale(
            self.grid_width, self.grid_x_min, self.grid_x_max
        )
        self._grid_scale_y = self._compute_grid_scale(
            self.grid_height, self.grid_y_min, self.grid_y_max
        )

    def __repr__(self) -> str:
        return (
            f"UnprojectLUT(image={self.image_width}x{self.image_height}, "
            f"grid={self.grid_width}x{self.grid_height})"
        )

    @property
    def grid_size_wh(self) -> tuple[int, int]:
        """Return the cached grid size as ``(width, height)``.

        Returns:
            Grid size in samples.
        """
        return self.grid_width, self.grid_height

    @property
    def grid_extents_xy(self) -> tuple[float, float, float, float]:
        """Return the covered pixel domain.

        Returns:
            ``(x_min, x_max, y_min, y_max)`` in pixel coordinates.
        """
        return self.grid_x_min, self.grid_x_max, self.grid_y_min, self.grid_y_max

    @property
    def grid_stride_xy(self) -> tuple[float, float]:
        """Return the actual spacing between neighboring cached samples.

        Returns:
            ``(stride_x, stride_y)`` in pixel coordinates. These values are derived
            from the grid extents and sample counts, so they may be fractional.
        """
        stride_x = 0.0
        if self.grid_width > 1:
            stride_x = (self.grid_x_max - self.grid_x_min) / (self.grid_width - 1)
        stride_y = 0.0
        if self.grid_height > 1:
            stride_y = (self.grid_y_max - self.grid_y_min) / (self.grid_height - 1)
        return float(stride_x), float(stride_y)

    @property
    def supported_interpolations(self) -> tuple[str, ...]:
        """Return the interpolation modes supported by the LUT.

        Returns:
            Tuple of interpolation mode names: ``"nearest"``, ``"bilinear"``,
            ``"bicubic"``.
        """
        return SUPPORTED_INTERPOLATIONS

    @property
    def supported_bounds(self) -> tuple[str, ...]:
        """Return the bounds behaviors supported by the LUT.

        Returns:
            Tuple of bounds mode names: ``"strict"``, ``"clamp"``,
            ``"extrapolate"``.
        """
        return SUPPORTED_BOUNDS

    @staticmethod
    def _compute_grid_scale(size: int, minimum: float, maximum: float) -> float:
        if size == 1 or maximum == minimum:
            return 0.0
        return (size - 1) / (maximum - minimum)

    @staticmethod
    def from_camera_model(
        camera_model: CameraModel,
        *,
        grid_size_wh: tuple[int, int] | None = None,
        pixel_stride: float | tuple[float, float] | None = None,
    ) -> UnprojectLUT:
        """Build a LUT from a camera model.

        Args:
            camera_model: Camera model to sample.
            grid_size_wh: Number of samples as ``(width, height)``. If omitted,
                the LUT uses a per-pixel grid.
            pixel_stride: Approximate pixel spacing between cached samples. Mutually
                exclusive with ``grid_size_wh``.

        Returns:
            A populated unprojection LUT.
        """
        if grid_size_wh is not None and pixel_stride is not None:
            raise ValueError("grid_size_wh and pixel_stride are mutually exclusive.")

        if grid_size_wh is None:
            if pixel_stride is None:
                grid_width = camera_model.image_width
                grid_height = camera_model.image_height
            else:
                stride_x, stride_y = _normalize_pixel_stride(pixel_stride)
                grid_width = _grid_size_from_stride(camera_model.image_width, stride_x)
                grid_height = _grid_size_from_stride(camera_model.image_height, stride_y)
        else:
            grid_width, grid_height = int(grid_size_wh[0]), int(grid_size_wh[1])

        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("grid_size_wh must contain positive integers.")

        x_coords = np.linspace(
            0.0, float(camera_model.image_width - 1), grid_width, dtype=np.float64
        )
        y_coords = np.linspace(
            0.0, float(camera_model.image_height - 1), grid_height, dtype=np.float64
        )

        xy_grid = _sample_xy_grid_seeded(
            camera_model,
            x_coords=x_coords,
            y_coords=y_coords,
        )

        lut = UnprojectLUT(
            image_width=camera_model.image_width,
            image_height=camera_model.image_height,
            grid_width=grid_width,
            grid_height=grid_height,
            grid_x_min=0.0,
            grid_x_max=float(camera_model.image_width - 1),
            grid_y_min=0.0,
            grid_y_max=float(camera_model.image_height - 1),
            xy_grid=xy_grid,
        )
        return lut

    def save(self, dir_path: Path | str) -> None:
        """Serialize the LUT to a directory.

        Writes ``metadata.json`` with scalar parameters and ``xy_grid.npy``
        with the raw float32 ray grid.

        Args:
            dir_path: Destination directory (created if it doesn't exist).
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        metadata = {
            "lensboy-version": self.lensboy_version,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "grid_x_min": self.grid_x_min,
            "grid_x_max": self.grid_x_max,
            "grid_y_min": self.grid_y_min,
            "grid_y_max": self.grid_y_max,
        }
        (p / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=4))

        xy_grid = np.ascontiguousarray(self.xy_grid, dtype=_STORAGE_DTYPE)
        np.save(p / _XY_GRID_FILENAME, xy_grid, allow_pickle=False)

    @staticmethod
    def load(dir_path: Path | str) -> UnprojectLUT:
        """Load a LUT from a directory written by :meth:`save`.

        Args:
            dir_path: Directory containing ``metadata.json`` and ``xy_grid.npy``.

        Returns:
            Reconstructed LUT.
        """
        p = Path(dir_path)
        metadata = json.loads((p / _METADATA_FILENAME).read_text())

        version = metadata.get("lensboy-version")
        if version is None or int(version.split(".")[0]) < 3:
            raise ValueError(
                "This unproject LUT was created with an incompatible version of "
                "lensboy (< 3.0.0). Please regenerate it with the current version."
            )

        xy_grid = np.load(p / _XY_GRID_FILENAME, allow_pickle=False).astype(np.float64)
        grid_height, grid_width = xy_grid.shape[:2]

        return UnprojectLUT(
            image_width=int(metadata["image_width"]),
            image_height=int(metadata["image_height"]),
            grid_width=int(grid_width),
            grid_height=int(grid_height),
            grid_x_min=float(metadata["grid_x_min"]),
            grid_x_max=float(metadata["grid_x_max"]),
            grid_y_min=float(metadata["grid_y_min"]),
            grid_y_max=float(metadata["grid_y_max"]),
            xy_grid=xy_grid,
            lensboy_version=version,
        )

    def normalize_points(
        self,
        pixel_coords: np.ndarray,
        *,
        interpolation: str = "bilinear",
        bounds: str = "strict",
        return_valid_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Query the LUT for camera-frame rays.

        Args:
            pixel_coords: Pixel coordinates with shape ``(N, 2)``.
            interpolation: Interpolation mode to use. One of ``"nearest"``,
                ``"bilinear"``, ``"bicubic"``.
            bounds: Bounds behavior for out-of-domain pixels. One of
                ``"strict"`` (raise unless ``return_valid_mask`` is set),
                ``"clamp"`` (snap queries onto the LUT domain), or
                ``"extrapolate"`` (extend the interpolant past the domain).
            return_valid_mask: Whether to also return a boolean validity mask.

        Returns:
            Rays with shape ``(N, 3)``.
            If ``return_valid_mask`` is True, also returns a boolean mask with
            shape ``(N,)`` indicating which rows were valid in strict mode.
        """
        interpolation = _validate_interpolation_mode(interpolation)
        bounds = _validate_bounds_mode(bounds)

        pts = np.asarray(pixel_coords, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"pixel_coords must have shape (N, 2), got {pts.shape}.")

        query_x = pts[:, 0]
        query_y = pts[:, 1]
        valid_mask = self._valid_mask(query_x, query_y)

        if bounds == "strict" and not return_valid_mask and not np.all(valid_mask):
            raise ValueError("Some pixel coordinates lie outside the LUT domain.")

        rays = np.full((len(pts), 3), np.nan, dtype=np.float64)
        rays[:, 2] = 1.0

        if bounds == "strict":
            active_mask = valid_mask
            sample_x = query_x[active_mask]
            sample_y = query_y[active_mask]
        elif bounds == "clamp":
            active_mask = np.ones(len(pts), dtype=bool)
            sample_x = np.clip(query_x, self.grid_x_min, self.grid_x_max)
            sample_y = np.clip(query_y, self.grid_y_min, self.grid_y_max)
            valid_mask = np.ones(len(pts), dtype=bool)
        else:
            active_mask = np.ones(len(pts), dtype=bool)
            sample_x = query_x
            sample_y = query_y
            valid_mask = np.ones(len(pts), dtype=bool)

        if np.any(active_mask):
            approx_xy = self._interpolate_xy(sample_x, sample_y, interpolation, bounds)
            rays[active_mask, :2] = approx_xy

        if return_valid_mask:
            return rays, valid_mask
        return rays

    def _valid_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (x >= self.grid_x_min)
            & (x <= self.grid_x_max)
            & (y >= self.grid_y_min)
            & (y <= self.grid_y_max)
        )

    def _interpolate_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        interpolation: str,
        bounds: str,
    ) -> np.ndarray:
        if len(x) == 0:
            return np.empty((0, 2), dtype=np.float64)

        gx = (x - self.grid_x_min) * self._grid_scale_x
        gy = (y - self.grid_y_min) * self._grid_scale_y

        if interpolation == "nearest":
            return self._query_nearest(gx, gy)
        if interpolation == "bilinear":
            return self._query_bilinear(gx, gy, bounds)
        return self._query_bicubic(gx, gy, bounds)

    def _query_nearest(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        ix = np.clip(np.rint(gx).astype(np.int64), 0, self.grid_width - 1)
        iy = np.clip(np.rint(gy).astype(np.int64), 0, self.grid_height - 1)
        return self.xy_grid[iy, ix]

    def _query_bilinear(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        bounds: str,
    ) -> np.ndarray:
        ix0, tx = self._linear_indices_and_weights(gx, self.grid_width, bounds)
        iy0, ty = self._linear_indices_and_weights(gy, self.grid_height, bounds)

        ix1 = np.clip(ix0 + 1, 0, self.grid_width - 1)
        iy1 = np.clip(iy0 + 1, 0, self.grid_height - 1)

        v00 = self.xy_grid[iy0, ix0]
        v10 = self.xy_grid[iy0, ix1]
        v01 = self.xy_grid[iy1, ix0]
        v11 = self.xy_grid[iy1, ix1]

        tx_col = tx[:, None]
        ty_col = ty[:, None]
        top = v00 * (1.0 - tx_col) + v10 * tx_col
        bottom = v01 * (1.0 - tx_col) + v11 * tx_col
        return top * (1.0 - ty_col) + bottom * ty_col

    def _query_bicubic(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        bounds: str,
    ) -> np.ndarray:
        bilinear_xy = self._query_bilinear(gx, gy, bounds)
        if self.grid_width < 4 or self.grid_height < 4:
            return bilinear_xy

        if bounds == "extrapolate":
            gx_work = gx
            gy_work = gy
        else:
            gx_work = np.clip(gx, 0.0, self.grid_width - 1.0)
            gy_work = np.clip(gy, 0.0, self.grid_height - 1.0)

        ix1 = np.floor(gx_work).astype(np.int64)
        iy1 = np.floor(gy_work).astype(np.int64)
        cubic_mask = (
            (ix1 >= 1)
            & (ix1 <= self.grid_width - 3)
            & (iy1 >= 1)
            & (iy1 <= self.grid_height - 3)
        )
        if not np.any(cubic_mask):
            return bilinear_xy

        ix1 = ix1[cubic_mask]
        iy1 = iy1[cubic_mask]
        tx = gx_work[cubic_mask] - ix1
        ty = gy_work[cubic_mask] - iy1

        ix = np.stack(
            [
                ix1 - 1,
                ix1,
                ix1 + 1,
                ix1 + 2,
            ],
            axis=1,
        )
        iy = np.stack(
            [
                iy1 - 1,
                iy1,
                iy1 + 1,
                iy1 + 2,
            ],
            axis=1,
        )

        wx = _catmull_rom_weights(tx)
        wy = _catmull_rom_weights(ty)

        neighborhood = self.xy_grid[iy[:, :, None], ix[:, None, :], :]
        weighted_x = np.sum(neighborhood * wx[:, None, :, None], axis=2)
        bilinear_xy[cubic_mask] = np.sum(weighted_x * wy[:, :, None], axis=1)
        return bilinear_xy

    @staticmethod
    def _linear_indices_and_weights(
        g: np.ndarray,
        size: int,
        bounds: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if size == 1:
            return np.zeros(len(g), dtype=np.int64), np.zeros(len(g), dtype=np.float64)

        if bounds == "extrapolate":
            base = np.floor(g).astype(np.int64)
            base = np.clip(base, 0, size - 2)
            t = g - base
        else:
            g_clipped = np.clip(g, 0.0, size - 1.0)
            base = np.floor(g_clipped).astype(np.int64)
            base = np.minimum(base, size - 2)
            t = g_clipped - base

        return base, t
