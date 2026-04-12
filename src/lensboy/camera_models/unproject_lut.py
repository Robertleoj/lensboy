from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from lensboy.camera_models.base_model import CameraModel
    from lensboy.camera_models.opencv import OpenCV
    from lensboy.camera_models.pinhole_splined import PinholeSplined

InterpolationMode = Literal["nearest", "bilinear", "bicubic"]
BoundsMode = Literal["strict", "clamp", "extrapolate"]
StorageEncoding = Literal["float64_xy", "float32_xy", "float16_xy"]

_FORMAT_NAME = "lensboy_unproject_LUT"
_FORMAT_VERSION = 1
_HEADER_END_MARKER = "END_HEADER"
_PAYLOAD_LAYOUT = "row_major_interleaved_xy"
_PAYLOAD_ENDIANNESS = "little"
_SUPPORTED_INTERPOLATIONS: tuple[InterpolationMode, ...] = (
    "nearest",
    "bilinear",
    "bicubic",
)
_SUPPORTED_BOUNDS: tuple[BoundsMode, ...] = ("strict", "clamp", "extrapolate")
_SUPPORTED_ENCODINGS: dict[StorageEncoding, np.dtype] = {
    "float64_xy": np.dtype("<f8"),
    "float32_xy": np.dtype("<f4"),
    "float16_xy": np.dtype("<f2"),
}
_MAX_HEADER_BYTES = 512 * 1024 * 1024


def _validate_interpolation_mode(interpolation: str) -> InterpolationMode:
    if interpolation not in _SUPPORTED_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported interpolation mode {interpolation!r}. "
            f"Expected one of {_SUPPORTED_INTERPOLATIONS}."
        )
    return interpolation  # type: ignore[return-value]


def _validate_bounds_mode(bounds: str) -> BoundsMode:
    if bounds not in _SUPPORTED_BOUNDS:
        raise ValueError(
            f"Unsupported bounds mode {bounds!r}. Expected one of {_SUPPORTED_BOUNDS}."
        )
    return bounds  # type: ignore[return-value]


def _validate_storage_encoding(storage_encoding: str) -> StorageEncoding:
    if storage_encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(
            f"Unsupported storage encoding {storage_encoding!r}. "
            f"Expected one of {tuple(_SUPPORTED_ENCODINGS)}."
        )
    return storage_encoding  # type: ignore[return-value]


def _parse_pair_of_ints(text: str, field_name: str) -> tuple[int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{field_name} must contain exactly 2 comma-separated values.")
    return int(parts[0]), int(parts[1])


def _parse_quad_of_floats(
    text: str, field_name: str
) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{field_name} must contain exactly 4 comma-separated values.")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def _parse_pair_of_floats(text: str, field_name: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{field_name} must contain exactly 2 comma-separated values.")
    return float(parts[0]), float(parts[1])


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "not_computed"
    return f"{value:.17g}"


def _format_csv(values: list[str]) -> str:
    return ", ".join(values)


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
    payload_dtype: np.dtype,
) -> np.ndarray:
    """Sample the unproject grid using seeded C++ normalization.

    Builds a seed correspondence grid from a stereographic-space sampling,
    then dispatches to C++ for spatial-index lookup, bilinear initial guess,
    and Newton refinement.

    Args:
        camera_model: Camera model to sample.
        x_coords: Grid x pixel coordinates, shape ``(grid_width,)``.
        y_coords: Grid y pixel coordinates, shape ``(grid_height,)``.
        payload_dtype: Storage dtype for quantization simulation.

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
    if payload_dtype != np.dtype(np.float64):
        xy = np.asarray(xy, dtype=payload_dtype).astype(np.float64)

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
        storage_encoding: On-disk payload encoding.
        xy_grid: Cached x/y ray components with shape ``(grid_height, grid_width, 2)``.
        default_interpolation: Default interpolation mode for runtime queries.
        default_bounds: Default bounds behavior for runtime queries.
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
    storage_encoding: StorageEncoding
    xy_grid: np.ndarray
    default_interpolation: InterpolationMode = "bilinear"
    default_bounds: BoundsMode = "strict"
    lensboy_version: str = field(default_factory=lambda: _package_version("lensboy"))
    _grid_scale_x: float = field(init=False, repr=False)
    _grid_scale_y: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.storage_encoding = _validate_storage_encoding(self.storage_encoding)
        self.default_interpolation = _validate_interpolation_mode(
            self.default_interpolation
        )
        self.default_bounds = _validate_bounds_mode(self.default_bounds)
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
            f"grid={self.grid_width}x{self.grid_height}, "
            f"encoding={self.storage_encoding})"
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
    def header_text(self) -> str:
        """Return the ASCII file header for this LUT.

        Returns:
            Header text exactly as it would be written before the binary payload.
        """
        return self._encode_header()

    @property
    def payload_offset_bytes(self) -> int:
        """Return the byte offset where the binary payload begins.

        Returns:
            Number of bytes occupied by the serialized ASCII header.
        """
        return len(self.header_text.encode("ascii"))

    @property
    def payload_bytes(self) -> int:
        """Return the size of the serialized binary payload.

        Returns:
            Number of bytes in the row-major interleaved x/y payload.
        """
        dtype = _SUPPORTED_ENCODINGS[self.storage_encoding]
        return self.grid_width * self.grid_height * 2 * dtype.itemsize

    @property
    def total_bytes(self) -> int:
        """Return the total serialized file size for this LUT.

        Returns:
            Number of bytes in the full `.unproject_LUT` file.
        """
        return self.payload_offset_bytes + self.payload_bytes

    def header_preview(self, max_lines: int = 0) -> str:
        """Return a short human-readable preview of the file header.

        Args:
            max_lines: Maximum number of header lines to include. Use ``0`` to
                include the full header.

        Returns:
            Preview text containing the first header lines.
        """
        if max_lines < 0:
            raise ValueError("max_lines must be non-negative.")
        if max_lines == 0:
            return self.header_text.rstrip("\n")
        return "\n".join(self.header_text.splitlines()[:max_lines])

    @property
    def supported_interpolations(self) -> tuple[InterpolationMode, ...]:
        """Return the interpolation modes supported by the LUT.

        Returns:
            Tuple of interpolation mode names.
        """
        return _SUPPORTED_INTERPOLATIONS

    @property
    def supported_bounds(self) -> tuple[BoundsMode, ...]:
        """Return the bounds behaviors supported by the LUT.

        Returns:
            Tuple of bounds mode names.
        """
        return _SUPPORTED_BOUNDS

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
        storage_encoding: StorageEncoding = "float64_xy",
    ) -> UnprojectLUT:
        """Build a LUT from a camera model.

        Args:
            camera_model: Camera model to sample.
            grid_size_wh: Number of samples as ``(width, height)``. If omitted,
                the LUT uses a per-pixel grid.
            pixel_stride: Approximate pixel spacing between cached samples. Mutually
                exclusive with ``grid_size_wh``.
            storage_encoding: On-disk payload encoding to use when saving.

        Returns:
            A populated unprojection LUT.
        """
        storage_encoding = _validate_storage_encoding(storage_encoding)
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
        payload_dtype = _SUPPORTED_ENCODINGS[storage_encoding]

        xy_grid = _sample_xy_grid_seeded(
            camera_model,
            x_coords=x_coords,
            y_coords=y_coords,
            payload_dtype=payload_dtype,
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
            storage_encoding=storage_encoding,
            xy_grid=xy_grid,
        )
        return lut

    def save(self, path: Path | str) -> None:
        """Write the LUT to a `.unproject_LUT` file.

        Args:
            path: Destination path. Must end with ``.unproject_LUT``.

        Returns:
            None.
        """
        output_path = Path(path)
        if output_path.suffix != ".unproject_LUT":
            raise ValueError("UnprojectLUT files must use the .unproject_LUT suffix.")

        header = self._encode_header().encode("ascii")
        payload_dtype = _SUPPORTED_ENCODINGS[self.storage_encoding]
        payload = np.asarray(self.xy_grid, dtype=payload_dtype, order="C").tobytes()

        with output_path.open("wb") as f:
            f.write(header)
            f.write(payload)

    @staticmethod
    def load(path: Path | str) -> UnprojectLUT:
        """Load a LUT from disk.

        Args:
            path: Path to a `.unproject_LUT` file.

        Returns:
            The loaded LUT.
        """
        input_path = Path(path)
        with input_path.open("rb") as f:
            header_lines = UnprojectLUT._read_header_lines(f)
            header = UnprojectLUT._parse_header_lines(header_lines)
            payload_offset_bytes = f.tell()
            payload = f.read()

        format_name = header["format"]
        if format_name != _FORMAT_NAME:
            raise ValueError(f"Unsupported LUT format {format_name!r}.")

        format_version = int(header["format_version"])
        if format_version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported LUT format_version {format_version}. "
                f"Expected {_FORMAT_VERSION}."
            )
        UnprojectLUT._validate_header_fields(header)

        storage_encoding = _validate_storage_encoding(header["storage_encoding"])
        if header["payload_layout"] != _PAYLOAD_LAYOUT:
            raise ValueError(f"Unsupported payload_layout {header['payload_layout']!r}.")
        if header["payload_endianness"] != _PAYLOAD_ENDIANNESS:
            raise ValueError(
                f"Unsupported payload_endianness {header['payload_endianness']!r}."
            )
        declared_payload_offset_bytes = int(header["payload_offset_bytes"])
        if payload_offset_bytes != declared_payload_offset_bytes:
            raise ValueError(
                f"Header payload_offset_bytes={declared_payload_offset_bytes}, "
                f"but payload begins at byte offset {payload_offset_bytes}."
            )

        image_width, image_height = _parse_pair_of_ints(
            header["image_size_wh"], "image_size_wh"
        )
        grid_width, grid_height = _parse_pair_of_ints(
            header["grid_size_wh"], "grid_size_wh"
        )
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = _parse_quad_of_floats(
            header["grid_extents_xy"], "grid_extents_xy"
        )
        header_stride_x, header_stride_y = _parse_pair_of_floats(
            header["grid_stride_xy"], "grid_stride_xy"
        )
        dtype = _SUPPORTED_ENCODINGS[storage_encoding]
        expected_payload_bytes = grid_width * grid_height * 2 * dtype.itemsize
        if len(payload) != expected_payload_bytes:
            raise ValueError(
                f"Unexpected payload size {len(payload)} bytes; expected "
                f"{expected_payload_bytes}."
            )
        expected_stride_x = (
            0.0 if grid_width <= 1 else (grid_x_max - grid_x_min) / (grid_width - 1)
        )
        expected_stride_y = (
            0.0 if grid_height <= 1 else (grid_y_max - grid_y_min) / (grid_height - 1)
        )
        if not math.isclose(
            header_stride_x, expected_stride_x, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "grid_stride_xy does not match grid_extents_xy and grid_size_wh."
            )
        if not math.isclose(
            header_stride_y, expected_stride_y, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "grid_stride_xy does not match grid_extents_xy and grid_size_wh."
            )

        xy_grid = (
            np.frombuffer(payload, dtype=dtype)
            .astype(np.float64)
            .reshape(grid_height, grid_width, 2)
        )

        return UnprojectLUT(
            image_width=image_width,
            image_height=image_height,
            grid_width=grid_width,
            grid_height=grid_height,
            grid_x_min=grid_x_min,
            grid_x_max=grid_x_max,
            grid_y_min=grid_y_min,
            grid_y_max=grid_y_max,
            storage_encoding=storage_encoding,
            xy_grid=xy_grid,
            default_interpolation=_validate_interpolation_mode(
                header["default_interpolation"]
            ),
            default_bounds=_validate_bounds_mode(header["default_bounds"]),
            lensboy_version=header["lensboy_version"],
        )

    @staticmethod
    def _read_header_lines(file_obj) -> list[str]:
        header_lines: list[str] = []
        total_bytes = 0
        while True:
            raw_line = file_obj.readline()
            if raw_line == b"":
                raise ValueError("Reached end of file before END_HEADER.")
            total_bytes += len(raw_line)
            if total_bytes > _MAX_HEADER_BYTES:
                raise ValueError("Header exceeds the maximum supported size.")
            try:
                line = raw_line.decode("ascii").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise ValueError("Header must contain only ASCII text.") from exc
            if line == _HEADER_END_MARKER:
                return header_lines
            header_lines.append(line)

    @staticmethod
    def _parse_header_lines(header_lines: list[str]) -> dict[str, str]:
        header: dict[str, str] = {}
        for line in header_lines:
            if ":" not in line:
                raise ValueError(f"Invalid header line {line!r}. Expected 'key: value'.")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError("Header keys must be non-empty.")
            if key in header:
                raise ValueError(f"Duplicate header key {key!r}.")
            header[key] = value
        return header

    @staticmethod
    def _validate_header_fields(header: dict[str, str]) -> None:
        required_fields = {
            "format",
            "format_version",
            "lensboy_version",
            "image_size_wh",
            "grid_size_wh",
            "grid_extents_xy",
            "grid_stride_xy",
            "storage_encoding",
            "default_interpolation",
            "default_bounds",
            "payload_offset_bytes",
            "payload_layout",
            "payload_endianness",
        }

        missing = required_fields - set(header)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing required header fields: {missing_text}.")

        removed_fields = {
            "error_report_mode",
            "error_report_max_depth",
            "error_report_min_cell_size",
        }
        removed_prefixes = (
            "estimated_max_angular_error_",
            "estimated_median_angular_error_",
        )
        unexpected_removed_fields = sorted(
            key
            for key in header
            if key in removed_fields
            or any(key.startswith(prefix) for prefix in removed_prefixes)
        )
        if unexpected_removed_fields:
            removed_text = ", ".join(unexpected_removed_fields)
            raise ValueError(
                "This runtime-only .unproject_LUT format does not support "
                f"legacy error-report header fields: {removed_text}."
            )

    def _encode_header(self) -> str:
        stride_x, stride_y = self.grid_stride_xy

        def build_lines(payload_offset_bytes_text: str) -> list[str]:
            return [
                f"format: {_FORMAT_NAME}",
                f"payload_offset_bytes: {payload_offset_bytes_text}",
                f"format_version: {_FORMAT_VERSION}",
                f"lensboy_version: {self.lensboy_version}",
                "image_size_wh: "
                + _format_csv([str(self.image_width), str(self.image_height)]),
                "grid_size_wh: "
                + _format_csv([str(self.grid_width), str(self.grid_height)]),
                "grid_extents_xy: "
                + _format_csv(
                    [
                        _format_float(self.grid_x_min),
                        _format_float(self.grid_x_max),
                        _format_float(self.grid_y_min),
                        _format_float(self.grid_y_max),
                    ]
                ),
                "grid_stride_xy: "
                + _format_csv([_format_float(stride_x), _format_float(stride_y)]),
                f"storage_encoding: {self.storage_encoding}",
                f"default_interpolation: {self.default_interpolation}",
                f"default_bounds: {self.default_bounds}",
                f"payload_layout: {_PAYLOAD_LAYOUT}",
                f"payload_endianness: {_PAYLOAD_ENDIANNESS}",
                _HEADER_END_MARKER,
            ]

        payload_offset_bytes_text = "0"
        while True:
            header = "\n".join(build_lines(payload_offset_bytes_text)) + "\n"
            next_payload_offset_bytes_text = str(len(header.encode("ascii")))
            if next_payload_offset_bytes_text == payload_offset_bytes_text:
                return header
            payload_offset_bytes_text = next_payload_offset_bytes_text

    def normalize_points(
        self,
        pixel_coords: np.ndarray,
        *,
        interpolation: InterpolationMode = "bilinear",
        bounds: BoundsMode = "strict",
        return_valid_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Query the LUT for camera-frame rays.

        Args:
            pixel_coords: Pixel coordinates with shape ``(N, 2)``.
            interpolation: Interpolation mode to use.
            bounds: Bounds behavior for out-of-domain pixels.
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
        interpolation: InterpolationMode,
        bounds: BoundsMode,
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
        bounds: BoundsMode,
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
        bounds: BoundsMode,
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
        bounds: BoundsMode,
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
