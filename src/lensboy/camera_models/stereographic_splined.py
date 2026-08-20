from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig

if TYPE_CHECKING:
    from lensboy.camera_models.unproject_lut import UnprojectLUT


@dataclass
class StereographicSplinedConfig(CameraModelConfig):
    """Configuration for fitting a StereographicSplined model.

    Attributes:
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        num_knots_x: Number of spline knots along the x axis.
        num_knots_y: Number of spline knots along the y axis.
        initial_focal_length: Initial focal length guess in pixels, or None to
            estimate automatically from the calibration data.
        fov_deg_xy: Explicit FOV in degrees (x, y) for the spline grid. If None,
            the FOV is estimated from a separate distortion-free stereographic fit.
        smoothness_lambda: Strength of the smoothness prior applied to spline
            knots in regions without calibration data.
    """

    image_height: int
    image_width: int

    num_knots_x: int
    num_knots_y: int

    initial_focal_length: float | None = None
    fov_deg_xy: tuple[float, float] | None = None
    smoothness_lambda: float = 1.0


@dataclass
class StereographicSplined(CameraModel):
    """Stereographic camera model with a 2D B-spline distortion field.

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        dx_grid: Spline knot values for x stereographic correction,
            shape (num_knots_y, num_knots_x).
        dy_grid: Spline knot values for y stereographic correction,
            shape (num_knots_y, num_knots_x).
        num_knots_x: Number of spline knots along the x axis.
        num_knots_y: Number of spline knots along the y axis.
        fov_deg_x: Horizontal field of view in degrees.
        fov_deg_y: Vertical field of view in degrees.
    """

    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    dx_grid: np.ndarray
    dy_grid: np.ndarray

    num_knots_x: int
    num_knots_y: int

    fov_deg_x: float
    fov_deg_y: float

    def __post_init__(self):
        assert self.dx_grid.ndim == 2, f"Expected 2D dx_grid, got {self.dx_grid.ndim}D"
        assert np.issubdtype(self.dx_grid.dtype, np.floating), (
            f"Expected floating dtype for dx_grid, got {self.dx_grid.dtype}"
        )
        assert self.dy_grid.ndim == 2, f"Expected 2D dy_grid, got {self.dy_grid.ndim}D"
        assert np.issubdtype(self.dy_grid.dtype, np.floating), (
            f"Expected floating dtype for dy_grid, got {self.dy_grid.dtype}"
        )

    def __repr__(self) -> str:
        return (
            f"StereographicSplined({self.image_width}x{self.image_height}, "
            f"f=[{self.fx:.1f}, {self.fy:.1f}], "
            f"c=[{self.cx:.1f}, {self.cy:.1f}], "
            f"knots={self.num_knots_x}x{self.num_knots_y}, "
            f"fov=[{self.fov_deg_x:.1f}°, {self.fov_deg_y:.1f}°])"
        )

    def _stereographic_parameters(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    def _cpp_model_definition(self) -> lbb.StereographicSplinedModelDefinition:
        return lbb.StereographicSplinedModelDefinition(
            self.image_width,
            self.image_height,
            self.fov_deg_x,
            self.fov_deg_y,
            self.num_knots_x,
            self.num_knots_y,
        )

    def _cpp_params(self) -> lbb.StereographicSplinedIntrinsicsParameters:
        return lbb.StereographicSplinedIntrinsicsParameters(
            self._stereographic_parameters(),
            self.dx_grid,
            self.dy_grid,
        )

    def project_points(self, points_in_cam: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame points to pixel coordinates.

        Args:
            points_in_cam: Shape (N, 3).

        Returns:
            Projected pixel coordinates, shape (N, 2).
        """
        pts = np.asarray(points_in_cam, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3, (
            f"Expected (N, 3) array, got {pts.shape}"
        )
        return lbb.project_stereographic_splined_points(
            self._cpp_model_definition(),
            self._cpp_params(),
            pts,
        )

    def normalize_points(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to camera-frame unit bearing vectors.

        Args:
            pixel_coords: Shape (N, 2).

        Returns:
            Unit bearing vectors in camera frame, shape (N, 3).
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2, (
            f"Expected (N, 2) array, got {pts.shape}"
        )
        return lbb.normalize_stereographic_splined_points(
            self._cpp_model_definition(),
            self._cpp_params(),
            pts,
        )

    def K(self) -> np.ndarray:
        """Return the 3x3 camera intrinsics matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def get_unproject_lut(
        self,
        *,
        grid_size_wh: tuple[int, int] | None = None,
        pixel_stride: float | tuple[float, float] | None = None,
    ) -> UnprojectLUT:
        """Build a lookup table that caches ``normalize_points()`` on a grid.

        Args:
            grid_size_wh: Number of cached samples as ``(width, height)``.
            pixel_stride: Approximate sample spacing in pixels. Mutually
                exclusive with ``grid_size_wh``.

        Returns:
            A populated unprojection lookup table.
        """
        from lensboy.camera_models.unproject_lut import UnprojectLUT

        return UnprojectLUT.from_camera_model(
            self,
            grid_size_wh=grid_size_wh,
            pixel_stride=pixel_stride,
        )

    def save(self, path: Path | str) -> None:
        """Serialize the model to a JSON file.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=4))

    @staticmethod
    def load(path: Path | str) -> StereographicSplined:
        """Load a model from a JSON file written by save().

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed model.
        """
        return StereographicSplined.from_json(json.loads(Path(path).read_text()))

    def to_json(self) -> dict:
        """Serialize the model to a JSON-compatible dict.

        Returns:
            Dict with all model parameters. Spline grids are stored as nested
            lists of shape (num_knots_y, num_knots_x).
        """
        return {
            "type": "stereographic_splined",
            "lensboy-version": _package_version("lensboy"),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dx_grid": self.dx_grid.tolist(),
            "dy_grid": self.dy_grid.tolist(),
            "num_knots_x": self.num_knots_x,
            "num_knots_y": self.num_knots_y,
            "fov_deg_x": self.fov_deg_x,
            "fov_deg_y": self.fov_deg_y,
        }

    @staticmethod
    def from_json(data: dict) -> StereographicSplined:
        """Reconstruct a model from a dict produced by to_json().

        Args:
            data: Dict with all model parameters.

        Returns:
            Reconstructed model.
        """
        return StereographicSplined(
            image_width=data["image_width"],
            image_height=data["image_height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            dx_grid=np.array(data["dx_grid"], dtype=np.float64),
            dy_grid=np.array(data["dy_grid"], dtype=np.float64),
            num_knots_x=data["num_knots_x"],
            num_knots_y=data["num_knots_y"],
            fov_deg_x=data["fov_deg_x"],
            fov_deg_y=data["fov_deg_y"],
        )
