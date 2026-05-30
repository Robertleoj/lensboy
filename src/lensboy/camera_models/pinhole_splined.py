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
from lensboy.camera_models.pinhole_remapped import PinholeRemapped


@dataclass
class PinholeSplinedConfig(CameraModelConfig):
    """Configuration for fitting a PinholeSplined model.

    Attributes:
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        num_knots_x: Number of spline knots along the x axis.
        num_knots_y: Number of spline knots along the y axis.
        initial_focal_length: Initial focal length guess in pixels, or None to
            estimate automatically from the calibration data.
        fov_deg_xy: Explicit FOV in degrees (x, y) for the spline grid. If None,
            the FOV is computed from the seed OpenCV model with padding.
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
class PinholeSplined(CameraModel):
    """Pinhole camera model with a 2D B-spline distortion field.

    The distortion is represented as two grids of spline knot values (dx_grid,
    dy_grid) defined over the image domain. Use get_pinhole_model(),
    get_pinhole_model_fov(), or get_pinhole_model_alpha() to obtain an
    undistorted PinholeRemapped view.

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        dx_grid: Spline knot values for the x distortion component,
            shape (num_knots_y, num_knots_x).
        dy_grid: Spline knot values for the y distortion component,
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

    def __repr__(self) -> str:
        return (
            f"PinholeSplined({self.image_width}x{self.image_height}, "
            f"f=[{self.fx:.1f}, {self.fy:.1f}], "
            f"c=[{self.cx:.1f}, {self.cy:.1f}], "
            f"knots={self.num_knots_x}x{self.num_knots_y}, "
            f"fov=[{self.fov_deg_x:.1f}°, {self.fov_deg_y:.1f}°])"
        )

    def __post_init__(self):
        assert self.dx_grid.ndim == 2, f"Expected 2D dx_grid, got {self.dx_grid.ndim}D"
        assert np.issubdtype(self.dx_grid.dtype, np.floating), (
            f"Expected floating dtype for dx_grid, got {self.dx_grid.dtype}"
        )
        assert self.dy_grid.ndim == 2, f"Expected 2D dy_grid, got {self.dy_grid.ndim}D"
        assert np.issubdtype(self.dy_grid.dtype, np.floating), (
            f"Expected floating dtype for dy_grid, got {self.dy_grid.dtype}"
        )

    def save(self, path: Path | str) -> None:
        """Serialize the model to a JSON file.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=4))

    @staticmethod
    def load(path: Path | str) -> PinholeSplined:
        """Load a model from a JSON file written by save().

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed model.
        """
        return PinholeSplined.from_json(json.loads(Path(path).read_text()))

    def to_json(self) -> dict:
        """Serialize the model to a JSON-compatible dict.

        Returns:
            Dict with all model parameters. Spline grids are stored as nested
            lists of shape (num_knots_y, num_knots_x).
        """
        return {
            "type": "pinhole_splined",
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
    def from_json(data: dict) -> PinholeSplined:
        """Reconstruct a model from a dict produced by to_json().

        Args:
            data: Dict with all model parameters.

        Returns:
            Reconstructed model.
        """
        version = data.get("lensboy-version")
        if version is None or int(version.split(".")[0]) < 3:
            raise ValueError(
                "This spline model was created with an incompatible version of "
                "lensboy (< 3.0.0). Please re-calibrate with the current version."
            )
        return PinholeSplined(
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

    @staticmethod
    def _camera_model_name() -> str:
        return "pinhole_splined"

    def _cpp_model_definition(self) -> lbb.PinholeSplinedModelDefinition:
        return lbb.PinholeSplinedModelDefinition(
            self.image_width,
            self.image_height,
            self.fov_deg_x,
            self.fov_deg_y,
            self.num_knots_x,
            self.num_knots_y,
        )

    def _cpp_params(self) -> lbb.PinholeSplinedIntrinsicsParameters:
        return lbb.PinholeSplinedIntrinsicsParameters(
            self._pinhole_parameters(), self.dx_grid, self.dy_grid
        )

    def normalize_points(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera-frame points with z=1.

        Iteratively inverts the spline projection using Newton's method with
        Ceres autodiff Jacobians. Rebuilds per-point when the solution crosses
        a spline cell boundary.

        Args:
            pixel_coords: Shape (N, 2).

        Returns:
            Normalized points in camera frame, shape (N, 3) with z=1.
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2, (
            f"Expected (N, 2) array, got {pts.shape}"
        )
        return lbb.normalize_pinhole_splined_points(
            self._cpp_model_definition(),
            self._cpp_params(),
            pixel_coords=pts,
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

    def project_points(
        self,
        points_in_cam: np.ndarray,
    ) -> np.ndarray:
        """Project 3D camera-frame points to pixel coordinates.

        Args:
            points_in_cam: Shape (N, 3).

        Returns:
            Projected pixel coordinates, shape (N, 2).
        """
        assert points_in_cam.ndim == 2 and points_in_cam.shape[1] == 3, (
            f"Expected (N, 3) array, got {points_in_cam.shape}"
        )
        assert np.issubdtype(points_in_cam.dtype, np.floating), (
            f"Expected floating dtype, got {points_in_cam.dtype}"
        )
        return lbb.project_pinhole_splined_points(
            self._cpp_model_definition(),
            self._cpp_params(),
            points_in_camera=points_in_cam,
        )

    def _pinhole_parameters(self):
        return (self.fx, self.fy, self.cx, self.cy)

    def K(self) -> np.ndarray:
        """Return the 3x3 camera intrinsics matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def get_pinhole_model_fov(
        self,
        target_fov_deg_x: float | None = None,
        target_fov_deg_y: float | None = None,
        image_size_wh: tuple[int, int] | None = None,
    ) -> PinholeRemapped:
        """Build an undistorted pinhole view with a specified field of view.

        Args:
            target_fov_deg_x: Desired horizontal FOV in degrees.
                Defaults to the model's fov_deg_x.
            target_fov_deg_y: Desired vertical FOV in degrees.
                Defaults to the model's fov_deg_y.
            image_size_wh: Output image size as (width, height).
                Defaults to the model's image size.

        Returns:
            Undistorted pinhole model with precomputed remap tables.
        """
        fov_x = self.fov_deg_x
        if target_fov_deg_x is not None:
            fov_x = target_fov_deg_x
        fov_y = self.fov_deg_y
        if target_fov_deg_y is not None:
            fov_y = target_fov_deg_y

        if image_size_wh is None:
            image_size_wh = (self.image_width, self.image_height)

        image_w, image_h = image_size_wh

        fx = image_w / (2 * np.tan(np.deg2rad(fov_x) / 2))
        fy = image_h / (2 * np.tan(np.deg2rad(fov_y) / 2))
        cx = image_w / 2.0
        cy = image_h / 2.0

        return self.get_pinhole_model(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_size_wh=image_size_wh,
        )

    def get_pinhole_model(
        self,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        image_size_wh: tuple[int, int] | None = None,
    ) -> PinholeRemapped:
        """Build an undistorted pinhole view with explicit intrinsics.

        Args:
            fx: Horizontal focal length. Defaults to the model's fx.
            fy: Vertical focal length. Defaults to the model's fy.
            cx: Principal point x. Defaults to the model's cx.
            cy: Principal point y. Defaults to the model's cy.
            image_size_wh: Output image size as (width, height).
                Defaults to the model's image size.

        Returns:
            Undistorted pinhole model with precomputed remap tables.
        """
        if fx is None:
            fx = self.fx
        if fy is None:
            fy = self.fy
        if cx is None:
            cx = self.cx
        if cy is None:
            cy = self.cy

        if image_size_wh is None:
            image_size_wh = (self.image_width, self.image_height)

        pinhole_parameters = (fx, fy, cx, cy)

        map_x, map_y = lbb.make_undistortion_maps_pinhole_splined(
            self._cpp_model_definition(),
            self._cpp_params(),
            np.array(pinhole_parameters, dtype=float),
            image_size_wh,
        )

        return PinholeRemapped(
            image_width=image_size_wh[0],
            image_height=image_size_wh[1],
            input_image_width=self.image_width,
            input_image_height=self.image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            map_x=map_x,
            map_y=map_y,
        )

    def get_pinhole_model_alpha(
        self,
        alpha: float,
        image_size_wh: tuple[int, int] | None = None,
    ) -> PinholeRemapped:
        """Build an undistorted pinhole view with alpha-controlled cropping.

        Unprojects the image border through the spline model to determine the
        valid field of view, then interpolates between the inner bounding box
        (no black pixels) and the outer bounding box (all source pixels visible).

        Args:
            alpha: Scaling parameter in [0, 1]. 0 crops to only valid pixels,
                1 keeps all pixels (with black borders).
            image_size_wh: Output image size as (width, height).
                Defaults to the model's image size.

        Returns:
            Undistorted pinhole model with precomputed remap tables.
        """
        if image_size_wh is None:
            image_size_wh = (self.image_width, self.image_height)

        W, H = self.image_width, self.image_height
        n = 500

        xs = np.linspace(0, W - 1, n)
        ys = np.linspace(0, H - 1, n)

        top = np.column_stack([xs, np.zeros(n)])
        bottom = np.column_stack([xs, np.full(n, H - 1)])
        left = np.column_stack([np.zeros(n), ys])
        right = np.column_stack([np.full(n, W - 1), ys])

        top_norm = self.normalize_points(top)[:, :2]
        bottom_norm = self.normalize_points(bottom)[:, :2]
        left_norm = self.normalize_points(left)[:, :2]
        right_norm = self.normalize_points(right)[:, :2]

        all_norm = np.vstack([top_norm, bottom_norm, left_norm, right_norm])

        # Inner box (alpha=0): largest rect with no black pixels.
        # Each edge curves inward, so the tightest constraint per side is:
        inner_left = np.max(left_norm[:, 0])
        inner_right = np.min(right_norm[:, 0])
        inner_top = np.max(top_norm[:, 1])
        inner_bottom = np.min(bottom_norm[:, 1])

        # Outer box (alpha=1): bounding box of all border points.
        outer_left = np.min(all_norm[:, 0])
        outer_right = np.max(all_norm[:, 0])
        outer_top = np.min(all_norm[:, 1])
        outer_bottom = np.max(all_norm[:, 1])

        # Interpolate between inner and outer bounds.
        bound_left = inner_left + alpha * (outer_left - inner_left)
        bound_right = inner_right + alpha * (outer_right - inner_right)
        bound_top = inner_top + alpha * (outer_top - inner_top)
        bound_bottom = inner_bottom + alpha * (outer_bottom - inner_bottom)

        out_w, out_h = image_size_wh
        fx = out_w / (bound_right - bound_left)
        fy = out_h / (bound_bottom - bound_top)
        cx = -bound_left * fx
        cy = -bound_top * fy

        return self.get_pinhole_model(
            fx=fx, fy=fy, cx=cx, cy=cy, image_size_wh=image_size_wh
        )
