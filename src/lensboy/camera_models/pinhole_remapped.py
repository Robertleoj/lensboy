from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version as _package_version
from pathlib import Path

import cv2
import numpy as np

from lensboy.camera_models.base_model import CameraModel


@dataclass
class PinholeRemapped(CameraModel):
    """An undistorted pinhole view of a distorted camera model.

    Stores precomputed remap tables (map_x, map_y) that map each output pixel
    back to its location in the original distorted image. Use undistort() to
    remap images and project_points() to project into the undistorted image.

    Attributes:
        image_width: Output (undistorted) image width in pixels.
        image_height: Output (undistorted) image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        map_x: Per-pixel source x coordinate in the distorted image, shape (H, W).
        map_y: Per-pixel source y coordinate in the distorted image, shape (H, W).
        input_image_width: Expected input (distorted) image width in pixels.
        input_image_height: Expected input (distorted) image height in pixels.
    """

    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    map_x: np.ndarray
    map_y: np.ndarray

    input_image_width: int
    input_image_height: int

    def save(self, dir_path: Path | str) -> None:
        """Serialize the model to a directory.

        Writes model.json with scalar parameters, and map_x.npy / map_y.npy
        for the remap arrays.

        Args:
            dir_path: Destination directory (created if it doesn't exist).
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        d, map_x, map_y = self.to_json()
        (p / "model.json").write_text(json.dumps(d, indent=4))
        np.save(p / "map_x.npy", map_x)
        np.save(p / "map_y.npy", map_y)

    @staticmethod
    def load(dir_path: Path | str) -> PinholeRemapped:
        """Load a model from a directory written by save().

        Args:
            dir_path: Directory containing model.json, map_x.npy, map_y.npy.

        Returns:
            Reconstructed model.
        """
        p = Path(dir_path)
        data = json.loads((p / "model.json").read_text())
        map_x = np.load(p / "map_x.npy")
        map_y = np.load(p / "map_y.npy")
        return PinholeRemapped.from_json(data, map_x, map_y)

    def to_json(self) -> tuple[dict, np.ndarray, np.ndarray]:
        """Serialize scalar parameters and return the remap arrays separately.

        Returns:
            Tuple of (dict with scalar parameters, map_x of shape (H, W),
            map_y of shape (H, W)).
        """
        d = {
            "type": "pinhole_remapped",
            "lensboy-version": _package_version("lensboy"),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "input_image_width": self.input_image_width,
            "input_image_height": self.input_image_height,
        }
        return d, self.map_x, self.map_y

    @staticmethod
    def from_json(data: dict, map_x: np.ndarray, map_y: np.ndarray) -> PinholeRemapped:
        """Reconstruct a model from a dict and remap arrays.

        Args:
            data: Dict with scalar parameters, as produced by to_json().
            map_x: Per-pixel source x coordinates, shape (H, W).
            map_y: Per-pixel source y coordinates, shape (H, W).

        Returns:
            Reconstructed model.
        """
        return PinholeRemapped(
            image_width=data["image_width"],
            image_height=data["image_height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            map_x=map_x,
            map_y=map_y,
            input_image_width=data["input_image_width"],
            input_image_height=data["input_image_height"],
        )

    def __repr__(self) -> str:
        return (
            f"PinholeRemapped({self.image_width}x{self.image_height}, "
            f"f=[{self.fx:.1f}, {self.fy:.1f}], "
            f"c=[{self.cx:.1f}, {self.cy:.1f}], "
            f"input={self.input_image_width}x{self.input_image_height})"
        )

    def __post_init__(self):
        assert self.map_x.ndim == 2, f"Expected 2D map_x, got {self.map_x.ndim}D"
        assert np.issubdtype(self.map_x.dtype, np.floating), (
            f"Expected floating dtype for map_x, got {self.map_x.dtype}"
        )
        assert self.map_y.ndim == 2, f"Expected 2D map_y, got {self.map_y.ndim}D"
        assert np.issubdtype(self.map_y.dtype, np.floating), (
            f"Expected floating dtype for map_y, got {self.map_y.dtype}"
        )

    def undistort(
        self,
        image: np.ndarray,
        *,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int | float | tuple[int, int, int] | tuple[int, int, int, int] = 0,
    ) -> np.ndarray:
        """Remap a distorted input image to the undistorted pinhole view.

        Args:
            image: Input image, shape (H, W) or (H, W, C).
            interpolation: OpenCV interpolation flag.
            border_mode: OpenCV border mode flag.
            border_value: Fill value for out-of-bounds pixels.

        Returns:
            Undistorted image, shape (image_height, image_width) or
            (image_height, image_width, C).
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"image must be HxW or HxWxC, got {image.shape}")

        map_x = np.asarray(self.map_x, dtype=np.float32, order="C")
        map_y = np.asarray(self.map_y, dtype=np.float32, order="C")

        h, w = image.shape[:2]
        if (h, w) != (self.input_image_height, self.input_image_width):
            raise ValueError(
                f"image shape {(h, w)} != model image size "
                f"{(self.input_image_height, self.input_image_width)}"
            )

        border_scalar = (float(border_value),) if isinstance(border_value, (int, float)) else border_value
        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_scalar,
        )

    def normalize_points(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera-frame points with z=1.

        Args:
            pixel_coords: Shape (N, 2).

        Returns:
            Normalized points in camera frame, shape (N, 3) with z=1.
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2, (
            f"Expected (N, 2) array, got {pts.shape}"
        )
        x = (pts[:, 0] - self.cx) / self.fx
        y = (pts[:, 1] - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=1)

    @property
    def fov_deg_x(self) -> float:
        """Horizontal field of view in degrees."""
        return float(2 * np.rad2deg(np.arctan(self.image_width / (2 * self.fx))))

    @property
    def fov_deg_y(self) -> float:
        """Vertical field of view in degrees."""
        return float(2 * np.rad2deg(np.arctan(self.image_height / (2 * self.fy))))

    def K(self) -> np.ndarray:
        """Return the 3x3 camera intrinsics matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def project_points(self, points_in_cam: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame points into the undistorted image.

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
        points_cam = np.asarray(points_in_cam, dtype=np.float64)

        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]

        x = X / Z
        y = Y / Z

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.stack([u, v], axis=1)
