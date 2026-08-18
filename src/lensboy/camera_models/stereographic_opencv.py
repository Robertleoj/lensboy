from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig
from lensboy.camera_models.opencv import K1, K2, K3, K4, K5, K6, P1, P2, S1, S2, S3, S4

if TYPE_CHECKING:
    from lensboy.camera_models.unproject_lut import UnprojectLUT


def _mask(*idx: int) -> np.ndarray:
    m = np.zeros(14, dtype=bool)
    if len(idx) > 0:
        m[list(idx)] = True
    return m


@functools.lru_cache(maxsize=128)
def _camera_matrix_cached(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class StereographicOpenCVConfig(CameraModelConfig):
    """Configuration for fitting a stereographic + OpenCV-distortion model.

    Attributes:
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        initial_focal_length: Initial focal length guess in pixels, or None to
            estimate automatically from the calibration data.
        included_distortion_coefficients: Boolean mask selecting which OpenCV
            distortion coefficients to optimise in the stereographic plane,
            shape (14,).
    """

    image_height: int
    image_width: int
    initial_focal_length: float | None = None
    included_distortion_coefficients: np.ndarray = field(
        default_factory=lambda: StereographicOpenCVConfig.FULL_14
    )

    NONE = _mask()
    STANDARD = _mask(K1, K2, P1, P2, K3)
    RADIAL_6 = _mask(K1, K2, K3, K4, K5, K6)
    TANGENTIAL = _mask(P1, P2)
    THIN_PRISM = _mask(S1, S2, S3, S4)
    FULL_14 = _mask(*range(14))

    def __post_init__(self):
        assert self.included_distortion_coefficients.shape == (14,), (
            f"Expected (14,) mask, got {self.included_distortion_coefficients.shape}"
        )
        assert self.included_distortion_coefficients.dtype == np.bool_, (
            f"Expected bool dtype, got {self.included_distortion_coefficients.dtype}"
        )

    def optimize_mask(self) -> np.ndarray:
        """Return the optimization mask over [fx, fy, cx, cy, *distortion].

        Returns:
            Boolean mask of shape (18,); the first 4 entries are always True.
        """
        mask = np.zeros(18, dtype=bool)
        mask[:4] = True
        mask[4:] = self.included_distortion_coefficients
        return mask

    def get_initial_value(self) -> StereographicOpenCV:
        """Construct the initial model from this config.

        Raises:
            ValueError: If initial_focal_length is None.

        Returns:
            Initial model with zero distortion coefficients.
        """
        if self.initial_focal_length is None:
            raise ValueError(
                "initial_focal_length must be set before calling get_initial_value()"
            )
        return StereographicOpenCV(
            image_height=self.image_height,
            image_width=self.image_width,
            fx=self.initial_focal_length,
            fy=self.initial_focal_length,
            cx=self.image_width / 2.0,
            cy=self.image_height / 2.0,
            distortion_coeffs=np.zeros(14, dtype=np.float64),
        )


@dataclass
class StereographicOpenCV(CameraModel):
    """Stereographic projection with OpenCV distortion in the stereographic plane.

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        distortion_coeffs: OpenCV distortion coefficients, shape (14,).
    """

    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    distortion_coeffs: np.ndarray

    def __post_init__(self):
        coeffs = np.asarray(self.distortion_coeffs, dtype=np.float64)
        assert coeffs.ndim == 1 and len(coeffs) <= 14, (
            "Expected 1-D distortion_coeffs with at most 14 elements, "
            f"got shape {coeffs.shape}"
        )
        if len(coeffs) < 14:
            coeffs = np.pad(coeffs, (0, 14 - len(coeffs)))
        self.distortion_coeffs = coeffs

    def __repr__(self) -> str:
        n_dist = int(np.count_nonzero(self.distortion_coeffs))
        return (
            f"StereographicOpenCV({self.image_width}x{self.image_height}, "
            f"f=[{self.fx:.1f}, {self.fy:.1f}], "
            f"c=[{self.cx:.1f}, {self.cy:.1f}], "
            f"{n_dist} distortion coeffs)"
        )

    def _params(self) -> list[float]:
        return [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs]

    def _with_params(self, params: list[float]) -> StereographicOpenCV:
        assert len(params) == 18
        return StereographicOpenCV(
            image_width=self.image_width,
            image_height=self.image_height,
            fx=params[0],
            fy=params[1],
            cx=params[2],
            cy=params[3],
            distortion_coeffs=np.array(params[4:], dtype=np.float64),
        )

    def _cpp_model_definition(self) -> lbb.StereographicOpenCVModelDefinition:
        return lbb.StereographicOpenCVModelDefinition(
            self.image_width,
            self.image_height,
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
        return lbb.project_stereographic_opencv_points(
            self._cpp_model_definition(),
            np.array(self._params(), dtype=np.float64),
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
        return lbb.normalize_stereographic_opencv_points(
            self._cpp_model_definition(),
            np.array(self._params(), dtype=np.float64),
            pts,
        )

    def K(self) -> np.ndarray:
        """Return the 3x3 camera intrinsics matrix."""
        return _camera_matrix_cached(
            float(self.fx),
            float(self.fy),
            float(self.cx),
            float(self.cy),
        ).copy()

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
    def load(path: Path | str) -> StereographicOpenCV:
        """Load a model from a JSON file written by save().

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed model.
        """
        return StereographicOpenCV.from_json(json.loads(Path(path).read_text()))

    def to_json(self) -> dict:
        """Serialize the model to a JSON-compatible dict.

        Returns:
            Dict with all model parameters.
        """
        return {
            "type": "stereographic_opencv",
            "lensboy-version": _package_version("lensboy"),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_coeffs": self.distortion_coeffs.tolist(),
        }

    @staticmethod
    def from_json(data: dict) -> StereographicOpenCV:
        """Reconstruct a model from a dict produced by to_json().

        Args:
            data: Dict with all model parameters.

        Returns:
            Reconstructed model.
        """
        return StereographicOpenCV(
            image_width=data["image_width"],
            image_height=data["image_height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            distortion_coeffs=np.array(data["distortion_coeffs"], dtype=np.float64),
        )
