from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field, replace
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lensboy.camera_models.base_model import CameraModel, CameraModelConfig

if TYPE_CHECKING:
    from lensboy.camera_models.unproject_lut import UnprojectLUT

K1, K2, P1, P2, K3, K4, K5, K6, S1, S2, S3, S4, TX, TY = range(14)
_UNDISTORT_POINTS_ITER_CRITERIA = (
    cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
    100,
    1e-14,
)


def _mask(*idx: int) -> np.ndarray:
    m = np.zeros(14, dtype=bool)
    if len(idx) > 0:
        m[list(idx)] = True
    return m


@functools.lru_cache(maxsize=128)
def _camera_matrix_cached(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class OpenCVConfig(CameraModelConfig):
    """Configuration for fitting an OpenCV pinhole + distortion model.

    Preset boolean masks for common distortion subsets are available as class
    attributes: NONE, STANDARD, RADIAL_6, TANGENTIAL, THIN_PRISM, TILT, FULL_14.

    Attributes:
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        initial_focal_length: Initial focal length guess in pixels, or None to
            estimate automatically from the calibration data.
        included_distortion_coefficients: Boolean mask selecting which of the 14
            OpenCV distortion coefficients to optimise, shape (14,).
    """

    image_height: int
    image_width: int

    initial_focal_length: float | None = None
    included_distortion_coefficients: np.ndarray = field(
        default_factory=lambda: OpenCVConfig.STANDARD
    )

    NONE = _mask()
    STANDARD = _mask(K1, K2, P1, P2, K3)
    RADIAL_6 = _mask(K1, K2, K3, K4, K5, K6)
    TANGENTIAL = _mask(P1, P2)
    THIN_PRISM = _mask(S1, S2, S3, S4)
    TILT = _mask(TX, TY)
    FULL_14 = _mask(*range(14))

    def __post_init__(self):
        assert self.included_distortion_coefficients.shape == (14,), (
            f"Expected (14,) mask, got {self.included_distortion_coefficients.shape}"
        )
        assert self.included_distortion_coefficients.dtype == np.bool_, (
            f"Expected bool dtype, got {self.included_distortion_coefficients.dtype}"
        )

    def optimize_mask(self) -> np.ndarray:
        """Return the optimization parameter mask over [fx, fy, cx, cy, *distortion].

        Returns:
            Boolean mask of shape (18,); the first 4 entries (intrinsics) are
            always True.
        """
        mask = np.zeros(18, dtype=bool)

        mask[:4] = True
        mask[4:] = self.included_distortion_coefficients
        return mask

    def get_initial_value(self) -> OpenCV:
        """Construct the initial OpenCV model from this config.

        Raises:
            ValueError: If initial_focal_length is None.
        """
        if self.initial_focal_length is None:
            raise ValueError(
                "initial_focal_length must be set before calling get_initial_value()"
            )
        return OpenCV(
            image_height=self.image_height,
            image_width=self.image_width,
            fx=self.initial_focal_length,
            fy=self.initial_focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            distortion_coeffs=np.zeros(14, dtype=np.float64),
        )


@dataclass
class OpenCV(CameraModel):
    """OpenCV pinhole + distortion camera model.

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        distortion_coeffs: OpenCV distortion vector, shape (14,).
            Inputs shorter than 14 are zero-padded on construction.
    """

    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    distortion_coeffs: np.ndarray

    def __repr__(self) -> str:
        n_dist = int(np.count_nonzero(self.distortion_coeffs))
        return (
            f"OpenCV({self.image_width}x{self.image_height}, "
            f"f=[{self.fx:.1f}, {self.fy:.1f}], "
            f"c=[{self.cx:.1f}, {self.cy:.1f}], "
            f"{n_dist} distortion coeffs)"
        )

    def __post_init__(self):
        dc = np.asarray(self.distortion_coeffs, dtype=np.float64)
        assert dc.ndim == 1 and len(dc) <= 14, (
            "Expected 1-D distortion_coeffs with at most 14 elements, "
            f"got shape {dc.shape}"
        )
        if len(dc) < 14:
            dc = np.pad(dc, (0, 14 - len(dc)))
        self.distortion_coeffs = dc

    def _params(self):
        return [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs]

    def _with_params(self, params: list[float]) -> OpenCV:
        assert len(params) == 4 + 14

        fx, fy, cx, cy = params[:4]

        distortion_coeffs = np.array(params[4:])

        return replace(
            self, fx=fx, fy=fy, cx=cx, cy=cy, distortion_coeffs=distortion_coeffs
        )

    def normalize_points(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera-frame points with z=1.

        Uses cv2.undistortPoints to invert the full OpenCV distortion model.

        Args:
            pixel_coords: Shape (N, 2).

        Returns:
            Normalized points in camera frame, shape (N, 3) with z=1.
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2, (
            f"Expected (N, 2) array, got {pts.shape}"
        )
        undistorted = cv2.undistortPointsIter(
            pts.reshape(-1, 1, 2),
            self._camera_matrix_cached(),
            self.distortion_coeffs,
            R=None,  # type: ignore
            P=None,  # type: ignore
            criteria=_UNDISTORT_POINTS_ITER_CRITERIA,
        ).reshape(-1, 2)
        return np.column_stack([undistorted, np.ones(len(undistorted))])

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
        return cv2.projectPoints(
            points_in_cam,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=self._camera_matrix_cached(),
            distCoeffs=self.distortion_coeffs,
        )[0].reshape(-1, 2)

    def K(self):
        """Return the 3x3 camera intrinsics matrix."""
        return self._camera_matrix_cached().copy()

    def _camera_matrix_cached(self) -> np.ndarray:
        return _camera_matrix_cached(
            float(self.fx),
            float(self.fy),
            float(self.cx),
            float(self.cy),
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

    @property
    def fov_deg_x(self) -> float:
        """Horizontal field of view in degrees.

        Computed by undistorting the image border pixels and measuring the
        angular span of the resulting normalised coordinates.
        """
        return self._compute_fov()[0]

    @property
    def fov_deg_y(self) -> float:
        """Vertical field of view in degrees.

        Computed by undistorting the image border pixels and measuring the
        angular span of the resulting normalised coordinates.
        """
        return self._compute_fov()[1]

    def save(self, path: Path | str) -> None:
        """Serialize the model to a JSON file.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=4))

    @staticmethod
    def load(path: Path | str) -> OpenCV:
        """Load a model from a JSON file written by save().

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed model.
        """
        return OpenCV.from_json(json.loads(Path(path).read_text()))

    def to_json(self) -> dict:
        """Serialize the model to a JSON-compatible dict.

        Returns:
            Dict with all model parameters. Distortion coefficients are stored
            as a list of length 14.
        """
        return {
            "type": "opencv",
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
    def from_json(data: dict) -> OpenCV:
        """Reconstruct a model from a dict produced by to_json().

        Args:
            data: Dict with all model parameters.

        Returns:
            Reconstructed model.
        """
        return OpenCV(
            image_width=data["image_width"],
            image_height=data["image_height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            distortion_coeffs=np.array(data["distortion_coeffs"], dtype=np.float64),
        )

    def _compute_fov(self) -> tuple[float, float]:
        """Compute FOV by undistorting image border pixels."""
        W, H = self.image_width, self.image_height
        n = 200
        xs = np.linspace(0, W, n)
        ys = np.linspace(0, H, n)
        border = np.vstack(
            [
                np.column_stack([xs, np.zeros(n)]),
                np.column_stack([xs, np.full(n, H)]),
                np.column_stack([np.zeros(n), ys]),
                np.column_stack([np.full(n, W), ys]),
            ]
        ).astype(np.float64)

        undistorted = cv2.undistortPoints(
            border.reshape(-1, 1, 2),
            self.K(),
            self.distortion_coeffs,
        ).reshape(-1, 2)

        nx, ny = undistorted[:, 0], undistorted[:, 1]
        fov_x = np.arctan(np.max(nx)) + np.arctan(-np.min(nx))
        fov_y = np.arctan(np.max(ny)) + np.arctan(-np.min(ny))

        return float(np.degrees(fov_x)), float(np.degrees(fov_y))
