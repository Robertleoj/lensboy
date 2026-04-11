from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lensboy.camera_models.unproject_lut import (
        StorageEncoding,
        UnprojectLUT,
    )


@dataclass
class CameraModelConfig(ABC):
    pass


@dataclass
class CameraModel(ABC):
    image_width: int
    image_height: int

    @abstractmethod
    def project_points(self, points_in_cam: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame points to pixel coordinates.

        Args:
            points_in_cam: Shape (N, 3).

        Returns:
            Projected pixel coordinates, shape (N, 2).
        """
        ...

    @abstractmethod
    def normalize_points(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera-frame points with z=1.

        Args:
            pixel_coords: Shape (N, 2).

        Returns:
            Normalized points in camera frame, shape (N, 3) with z=1.
        """
        ...

    def get_unproject_lut(
        self,
        *,
        grid_size_wh: tuple[int, int] | None = None,
        pixel_stride: float | tuple[float, float] | None = None,
        storage_encoding: StorageEncoding = "float64_xy",
        num_workers: int | None = None,
    ) -> UnprojectLUT:
        """Build a lookup table that caches `normalize_points()` over a regular grid.

        Args:
            grid_size_wh: Number of cached samples as (width, height). If None,
                a per-pixel grid is used unless `pixel_stride` is given.
            pixel_stride: Approximate sample spacing in pixels. Mutually exclusive
                with `grid_size_wh`.
            storage_encoding: On-disk payload encoding to use when saving the LUT.
            num_workers: Number of worker threads to use while sampling the LUT
                grid.

        Returns:
            A populated unprojection lookup table.
        """
        from lensboy.camera_models.unproject_lut import UnprojectLUT

        return UnprojectLUT.from_camera_model(
            self,
            grid_size_wh=grid_size_wh,
            pixel_stride=pixel_stride,
            storage_encoding=storage_encoding,
            num_workers=num_workers,
        )
