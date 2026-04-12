from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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
