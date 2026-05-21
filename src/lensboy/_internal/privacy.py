from __future__ import annotations

import cv2
import numpy as np

from lensboy._logging import progress
from lensboy.camera_models.base_model import CameraModel
from lensboy.geometry.pose import Pose

BUFFER_M = 50

SAMPLES_PER_EDGE = 50


def _target_rect_outline(target_points: np.ndarray) -> np.ndarray:
    """Sample points along the buffered bounding rectangle of the target in target space.

    Args:
        target_points: 3D target points (assumed planar, z~0), shape (N, 3).

    Returns:
        Sampled outline points in target space with z=0, shape (M, 3).
    """
    x_min, y_min = target_points[:, :2].min(axis=0) - BUFFER_M
    x_max, y_max = target_points[:, :2].max(axis=0) + BUFFER_M

    top = np.linspace([x_min, y_min], [x_max, y_min], SAMPLES_PER_EDGE)
    right = np.linspace([x_max, y_min], [x_max, y_max], SAMPLES_PER_EDGE)
    bottom = np.linspace([x_max, y_max], [x_min, y_max], SAMPLES_PER_EDGE)
    left = np.linspace([x_min, y_max], [x_min, y_min], SAMPLES_PER_EDGE)

    xy = np.concatenate([top, right, bottom, left], axis=0)
    z = np.zeros((xy.shape[0], 1))
    return np.hstack([xy, z])


def privatize_images(
    images: list[np.ndarray],
    cameras_T_target: list[Pose],
    camera_model: CameraModel,
    target_points: np.ndarray,
) -> list[np.ndarray]:
    """Black out image regions outside the calibration target area.

    Projects a buffered rectangle around the target points into each camera view
    and masks everything outside the resulting contour.

    Args:
        images: One image per camera view, each shape (H, W, 3) or (H, W).
        cameras_T_target: Camera-from-target pose for each image.
        camera_model: The camera model used for projection.
        target_points: 3D target points (assumed planar), shape (N, 3).

    Returns:
        Copies of the input images with regions outside the target blacked out.
    """
    outline_target = _target_rect_outline(target_points)

    result = []
    for image, cam_T_target in progress(
        zip(images, cameras_T_target), desc="privatizing"
    ):
        outline_cam = cam_T_target.apply(outline_target)
        outline_px = camera_model.project_points(outline_cam)

        contour = outline_px.astype(np.int32).reshape(-1, 1, 2)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], (255,))

        privatized = image.copy()
        privatized[mask == 0] = 0
        result.append(privatized)

    return result
