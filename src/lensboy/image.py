import cv2
import numpy as np


def to_gray(
    img: np.ndarray,
) -> np.ndarray:
    """Convert a grayscale or color image to a single-channel grayscale image.

    Args:
        img: Input image, shape (H, W), (H, W, 1), or (H, W, 3).

    Returns:
        Grayscale image with shape (H, W).
    """
    assert img.dtype == np.uint8, f"Expected uint8 image, got {img.dtype}"
    assert img.ndim in (2, 3), f"Expected 2D or 3D image, got {img.ndim}D"
    if len(img.shape) == 2 or img.shape[2] == 1:
        return img.reshape(img.shape[:2])

    if img.shape[2] != 3:
        raise ValueError("Image must have one or three channels")

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_color(
    img: np.ndarray,
) -> np.ndarray:
    """Convert a grayscale or color image to a 3-channel RGB image.

    Args:
        img: Input image, shape (H, W), (H, W, 1), or (H, W, 3).

    Returns:
        RGB image with shape (H, W, 3).
    """
    assert img.dtype == np.uint8, f"Expected uint8 image, got {img.dtype}"
    assert img.ndim in (2, 3), f"Expected 2D or 3D image, got {img.ndim}D"

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    if img.ndim == 3 and img.shape[2] != 1:
        raise ValueError("Image must have one or three channels")

    gray = img.reshape(img.shape[:2])
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
