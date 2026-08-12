import cv2
import numpy as np

from lensboy._logging import log, progress
from lensboy.calibration.calibrate import Frame
from lensboy.image import to_gray


def _detect_checkerboard(img: np.ndarray, pattern_size: tuple[int, int]) -> Frame | None:
    gray = to_gray(img)

    found, corners = cv2.findChessboardCornersSB(gray,
                                                pattern_size,
                                                flags=(
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE |
                                                    cv2.CALIB_CB_EXHAUSTIVE |
                                                    cv2.CALIB_CB_ACCURACY))

    if not found or corners is None:
        return None

    corners = np.asarray(corners).reshape(-1, 2)
    expected = pattern_size[0] * pattern_size[1]

    if len(corners) != expected:
        return None

    ids = np.arange(expected)

    return Frame(ids, corners)


def extract_frames_from_checkerboard(pattern_size: tuple[int, int],
                                    square_size: float,
                                    images: list[np.ndarray]) -> tuple[np.ndarray, list[Frame], list[int]]:

    """Detect checkerboard corners in a batch of images. Images where detection fails are silently skipped.
    Args:
        pattern_size: Number of inner checkerboard corners as (columns, rows).
        square_size: Physical size of one checkerboard square. The returned
            target points use the same unit as this value.
        images: Calibration images, each of shape (H, W) or (H, W, C).
            If you have an RGB camera with a Bayer filter and have raw Bayer,
            you should reconstruct the luminance image (e.g. via binning) and
            pass that. Higher bit-per-pixel is better for subpixel corner
            estimation.

    Returns:
        target_points: 3D checkerboard corner coordinates, shape (N, 3).
        frames: Detected frames (only for images where detection succeeded).
        image_indices: Index into the original images list for each frame.
    """

    frames = []
    image_indices = []

    for i, img in enumerate(progress(images, desc="Detecting checkerboard")):
        frame = _detect_checkerboard(img, pattern_size)

        if frame is not None:
            frames.append(frame)
            image_indices.append(i)

    log(f"Detected checkerboard in {len(frames)}/{len(images)} images")

    cols, rows = pattern_size

    target_points = np.zeros((cols * rows, 3), dtype=np.float64)
    target_points[:, :2] = (np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size)

    return target_points, frames, image_indices