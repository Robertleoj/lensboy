import cv2
import numpy as np

from lensboy._logging import log, progress
from lensboy.calibration.calibrate import Frame
from lensboy.image import to_gray


def _detect_charuco(
    img: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    refine_parameters: cv2.aruco.RefineParameters | None = None,
    detector_parameters: cv2.aruco.DetectorParameters | None = None,
    charuco_parameters: cv2.aruco.CharucoParameters | None = None,
) -> Frame | None:

    charuco_params = (
        cv2.arcuo.CharucoParameters()
        if charuco_parameters is None
        else charuco_parameters
    )
    refine_params = (
        cv2.aruco.RefineParameters() if refine_parameters is None else refine_parameters
    )
    detect_params = (
        cv2.aruco.DetectorParameters()
        if detector_parameters is None
        else detector_parameters
    )

    charuco_detector = cv2.aruco.CharucoDetector(
        board,
        charucoParams=charuco_params,
        refineParams=refine_params,
        detectorParams=detect_params,
    )

    gray = to_gray(img)

    (charuco_corners, charuco_ids, _marker_corners, _marker_ids) = (
        charuco_detector.detectBoard(gray)
    )

    if charuco_ids is None:
        return None

    return Frame(charuco_ids.squeeze(1), charuco_corners.squeeze(1))


def extract_frames_from_charuco(
    board: cv2.aruco.CharucoBoard,
    images: list[np.ndarray],
    detector_parameters: cv2.aruco.DetectorParameters | None = None,
    refine_parameters: cv2.aruco.RefineParameters | None = None,
    charuco_parameters: cv2.aruco.CharucoParameters | None = None,
) -> tuple[np.ndarray, list[Frame], list[int]]:
    """Detect ChArUco corners in a batch of images.

    Images where detection fails are silently skipped.

    Args:
        board: The ChArUco board definition.
        images: Calibration images, each of shape (H, W) or (H, W, C).
            If you have an RGB camera with bayer filter and have raw bayer,
            you should reconstruct the luminance image (e.g. via binning) and pass that.
            Higher bit-per-pixel is better for subpixel corner estimation.
        detector_parameters: The detector parameters, default otherwise.
            You can use this to e.g. use subpixel refinment on aruco corners
            (default off).
        refine_parameters: The refine parameters for charuco detection.
            Leave this default unless you know what you are doing.
        charuco_parameters: The charuco detection parameters, default otherwise.
            You can use this to set the minMarkers (default, 2 is recommended) and
            tryRefineMarkers (False by default).

    Returns:
        target_points: 3D corner coordinates from the board definition, shape (N, 3).
        frames: Detected frames (only for images where detection succeeded).
        image_indices: Index into the original images list for each frame.
    """
    frames: list[Frame] = []
    image_indices: list[int] = []

    for i, img in enumerate(progress(images, desc="Detecting charuco")):
        frame = _detect_charuco(
            img,
            board,
            detector_parameters=detector_parameters,
            refine_parameters=refine_parameters,
            charuco_parameters=charuco_parameters,
        )
        if frame is not None:
            frames.append(frame)
            image_indices.append(i)

    log(f"Detected charuco in {len(frames)}/{len(images)} images")

    target_points = np.array(board.getChessboardCorners())

    return target_points, frames, image_indices
