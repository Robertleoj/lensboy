import cv2
import numpy as np

from lensboy._logging import log, progress
from lensboy.calibration.calibrate import Frame
from lensboy.image import to_gray


def default_detector_parameters() -> cv2.aruco.DetectorParameters:
    """Create default ArUco detector parameters.

    Returns:
        Detector parameters configured with OpenCV defaults.
    """
    return cv2.aruco.DetectorParameters()


def default_refine_parameters() -> cv2.aruco.RefineParameters:
    """Create default ArUco marker refinement parameters.

    Returns:
        Refinement parameters configured with OpenCV defaults.
    """
    return cv2.aruco.RefineParameters()


def default_charuco_parameters() -> cv2.aruco.CharucoParameters:
    """Create default ChArUco detection parameters.

    Returns:
        ChArUco parameters configured to allow one marker per interpolated corner.
    """
    params = cv2.aruco.CharucoParameters()
    params.minMarkers = 1
    return params


def _detect_charuco(
    img: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    refine_parameters: cv2.aruco.RefineParameters | None = None,
    detector_parameters: cv2.aruco.DetectorParameters | None = None,
    charuco_parameters: cv2.aruco.CharucoParameters | None = None,
) -> Frame | None:
    if charuco_parameters is None:
        charuco_parameters = default_charuco_parameters()
    if refine_parameters is None:
        refine_parameters = default_refine_parameters()
    if detector_parameters is None:
        detector_parameters = default_detector_parameters()

    charuco_detector = cv2.aruco.CharucoDetector(
        board,
        charucoParams=charuco_parameters,
        refineParams=refine_parameters,
        detectorParams=detector_parameters,
    )

    gray = to_gray(img)

    (charuco_corners, charuco_ids, _marker_corners, _marker_ids) = (
        charuco_detector.detectBoard(gray)
    )

    if charuco_ids is None:
        return None

    return _make_frame_from_charuco_detection(charuco_ids, charuco_corners)


def _make_frame_from_charuco_detection(
    charuco_ids: np.ndarray,
    charuco_corners: np.ndarray,
) -> Frame:
    target_point_indices = charuco_ids.reshape(-1)
    detected_points_in_image = charuco_corners.reshape(-1, 2)
    return Frame(target_point_indices, detected_points_in_image)


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
        charuco_parameters: The ChArUco detection parameters, lensboy defaults otherwise.
            You can use this to set the minMarkers (lensboy default 1) and
            tryRefineMarkers (False by default).

    Returns:
        target_points: 3D corner coordinates from the board definition, shape (N, 3).
        frames: Detected frames (only for images where detection succeeded).
        image_indices: Index into the original images list for each frame.
    """
    frames: list[Frame] = []
    image_indices: list[int] = []

    if detector_parameters is None:
        detector_parameters = default_detector_parameters()
    if refine_parameters is None:
        refine_parameters = default_refine_parameters()
    if charuco_parameters is None:
        charuco_parameters = default_charuco_parameters()

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
