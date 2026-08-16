import numpy as np

from lensboy.common_targets.charuco import _make_frame_from_charuco_detection


def test_make_frame_from_charuco_detection_accepts_opencv_singleton_axes() -> None:
    charuco_ids = np.array([[3], [7], [12]], dtype=np.int32)
    charuco_corners = np.array(
        [[[10.0, 20.0]], [[30.0, 40.0]], [[50.0, 60.0]]],
        dtype=np.float32,
    )

    frame = _make_frame_from_charuco_detection(charuco_ids, charuco_corners)

    np.testing.assert_array_equal(frame.target_point_indices, np.array([3, 7, 12]))
    np.testing.assert_allclose(
        frame.detected_points_in_image,
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32),
    )


def test_make_frame_from_charuco_detection_accepts_flattened_opencv_axes() -> None:
    charuco_ids = np.array([3, 7, 12], dtype=np.int32)
    charuco_corners = np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        dtype=np.float32,
    )

    frame = _make_frame_from_charuco_detection(charuco_ids, charuco_corners)

    np.testing.assert_array_equal(frame.target_point_indices, np.array([3, 7, 12]))
    np.testing.assert_allclose(
        frame.detected_points_in_image,
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32),
    )
