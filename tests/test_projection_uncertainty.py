import numpy as np

import lensboy as lb


def _make_target() -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-70.0, 70.0, 6), np.linspace(-50.0, 50.0, 5))
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])


def _make_frames(
    model: lb.OpenCV,
    target_points: np.ndarray,
    *,
    num_frames: int = 8,
    seed: int = 1,
) -> list[lb.Frame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(target_points), dtype=np.int32)
    frames: list[lb.Frame] = []
    for _ in range(num_frames):
        pose = lb.Pose.from_rotvec_trans(
            rotvec=rng.normal(scale=0.15, size=3),
            trans=np.array(
                [
                    rng.normal(scale=10.0),
                    rng.normal(scale=10.0),
                    rng.uniform(450.0, 650.0),
                ]
            ),
        )
        points_in_camera = pose.apply(target_points)
        detected = model.project_points(points_in_camera)
        detected += rng.normal(scale=0.05, size=detected.shape)
        frames.append(
            lb.Frame(
                target_point_indices=indices,
                detected_points_in_image=detected.astype(np.float64),
            )
        )
    return frames


def _check_covariances(covariances: np.ndarray) -> None:
    assert covariances.shape[1:] == (2, 2)
    assert np.all(np.isfinite(covariances))
    np.testing.assert_allclose(covariances, np.swapaxes(covariances, 1, 2), atol=1e-8)
    eigvals = np.linalg.eigvalsh(covariances)
    assert np.min(eigvals) > -1e-8


def test_opencv_projection_uncertainty_shape_and_grouping() -> None:
    """OpenCV uncertainty returns full output covariances grouped by frame."""
    target_points = _make_target()
    model = lb.OpenCV(640, 480, 500.0, 500.0, 320.0, 240.0, np.zeros(14))
    frames = _make_frames(model, target_points, num_frames=7)

    result = lb.calibrate_camera(
        target_points,
        frames,
        lb.OpenCVConfig(
            image_height=480,
            image_width=640,
            included_distortion_coefficients=lb.OpenCVConfig.NONE,
        ),
        estimate_target_warp=False,
        outlier_threshold_stddevs=None,
    )

    rays = np.array([[0.0, 0.0, 1.0], [0.1, 0.05, 1.0], [-0.08, 0.03, 1.0]])
    uncertainty = result.projection_uncertainty(rays)

    assert uncertainty.covariances_px.shape == (3, 2, 2)
    assert uncertainty.trace_std_px.shape == (3,)
    assert uncertainty.metadata["num_groups"] == 7
    assert uncertainty.metadata["opencv_active_intrinsic_params"] == 4
    assert uncertainty.metadata["opencv_inactive_intrinsic_params"] == 14
    _check_covariances(uncertainty.covariances_px)


def test_spline_projection_uncertainty_shape() -> None:
    """Spline uncertainty supports the existing PinholeSplined model."""
    target_points = _make_target()
    model = lb.OpenCV(
        640,
        480,
        500.0,
        500.0,
        320.0,
        240.0,
        np.array([-0.05, 0.01, 0.0, 0.0, 0.0]),
    )
    frames = _make_frames(model, target_points, num_frames=8, seed=2)

    config = lb.PinholeSplinedConfig(
        image_height=480,
        image_width=640,
        num_knots_x=8,
        num_knots_y=6,
        smoothness_lambda=2.5,
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        config,
        estimate_target_warp=False,
        outlier_threshold_stddevs=None,
    )

    rays = np.array([[0.0, 0.0, 1.0], [0.1, 0.05, 1.0]])
    uncertainty = result.projection_uncertainty(rays)

    assert uncertainty.covariances_px.shape == (2, 2, 2)
    assert uncertainty.metadata["fixed_pinhole_parameters"] is True
    assert uncertainty.metadata["spline_smoothness_lambda"] == 2.5
    _check_covariances(uncertainty.covariances_px)


def test_projection_uncertainty_plot_smoke() -> None:
    """Projection uncertainty plotting returns a Matplotlib figure."""
    import matplotlib.pyplot as plt

    target_points = _make_target()
    model = lb.OpenCV(640, 480, 500.0, 500.0, 320.0, 240.0, np.zeros(14))
    frames = _make_frames(model, target_points, num_frames=6, seed=4)
    result = lb.calibrate_camera(
        target_points,
        frames,
        lb.OpenCVConfig(
            image_height=480,
            image_width=640,
            included_distortion_coefficients=lb.OpenCVConfig.NONE,
        ),
        estimate_target_warp=False,
        outlier_threshold_stddevs=None,
    )

    fig = result.plot_projection_uncertainty(grid_density=8, return_figure=True)
    assert fig is not None
    plt.close(fig)
