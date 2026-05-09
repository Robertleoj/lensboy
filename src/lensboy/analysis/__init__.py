try:
    from lensboy.analysis.plots import (
        draw_points,
        plot_detection_coverage,
        plot_distortion_grid,
        plot_projection_diff,
        plot_undistortion,
        plot_unproject_lut_error_heatmap,
    )
    from lensboy.analysis.unproject_lut import (
        UnprojectLUTAccuracyReport,
        UnprojectLUTErrorHeatmap,
        UnprojectLUTSampleAccuracy,
        compute_lut_error_heatmap,
        estimate_lut_accuracy,
        sample_lut_accuracy,
    )
except ImportError as e:
    raise ImportError(
        "The analysis module requires extra dependencies. "
        "Install them with: pip install lensboy[analysis]"
    ) from e

__all__ = [
    "UnprojectLUTAccuracyReport",
    "UnprojectLUTErrorHeatmap",
    "UnprojectLUTSampleAccuracy",
    "compute_lut_error_heatmap",
    "draw_points",
    "estimate_lut_accuracy",
    "plot_detection_coverage",
    "plot_distortion_grid",
    "plot_projection_diff",
    "plot_undistortion",
    "plot_unproject_lut_error_heatmap",
    "sample_lut_accuracy",
]
