from lensboy.analysis.unproject_lut import (
    UnprojectLUTAccuracyReport,
    UnprojectLUTErrorHeatmap,
    UnprojectLUTSampleAccuracy,
    compute_lut_error_heatmap,
    estimate_lut_accuracy,
    sample_lut_accuracy,
)

__all__ = [
    "UnprojectLUTAccuracyReport",
    "UnprojectLUTErrorHeatmap",
    "UnprojectLUTSampleAccuracy",
    "compute_lut_error_heatmap",
    "estimate_lut_accuracy",
    "sample_lut_accuracy",
]

try:
    from lensboy.analysis.plots import (  # noqa: F401
        draw_points,
        plot_detection_coverage,
        plot_distortion_grid,
        plot_projection_diff,
        plot_undistortion,
        plot_unproject_lut_error_heatmap,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "draw_points",
            "plot_detection_coverage",
            "plot_distortion_grid",
            "plot_projection_diff",
            "plot_unproject_lut_error_heatmap",
            "plot_undistortion",
        ]
    )
