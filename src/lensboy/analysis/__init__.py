try:
    from lensboy.analysis.plots import (
        draw_points,
        plot_detection_coverage,
        plot_distortion_grid,
        plot_projection_diff,
        plot_undistortion,
    )
    from lensboy.analysis.unproject_lut import (
        UnprojectLUTErrorHeatmap,
        compute_lut_error_heatmap,
    )
except ImportError as e:
    raise ImportError(
        "The analysis module requires extra dependencies. "
        "Install them with: pip install lensboy[analysis]"
    ) from e

__all__ = [
    "UnprojectLUTErrorHeatmap",
    "compute_lut_error_heatmap",
    "draw_points",
    "plot_detection_coverage",
    "plot_distortion_grid",
    "plot_projection_diff",
    "plot_undistortion",
]
