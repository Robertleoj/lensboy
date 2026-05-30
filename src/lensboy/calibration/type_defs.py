from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.opencv import OpenCV
from lensboy.camera_models.pinhole_splined import (
    PinholeSplined,
)
from lensboy.geometry.pose import Pose


@dataclass
class WarpCoordinates:
    """Maps target points into a planar frame scaled to [-1, 1] for warp estimation.

    Attributes:
        target_from_warp_frame: Pose placing the warp frame in target coordinates.
            The target must be coplanar with its xy plane.
        x_scale: Half-extent of the target along the warp x-axis, in target units.
        y_scale: Half-extent of the target along the warp y-axis, in target units.
    """

    target_from_warp_frame: Pose
    x_scale: float
    y_scale: float

    def _to_cpp(self) -> lbb.WarpCoordinates:
        """Serialise to the C++ bindings representation."""
        return lbb.WarpCoordinates(
            target_from_warp_frame=self.target_from_warp_frame._to_cpp(),
            x_scale=self.x_scale,
            y_scale=self.y_scale,
        )

    @staticmethod
    def _from_cpp(cpp: lbb.WarpCoordinates) -> WarpCoordinates:
        """Deserialise from the C++ bindings representation."""
        return WarpCoordinates(
            target_from_warp_frame=Pose._from_cpp(cpp.target_from_warp_frame),
            x_scale=cpp.x_scale,
            y_scale=cpp.y_scale,
        )


@dataclass
class TargetWarp:
    """Legendre-polynomial warp applied to the calibration target to model non-planarity.

    The warp displaces each point along the target normal by
    a*P2(x) + b*P2(y) + c*P2(x)*P2(y) + d*P4(x) + e*P4(y),
    where x and y are scaled to [-1, 1] and P2, P4 are Legendre polynomials.

    Attributes:
        warp_coordinates: Frame and scale used to map target points to [-1, 1].
        object_warp: Legendre warp coefficients (a, b, c, d, e).
    """

    warp_coordinates: WarpCoordinates
    object_warp: tuple[float, float, float, float, float]

    def __repr__(self) -> str:
        c = self.object_warp
        return (
            f"TargetWarp(coeffs=["
            f"{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, "
            f"{c[3]:.4f}, {c[4]:.4f}])"
        )

    def warp_target(self, target_points: np.ndarray) -> np.ndarray:
        """Apply the Legendre warp to 3D target points.

        Args:
            target_points: Shape (N, 3).

        Returns:
            Warped points in the target frame, shape (N, 3).
        """
        return lbb.warp_target_points(
            self.warp_coordinates._to_cpp(),
            np.array(self.object_warp),
            target_points,
        )

    def max_deflection(self, target_points: np.ndarray) -> float:
        """Peak-to-peak z-deflection caused by the warp, in target units.

        Args:
            target_points: Original 3D target coordinates, shape (N, 3).

        Returns:
            The spread (max minus min) of warp-induced z change.
        """
        warp_frame_from_target = self.warp_coordinates.target_from_warp_frame.inverse()
        z_original = warp_frame_from_target.apply(target_points)[:, 2]
        warped = self.warp_target(target_points)
        z_warped = warp_frame_from_target.apply(warped)[:, 2]
        dz = z_warped - z_original
        return float(np.max(dz) - np.min(dz))


@dataclass
class Frame:
    """Detected calibration target points in a single image.

    Attributes:
        target_point_indices: Index into the target point array for each detection,
            shape (N,).
        detected_points_in_image: Corresponding pixel coordinates, shape (N, 2).
    """

    target_point_indices: np.ndarray
    detected_points_in_image: np.ndarray

    def __post_init__(self):
        assert (
            self.target_point_indices.shape[0] == self.detected_points_in_image.shape[0]
        ), (
            "Expected target_point_indices to have the "
            "length shape as detected_points_in_image"
        )

        assert self.target_point_indices.ndim == 1, (
            f"Expected 1D target_point_indices, got {self.target_point_indices.ndim}D"
        )
        assert np.issubdtype(self.target_point_indices.dtype, np.integer), (
            "Expected integer dtype for target_point_indices, ",
            f"got {self.target_point_indices.dtype}",
        )

        assert (
            self.detected_points_in_image.ndim == 2
            and self.detected_points_in_image.shape[1] == 2
        ), f"Expected (N, 2) points in image, got {self.detected_points_in_image.shape}"
        assert np.issubdtype(self.detected_points_in_image.dtype, np.floating), (
            "Expected floating dtype for points, "
            f"got {self.detected_points_in_image.dtype}"
        )

    def _to_cpp(self) -> tuple[list[int], list[np.ndarray]]:
        return (self.target_point_indices.tolist(), list(self.detected_points_in_image))

    def __repr__(self) -> str:
        return f"Frame({len(self)} detections)"

    def __len__(self):
        return self.target_point_indices.shape[0]


IntrinsicsT = TypeVar("IntrinsicsT", OpenCV, PinholeSplined)


@dataclass
class FrameDiagnostics:
    """Per-image reprojection diagnostics computed after calibration.

    Attributes:
        projected_points: Model-projected pixel coordinates, shape (N, 2).
        residuals: Pixel-space residuals (detected minus projected), shape (N, 2).
        inlier_mask: Boolean mask indicating inlier points, shape (N,).
    """

    projected_points: np.ndarray
    residuals: np.ndarray
    inlier_mask: np.ndarray

    def __repr__(self) -> str:
        n = len(self.residuals)
        n_inliers = int(np.count_nonzero(self.inlier_mask))
        if n_inliers > 0:
            mean_res = float(
                np.mean(np.linalg.norm(self.residuals[self.inlier_mask], axis=1))
            )
        else:
            mean_res = 0.0
        return (
            f"FrameDiagnostics({n_inliers}/{n} inliers, mean_residual={mean_res:.3f}px)"
        )


@dataclass
class CalibrationResult(Generic[IntrinsicsT]):
    """Output of camera calibration.

    Attributes:
        camera_model: The calibrated camera model.
        cameras_from_target: One pose per image (camera-from-target).
        frame_diagnostics: Per-image reprojection diagnostics, one per input image.
        frames: Input detection frames used for calibration.
        target_points: 3D calibration target points, shape (M, 3).
        target_warp: Estimated target warp, or None if not estimated.
    """

    camera_model: IntrinsicsT
    cameras_from_target: list[Pose | None]
    frame_diagnostics: list[FrameDiagnostics | None]
    frames: list[Frame]
    target_points: np.ndarray
    target_warp: TargetWarp | None = None

    def __repr__(self) -> str:
        model = self.camera_model
        sigma = self.residual_sigma_map()
        n_out = self.num_outliers()
        n_det = self.num_detections()
        residual_line_suffix = "\n"
        target_warp_line = ""
        if self.target_warp is not None:
            residual_line_suffix = ",\n"
            target_warp_line = f"  target_warp={self.target_warp!r}\n"
        return (
            f"CalibrationResult(\n"
            f"  model={model!r},\n"
            f"  frames={len(self.frames)}, "
            f"detections={n_det}, outliers={n_out},\n"
            f"  residual_sigma={sigma:.4f}px"
            f"{residual_line_suffix}" + target_warp_line + ")"
        )

    def residual_sigma_map(self) -> float:
        """Robust MAP estimate of residual standard deviation over inliers.

        Uses the MAD-based estimator (sigma = 1.4826 * MAD) on all inlier
        residual components (x and y combined).

        Returns:
            Estimated residual standard deviation in pixels.
        """
        inlier_vals = np.concatenate(
            [
                fi.residuals[fi.inlier_mask]
                for fi in self.frame_diagnostics
                if fi is not None
            ]
        ).ravel()
        mu = float(np.median(inlier_vals))
        mad = float(np.median(np.abs(inlier_vals - mu)))
        return 1.4826 * mad

    def num_outliers(self) -> int:
        """Count the total number of outlier detections across all frames."""
        return sum(
            int(np.count_nonzero(~fi.inlier_mask))
            for fi in self.frame_diagnostics
            if fi is not None
        )

    def num_detections(self) -> int:
        """Count the total number of detections (inliers + outliers) across all frames."""
        return sum(len(fi.residuals) for fi in self.frame_diagnostics if fi is not None)

    # -- Plot forwarding methods --
    # These require the `analysis` extra (pip install lensboy[analysis]).

    def plot_detection_coverage(
        self,
        *,
        title: str = "Coverage",
        s: float = 6.0,
        grid_cells: int = 50,
        return_figure: bool = False,
    ) -> Figure | None:
        """Scatter-plot all detected points with empty grid cells highlighted.

        Divides the image into a grid and shades cells with no detections,
        making coverage gaps easy to spot.

        Args:
            title: Plot title.
            s: Marker size passed to ``ax.scatter``.
            grid_cells: Number of grid cells along the longer axis.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_detection_coverage

        return plot_detection_coverage(
            self.frames,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            title=title,
            s=s,
            grid_cells=grid_cells,
            return_figure=return_figure,
        )

    def plot_inlier_coverage(
        self,
        *,
        title: str = "Inlier coverage",
        s: float = 6.0,
        grid_cells: int = 50,
        return_figure: bool = False,
    ) -> Figure | None:
        """Scatter-plot inlier detections with empty grid cells highlighted.

        Like ``plot_detection_coverage`` but only includes inlier points,
        making coverage gaps caused by outlier rejection visible.

        Args:
            title: Plot title.
            s: Marker size passed to ``ax.scatter``.
            grid_cells: Number of grid cells along the longer axis.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_inlier_coverage

        return _plot_inlier_coverage(
            self.frames,
            self.frame_diagnostics,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            title=title,
            s=s,
            grid_cells=grid_cells,
            return_figure=return_figure,
        )

    def plot_outliers(
        self,
        *,
        title: str = "Outliers",
        s: float = 12.0,
        return_figure: bool = False,
    ) -> Figure | None:
        """Scatter-plot all outlier detections in the image.

        Shows where rejected points are located, making it easy to spot
        systematic detection problems in specific image regions.

        Args:
            title: Plot title.
            s: Marker size passed to ``ax.scatter``.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_outliers

        return _plot_outliers(
            self.frames,
            self.frame_diagnostics,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            title=title,
            s=s,
            return_figure=return_figure,
        )

    def plot_distortion_grid(
        self,
        *,
        grid_cells: int = 50,
        fov_fraction: float | None = None,
        ux_max: float | None = None,
        uy_max: float | None = None,
        cmap_name: str = "jet",
        show_spline_knots: bool = True,
        return_figure: bool = False,
    ) -> Figure | None:
        """Project a regular grid through a camera model to visualize distortion.

        Builds a grid in normalized (tan-angle) space from the model's FOV, projects
        it, and clips to the image bounds.

        Args:
            grid_cells: Number of grid cells along the longer axis.
            fov_fraction: Fraction of the full FOV to sample (0, 1].
            ux_max: Upper bound in normalized x, mirrored to negative.
            uy_max: Upper bound in normalized y, mirrored to negative.
            cmap_name: Matplotlib colormap name.
            show_spline_knots: When True and the model is a PinholeSplined,
                overlay the spline control points on both panels.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_distortion_grid

        return plot_distortion_grid(
            self.camera_model,
            grid_cells=grid_cells,
            fov_fraction=fov_fraction,
            ux_max=ux_max,
            uy_max=uy_max,
            cmap_name=cmap_name,
            show_spline_knots=show_spline_knots,
            return_figure=return_figure,
        )

    def plot_residuals(
        self,
        *,
        bins: int = 100,
        n_sigma: float = 6.0,
        axis_range: float | None = None,
        title: str = "Reprojection residuals",
        return_figure: bool = False,
    ) -> Figure | None:
        """Per-component histogram and 2D scatter of reprojection residuals.

        Top-left: inlier histogram with a 1D Gaussian fit overlaid.
        Bottom-left: 2D scatter of inlier residuals with fitted 2D Gaussian
        contours.  Both left panels are trimmed to ±``n_sigma`` standard
        deviations.  Right column: full-range scatter highlighting outliers
        in red.

        Args:
            bins: Number of histogram bins.
            n_sigma: Number of fitted-Gaussian standard deviations for axis limits.
            axis_range: Fixed symmetric axis limit (±value) for the histogram and
                2D scatter plots. The full-range plot is unaffected. Auto-scaled
                from n_sigma if None.
            title: Overall figure title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_residuals

        return _plot_residuals(
            self.frame_diagnostics,
            bins=bins,
            n_sigma=n_sigma,
            axis_range=axis_range,
            title=title,
            return_figure=return_figure,
        )

    def plot_residual_vectors(
        self,
        *,
        title: str = "Residual vectors",
        scale: float = 10.0,
        scale_by_magnitude: bool = True,
        color_by: str = "magnitude",
        return_figure: bool = False,
    ) -> Figure | None:
        """Quiver plot of reprojection residual vectors over the image plane.

        Each arrow is placed at the detected point location with direction and
        length given by the residual.

        Args:
            title: Plot title.
            scale: Multiplier applied to arrow lengths for visibility.
            scale_by_magnitude: When False, all arrows are drawn with uniform
                length (direction only).
            color_by: ``"magnitude"`` colours by residual norm, ``"angle"``
                colours by residual direction using a cyclic colormap.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_residual_vectors

        return _plot_residual_vectors(
            self.frames,
            self.frame_diagnostics,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            title=title,
            scale=scale,
            scale_by_magnitude=scale_by_magnitude,
            color_by=color_by,
            return_figure=return_figure,
        )

    def plot_residual_grid(
        self,
        *,
        grid_cells: int = 50,
        arrow_scale: float = 100.0,
        heatmap_max: float | None = None,
        title: str = "Residual grid",
        return_figure: bool = False,
    ) -> Figure | None:
        """Binned residual summary showing per-cell magnitude and mean direction.

        The image plane is divided into a grid. Each cell is coloured by the mean
        inlier residual magnitude and has an arrow showing the mean residual
        vector, revealing spatial bias patterns.

        Args:
            grid_cells: Number of grid cells along the longer axis.
            arrow_scale: Multiplier applied to the mean-residual arrows.
            heatmap_max: Upper limit for the colour scale. Auto-scaled if None.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_residual_grid

        return _plot_residual_grid(
            self.frames,
            self.frame_diagnostics,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            grid_cells=grid_cells,
            arrow_scale=arrow_scale,
            heatmap_max=heatmap_max,
            title=title,
            return_figure=return_figure,
        )

    def plot_target_and_poses(
        self,
        *,
        triad_scale: float = 20.0,
        title: str = "Target and camera poses",
        return_figure: bool = False,
    ) -> Figure | None:
        """3D scatter of the calibration target with camera poses shown as triads.

        Each camera is drawn as a coordinate-frame triad (X=red, Y=green, Z=blue)
        at the camera position in the target reference frame.

        Args:
            triad_scale: Length of each triad axis arrow in target units.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_target_and_poses

        return _plot_target_and_poses(
            self.target_points,
            self.cameras_from_target,
            triad_scale=triad_scale,
            title=title,
            return_figure=return_figure,
        )

    def plot_target_warp(
        self,
        *,
        grid_res: int = 300,
        contour_levels: int = 15,
        title: str = "Target warp",
        return_figure: bool = False,
    ) -> Figure | None:
        """Contour plot of the target warp z-displacement viewed from above.

        Evaluates the warp function over a dense grid in the warp frame's xy plane
        and shows filled contours of the z height, with target point positions
        scattered on top.

        Args:
            grid_res: Number of grid samples along each axis.
            contour_levels: Number of contour lines.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.

        Raises:
            ValueError: If no target warp was estimated.
        """
        from lensboy.analysis.plots import _plot_target_warp

        if self.target_warp is None:
            raise ValueError("No target warp was estimated in this calibration.")
        return _plot_target_warp(
            self.target_points,
            self.target_warp,
            grid_res=grid_res,
            contour_levels=contour_levels,
            title=title,
            return_figure=return_figure,
        )

    def plot_worst_residual_frames(
        self,
        images: list[np.ndarray],
        *,
        n: int = 5,
        scale: float = 100.0,
        title: str = "Worst residual frames",
        include_outliers: bool = True,
        return_figure: bool = False,
    ) -> Figure | None:
        """Show the frames with the largest residuals, with residual vectors overlaid.

        Frames are ranked by their single worst (max-magnitude) residual and the
        top ``n`` are displayed in a single-column figure.  Each subplot shows the
        image with quiver arrows from detected points in the direction and
        magnitude of the residual, coloured by magnitude.

        Args:
            images: Source images corresponding to each frame, shape (H, W) or (H, W, 3).
            n: Number of worst frames to display.
            scale: Multiplier applied to arrow lengths for visibility.
            title: Overall figure title.
            include_outliers: Whether to include outlier points. When False, only
                inlier points (per ``FrameDiagnostics.inlier_mask``) are shown and used
                for ranking.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_worst_residual_frames

        return _plot_worst_residual_frames(
            self.frame_diagnostics,
            self.frames,
            images,
            n=n,
            scale=scale,
            title=title,
            include_outliers=include_outliers,
            return_figure=return_figure,
        )

    def plot_per_image_rms(
        self,
        *,
        sort_by: Literal["inliers", "all"] | None = "all",
        title: str = "Per-image residual RMS",
        return_figure: bool = False,
    ) -> Figure | None:
        """Stacked bar chart of per-image residual RMS split by inlier/outlier.

        Each bar shows the RMS of all residuals in that image. The left (blue)
        portion is the inlier-only RMS; the right (red) portion covers the
        remainder up to the total RMS, indicating the outlier contribution.

        Args:
            sort_by: Sort bars by ``"inliers"`` (inlier-only RMS) or
                ``"all"`` (total RMS including outliers). None keeps the
                original image order.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_per_image_rms

        return _plot_per_image_rms(
            self.frame_diagnostics,
            sort_by=sort_by,
            title=title,
            return_figure=return_figure,
        )

    def plot_frame_residuals(
        self,
        index: int,
        images: list[np.ndarray] | None = None,
        *,
        scale: float = 100.0,
        title: str | None = None,
        return_figure: bool = False,
    ) -> Figure | None:
        """Residual vectors for a single calibration frame.

        Shows quiver arrows from detected points coloured by residual magnitude,
        with outliers highlighted in red. Optionally overlaid on the source image.

        Args:
            index: Frame index to plot.
            images: Optional list of source images. If provided, ``images[index]``
                is used as the background.
            scale: Multiplier applied to arrow lengths for visibility.
            title: Plot title. Auto-generated if None.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import _plot_frame_residuals

        fi = self.frame_diagnostics[index]
        if fi is None:
            raise ValueError(f"Frame {index} has no diagnostics (PnP failed)")
        image = None
        if images is not None:
            image = images[index]
        return _plot_frame_residuals(
            self.frames[index],
            fi,
            image=image,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            scale=scale,
            title=title,
            return_figure=return_figure,
        )
