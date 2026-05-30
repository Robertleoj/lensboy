"""Generate synthetic charuco images from a solved calibration for demos."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lensboy._logging import progress
from lensboy.calibration.calibrate import TargetWarp, WarpCoordinates
from lensboy.camera_models.opencv import OpenCV
from lensboy.geometry.pose import Pose

_DATA_PATH = Path(__file__).parent / "_demo_calibration.json"

_PIXELS_PER_UNIT = 4.0
_GRID_SUBDIVISIONS = 2


def _fill_triangle(
    map_x: np.ndarray,
    map_y: np.ndarray,
    dst_tri: np.ndarray,
    src_tri: np.ndarray,
) -> None:
    """Fill remap arrays for pixels inside one destination triangle."""
    out_h, out_w = map_x.shape

    min_xy = np.floor(dst_tri.min(axis=0)).astype(int)
    max_xy = np.ceil(dst_tri.max(axis=0)).astype(int)

    x0 = max(min_xy[0], 0)
    y0 = max(min_xy[1], 0)
    x1 = min(max_xy[0], out_w - 1)
    y1 = min(max_xy[1], out_h - 1)

    if x1 < x0 or y1 < y0:
        return

    local_w = x1 - x0 + 1
    local_h = y1 - y0 + 1
    local_tri = dst_tri - np.array([x0, y0], dtype=np.float32)

    mask = np.zeros((local_h, local_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local_tri).astype(np.int32), (1,))

    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return

    pts_dst = np.stack([xx + x0, yy + y0], axis=1).astype(np.float32)

    A = cv2.getAffineTransform(dst_tri.astype(np.float32), src_tri.astype(np.float32))
    pts_src = cv2.transform(pts_dst[None, :, :], A)[0]

    map_x[yy + y0, xx + x0] = pts_src[:, 0]
    map_y[yy + y0, xx + x0] = pts_src[:, 1]


def _warp_grid_piecewise_linear(
    image: np.ndarray,
    src_grid: np.ndarray,
    dst_grid: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Piecewise-linear warp from image using corresponding grid points.

    Splits each quad cell into two triangles and builds an affine remap per
    triangle, then samples the source image via cv2.remap.

    Args:
        image: Source image, shape (H, W) or (H, W, C).
        src_grid: Grid points in source image, shape (gy, gx, 2).
        dst_grid: Corresponding grid points in destination image, shape (gy, gx, 2).
        out_shape: (out_h, out_w) of the output image.

    Returns:
        Warped image, shape (out_h, out_w) or (out_h, out_w, C).
    """
    out_h, out_w = out_shape
    gy, gx, _ = src_grid.shape

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    for j in range(gy - 1):
        for i in range(gx - 1):
            s00 = src_grid[j, i]
            s10 = src_grid[j, i + 1]
            s01 = src_grid[j + 1, i]
            s11 = src_grid[j + 1, i + 1]

            d00 = dst_grid[j, i]
            d10 = dst_grid[j, i + 1]
            d01 = dst_grid[j + 1, i]
            d11 = dst_grid[j + 1, i + 1]

            for dst_tri, src_tri in [
                (np.array([d00, d10, d01]), np.array([s00, s10, s01])),
                (np.array([d11, d01, d10]), np.array([s11, s01, s10])),
            ]:
                _fill_triangle(map_x, map_y, dst_tri, src_tri)

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )


def _deserialize_warp(data: dict) -> TargetWarp:
    """Reconstruct a TargetWarp from a serialized dict."""
    wc = WarpCoordinates(
        target_from_warp_frame=Pose.from_rotvec_trans(
            rotvec=np.array(data["warp_frame_rotvec"]),
            trans=np.array(data["warp_frame_trans"]),
        ),
        x_scale=data["x_scale"],
        y_scale=data["y_scale"],
    )
    return TargetWarp(
        warp_coordinates=wc,
        object_warp=tuple(data["coefficients"]),
    )


def make_synthetic_images() -> list[np.ndarray]:
    """Generate synthetic charuco images from the bundled calibration data.

    Uses a precomputed camera model, poses, and target warp to render
    realistic charuco board images with lens distortion.

    Returns:
        List of grayscale uint8 images, each shape (image_height, image_width).
    """
    data = json.loads(_DATA_PATH.read_text())
    img_w = data["image_width"]
    img_h = data["image_height"]
    model = OpenCV.from_json(data["camera_model"])

    # Generate the board texture
    board_def = data["board"]
    board_size = tuple(board_def["size"])
    sq = board_def["square_length"]
    board_width = board_size[0] * sq
    board_height = board_size[1] * sq
    tex_w = int(round(board_width * _PIXELS_PER_UNIT))
    tex_h = int(round(board_height * _PIXELS_PER_UNIT))
    board = cv2.aruco.CharucoBoard(
        board_size,
        sq,
        board_def["marker_length"],
        cv2.aruco.getPredefinedDictionary(board_def["dictionary_id"]),
    )
    texture = board.generateImage((tex_w, tex_h))

    # Build warped 3D grid with margin beyond board edges
    margin = sq
    n_cols = board_size[0] * _GRID_SUBDIVISIONS + 1 + 2 * _GRID_SUBDIVISIONS
    n_rows = board_size[1] * _GRID_SUBDIVISIONS + 1 + 2 * _GRID_SUBDIVISIONS
    xs = np.linspace(-margin, board_width + margin, n_cols)
    ys = np.linspace(-margin, board_height + margin, n_rows)
    gx, gy = np.meshgrid(xs, ys)
    grid_flat = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n_rows * n_cols)])

    if "target_warp" in data:
        warp = _deserialize_warp(data["target_warp"])
        grid_flat = warp.warp_target(grid_flat)

    # Source grid: texture pixel coordinates (with margin mapped outside)
    margin_px = margin * _PIXELS_PER_UNIT
    src_xs = np.linspace(-margin_px, tex_w - 1 + margin_px, n_cols)
    src_ys = np.linspace(-margin_px, tex_h - 1 + margin_px, n_rows)
    src_gx, src_gy = np.meshgrid(src_xs, src_ys)
    src_grid = np.stack([src_gx, src_gy], axis=-1).astype(np.float32)

    poses = [
        Pose.from_rotvec_trans(
            rotvec=np.array(p["rotvec"]),
            trans=np.array(p["trans"]),
        )
        for p in data["poses"]
    ]

    images = []
    for pose in progress(poses, desc="Generating images"):
        points_in_cam = pose.apply(grid_flat)
        pixels = model.project_points(points_in_cam)
        dst_grid = pixels.reshape(n_rows, n_cols, 2).astype(np.float32)

        img = _warp_grid_piecewise_linear(texture, src_grid, dst_grid, (img_h, img_w))
        images.append(img)

    return images
