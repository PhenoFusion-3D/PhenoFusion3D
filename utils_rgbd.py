"""
Reusable helpers for RGB-D reconstruction (OpenCV + NumPy + Open3D).
Matches stakeholder notebook clean_pcd parameters where noted.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import cv2
from natsort import natsorted


def load_intrinsics(path: Path | str) -> dict[str, Any]:
    """Load JSON-format intrinsics (K 3x3, dist coeffs, height, width)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    K = np.asarray(data["K"], dtype=np.float64)
    dist = np.asarray(data.get("dist", [0] * 5), dtype=np.float64)
    return {
        "K": K,
        "dist": dist,
        "width": int(data["width"]),
        "height": int(data["height"]),
        "raw": data,
    }


def pair_frames(data_dir: Path | str) -> list[tuple[Path, Path]]:
    """
    Return sorted (rgb_path, depth_path) pairs.

    Prefer layout: rgb/<name>.png and depth/<same name>.png
    Fallback: flat rgb_*.png and depth_*.png in data_dir (natsorted, paired by index).
    """
    data_dir = Path(data_dir)
    rgb_sub = data_dir / "rgb"
    depth_sub = data_dir / "depth"

    if rgb_sub.is_dir() and depth_sub.is_dir():
        rgb_files = [
            rgb_sub / p.name for p in natsorted(rgb_sub.glob("*.png"), key=lambda x: x.name)
        ]
        pairs: list[tuple[Path, Path]] = []
        for rf in rgb_files:
            dd = depth_sub / rf.name
            if dd.is_file():
                pairs.append((rf, dd))
            else:
                raise FileNotFoundError(f"No depth image for RGB {rf.name} at {dd}")
        if not pairs:
            raise FileNotFoundError(f"No PNG pairs found under {rgb_sub} / {depth_sub}")
        return pairs

    rgbs = natsorted(data_dir.glob("rgb_*.png"), key=lambda p: p.as_posix())
    depths = natsorted(data_dir.glob("depth_*.png"), key=lambda p: p.as_posix())
    if not rgbs or not depths:
        raise FileNotFoundError(
            f"No rgb/depth subdirs or flat rgb_*.png / depth_*.png in {data_dir}"
        )
    if len(rgbs) != len(depths):
        raise ValueError(f"Mismatch: {len(rgbs)} rgb vs {len(depths)} depth in {data_dir}")
    return list(zip(rgbs, depths))


def _pinhole_intrinsic(width: int, height: int, K: np.ndarray) -> o3d.camera.PinholeCameraIntrinsic:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def plant_mask_bgr(color_bgr: np.ndarray) -> np.ndarray:
    """
    HSV plant mask tuned for the gantry plant dataset.

    Keeps dark/bright green leaves and brown stems while rejecting gray metal,
    black tray/table regions, and low-saturation background.

    Returns a uint8 mask (255 = plant pixel, 0 = background).
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = cv2.inRange(
        hsv,
        np.array([30, 35, 25], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    stem_mask = cv2.inRange(
        hsv,
        np.array([8, 45, 25], dtype=np.uint8),
        np.array([32, 255, 210], dtype=np.uint8),
    )

    mask = cv2.bitwise_or(leaf_mask, stem_mask)
    close_kernel = np.ones((5, 5), np.uint8)
    open_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask


def rgbd_to_pcd(
    color_bgr: np.ndarray,
    depth_uint16: np.ndarray,
    K: np.ndarray,
    *,
    depth_scale: float = 1000.0,
    depth_trunc: float = 2.5,
    erode_depth_edges: bool = True,
    bbox: list[int] | tuple[int, int, int, int] | None = None,
    intrinsic_width_height: tuple[int, int] | None = None,
) -> o3d.geometry.PointCloud:
    """
    Project aligned RGB-D to a colored Open3D point cloud (camera frame).

    color_bgr: HxWx3 uint8 (OpenCV default)
    depth_uint16: HxW in millimetres (divide by depth_scale after Open3D conventions:
        Open3D treats depth pixels as metres when divided by depth_scale; use 1000 for mm.)

    erode_depth_edges: zero out pixels within a 3x3 band around large depth discontinuities
        (Sobel gradient magnitude > 200 mm/px) to suppress flying-pixel artifacts from
        the IR structured-light sensor.
    bbox: optional [x_min, y_min, x_max, y_max] inclusive crop applied to both modalities.
    intrinsic_width_height: if set (W,H), warn when image differs (calibration sanity check).
    """
    K_now = np.array(K, copy=True).astype(np.float64)
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        color_bgr = color_bgr[ymin : ymax + 1, xmin : xmax + 1].copy()
        depth_uint16 = depth_uint16[ymin : ymax + 1, xmin : xmax + 1].copy()
        K_now[0, 2] -= xmin
        K_now[1, 2] -= ymin

    color_h, color_w = color_bgr.shape[:2]
    depth_h, depth_w = depth_uint16.shape[:2]
    if (depth_h, depth_w) != (color_h, color_w):
        raise ValueError(
            f"Depth shape {(depth_h, depth_w)} != color shape {(color_h, color_w)}"
        )
    height, width = color_h, color_w

    if intrinsic_width_height is not None:
        iw, ih = intrinsic_width_height
        if (iw, ih) != (width, height):
            import warnings

            warnings.warn(
                f"Image size {(width)}x{(height)} != intrinsics nominal {iw}x{ih}",
                stacklevel=2,
            )

    if erode_depth_edges:
        depth_f32 = depth_uint16.astype(np.float32)
        sobel_x = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.abs(sobel_x) + np.abs(sobel_y)
        # >200 mm/pixel jump = depth discontinuity (flying-pixel source)
        disc_mask = (edge_mag > 200).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        disc_dilated = cv2.dilate(disc_mask, kernel, iterations=1)
        depth_uint16 = depth_uint16.copy()
        depth_uint16[disc_dilated > 0] = 0

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_uint16.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    intrinsic = _pinhole_intrinsic(width, height, K_now)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # Open3D creates the cloud in a camera frame where +y is down and +z is forward.
    # Flip y and z to get a right-hand frame with +y up (standard for visualisation).
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def clean_pcd(
    pcd: o3d.geometry.PointCloud,
    *,
    nb_neighbors: int = 10,
    std_ratio: float = 2.0,
    nb_points_radius: int = 20,
    radius: float = 0.05,
) -> o3d.geometry.PointCloud:
    """Stakeholder 3D_reconstruction.ipynb clean_pcd (statistical then radius)."""
    out = copy.deepcopy(pcd)
    out, _ = out.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    out, _ = out.remove_radius_outlier(nb_points=nb_points_radius, radius=radius)
    return out


def colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.01,
    max_correspondence_distance: float | None = None,
    max_iteration: int = 100,
    init: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, float, float]:
    """
    Pairwise colored ICP. Returns (result, transformation 4x4, fitness, inlier_rmse).
    Mirrors stakeholder convention: (_, T, fitness, inlier_rmse).
    """
    if max_correspondence_distance is None:
        max_correspondence_distance = voxel_size * 4.0
    if init is None:
        init = np.eye(4, dtype=np.float64)

    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    if not src_down.has_colors() or not tgt_down.has_colors():
        raise ValueError("Source and target point clouds must have colors for colored ICP")

    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )

    try:
        est = o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.968)
    except TypeError:
        est = o3d.pipelines.registration.TransformationEstimationForColoredICP()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

    result = o3d.pipelines.registration.registration_colored_icp(
        src_down,
        tgt_down,
        max_correspondence_distance,
        init,
        est,
        criteria,
    )
    return result, np.asarray(result.transformation).copy(), result.fitness, result.inlier_rmse


def point_to_plane_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.01,
    max_correspondence_distance: float | None = None,
    max_iteration: int = 100,
    init: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, float, float]:
    """Fallback geometric ICP for cases where colour is misleading or sparse."""
    if max_correspondence_distance is None:
        max_correspondence_distance = voxel_size * 4.0
    if init is None:
        init = np.eye(4, dtype=np.float64)

    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    result = o3d.pipelines.registration.registration_icp(
        src_down,
        tgt_down,
        max_correspondence_distance,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )
    return result, np.asarray(result.transformation).copy(), result.fitness, result.inlier_rmse


def fpfh_ransac_initial_transform(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.01,
) -> tuple[Any, np.ndarray, float, float]:
    """Estimate a global source->target transform using FPFH + RANSAC."""
    distance_threshold = voxel_size * 4.0

    def _preprocess(pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, Any]:
        down = pcd.voxel_down_sample(voxel_size)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        feature = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        )
        return down, feature

    src_down, src_feature = _preprocess(source)
    tgt_down, tgt_feature = _preprocess(target)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_feature,
        tgt_feature,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result, np.asarray(result.transformation).copy(), result.fitness, result.inlier_rmse
