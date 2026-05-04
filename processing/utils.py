import numpy as np
import open3d as o3d
import cv2


def clean_pcd(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=0.005):
    """
    Downsample and remove statistical outliers from a point cloud.
    Returns cleaned PointCloud. Handles empty input gracefully.
    """
    if pcd is None or pcd.is_empty():
        print('[utils] WARNING: clean_pcd received empty point cloud, skipping.')
        return pcd

    # Voxel downsample first - reduces density and speeds up ICP
    pcd = pcd.voxel_down_sample(voxel_size)

    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd


def estimate_normals(pcd, radius=0.01, max_nn=30):
    """
    Estimate and orient normals on a point cloud.
    Required for point-to-plane ICP fallback.
    """
    if pcd is None or pcd.is_empty():
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    return pcd


def green_filter_pcd(pcd, h_range=(35, 90), s_min=0.2, v_min=0.08):
    """
    Keep only green-ish points from a coloured point cloud.
    Useful for isolating foliage from trays, boards, rails, and floor.
    """
    if pcd is None or pcd.is_empty() or not pcd.has_colors():
        return pcd

    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    bgr = (colors[:, ::-1] * 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv[:, 0]
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    mask = (h >= h_range[0]) & (h <= h_range[1]) & (s >= s_min) & (v >= v_min)

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[mask])
    filtered.colors = o3d.utility.Vector3dVector(colors[mask])
    return filtered


def check_gpu():
    """
    Returns True if CUDA + CuPy are available.
    Used to switch between numpy and cupy in the pipeline.
    """
    try:
        import torch
        if torch.cuda.is_available():
            import cupy
            print('[utils] GPU detected: using CuPy')
            return True
    except ImportError:
        pass
    print('[utils] No GPU/CuPy available: using NumPy')
    return False


def numpy_or_cupy():
    """
    Returns cupy if GPU available, numpy otherwise.
    Drop-in replacement for array operations.
    """
    if check_gpu():
        import cupy as cp
        return cp
    return np
