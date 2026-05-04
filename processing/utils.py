import numpy as np
import open3d as o3d


def clean_pcd(pcd, nb_neighbors=30, std_ratio=1.5, voxel_size=0.005):
    """
    Downsample and remove statistical outliers from a point cloud.
    Used for the final merged output only -- NOT before ICP registration.
    Returns cleaned PointCloud. Handles empty input gracefully.
    """
    if pcd is None or pcd.is_empty():
        print('[utils] WARNING: clean_pcd received empty point cloud, skipping.')
        return pcd

    # Voxel downsample first - reduces density and speeds up display/export
    pcd = pcd.voxel_down_sample(voxel_size)

    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd


def clean_pcd_for_registration(pcd):
    """
    Outlier removal WITHOUT voxel downsampling -- for use before ICP.

    Matches the stakeholder pipeline order at metre scale: remove radius
    outliers first to strip flying pixels, then apply statistical cleanup.
    """
    if pcd is None or pcd.is_empty():
        return pcd

    # Stakeholder radius=3 is in millimetres; our point clouds are metres.
    pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.003)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
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