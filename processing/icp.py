import open3d as o3d
import numpy as np


def _ensure_normals(pcd, radius, max_nn=30):
    """Estimate normals if the point cloud doesn't have them yet."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )


def color_icp(source, target, max_iter=50, voxel_size=0.005, init=None):
    """
    Colour-assisted ICP registration between two point clouds.
    Pre-estimates normals if missing (required by Open3D coloured ICP).
    Falls back to point_to_plane_icp if colour ICP fails.

    Args:
        init: 4x4 initial transform from target to source (Open3D convention). Default identity.

    Returns: (result, transformation, fitness, inlier_rmse)
    """
    if source.is_empty() or target.is_empty():
        print('[icp] WARNING: Empty point cloud passed to color_icp, skipping.')
        return None, np.eye(4), 0.0, 0.0

    radius = voxel_size * 2
    init_tf = init if init is not None else np.eye(4)

    # Normals are required by coloured ICP - estimate them upfront
    _ensure_normals(source, radius)
    _ensure_normals(target, radius)

    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            source, target,
            radius,
            init_tf,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter
            )
        )
        fitness = result.fitness
        inlier_rmse = result.inlier_rmse

        if fitness == 0.0:
            print('[icp] colour_icp fitness=0, falling back to point-to-plane ICP')
            return point_to_plane_icp(source, target, max_iter, voxel_size, init_tf)

        return result, result.transformation, fitness, inlier_rmse

    except Exception as e:
        print(f'[icp] colour_icp failed ({e}), falling back to point-to-plane ICP')
        return point_to_plane_icp(source, target, max_iter, voxel_size, init_tf)


def point_to_plane_icp(source, target, max_iter=50, voxel_size=0.005, init=None):
    """
    Point-to-plane ICP fallback. Estimates normals if not present.

    Args:
        init: 4x4 initial transform. Default identity.

    Returns: (result, transformation, fitness, inlier_rmse)
    """
    if source.is_empty() or target.is_empty():
        return None, np.eye(4), 0.0, 0.0

    radius = voxel_size * 2
    init_tf = init if init is not None else np.eye(4)
    _ensure_normals(source, radius)
    _ensure_normals(target, radius)

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=radius,
        init=init_tf,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )
    )

    return result, result.transformation, result.fitness, result.inlier_rmse


def fpfh_ransac_initial_transform(source, target, *, voxel_size=0.01):
    """
    Estimate a global source→target rigid transform using FPFH features + RANSAC.

    Useful as a recovery initialisation when ICP has diverged and no kinematic
    prior is available.  More expensive than ICP — only call as a last resort
    inside a recovery loop.

    Args:
        source, target : Open3D PointCloud objects (should already be roughly
                         co-located within ~0.5 m for best results)
        voxel_size     : Downsampling voxel size used for feature extraction.
                         Use ~2x the ICP voxel_size (default 0.01 m = 1 cm).

    Returns: (result, transformation 4×4, fitness, inlier_rmse)
    """
    distance_threshold = voxel_size * 4.0

    def _preprocess(pcd):
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