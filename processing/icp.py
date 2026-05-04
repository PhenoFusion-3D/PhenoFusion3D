import copy

import numpy as np
import open3d as o3d


def _prepare_for_registration(pcd, voxel_size):
    prepared = copy.deepcopy(pcd)
    if voxel_size > 0:
        prepared = prepared.voxel_down_sample(voxel_size)
    if prepared.is_empty():
        return prepared
    prepared.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(voxel_size * 2.0, 0.01),
            max_nn=30,
        )
    )
    return prepared


def color_icp(
    source,
    target,
    max_iter=50,
    voxel_size=0.005,
    init=None,
    max_correspondence_distance=None,
):
    """
    Colour-assisted ICP registration between two point clouds.
    Works on voxel-downsampled copies with estimated normals.
    Falls back to point-to-plane ICP when coloured ICP is unavailable or unstable.

    Args:
        init: 4x4 initial transform (Open3D convention). Default identity.
        max_correspondence_distance: ICP correspondence radius in metres;
            default scales with voxel_size.

    Returns: (result, transformation, fitness, inlier_rmse)
    """
    if source.is_empty() or target.is_empty():
        print('[icp] WARNING: Empty point cloud passed to color_icp, skipping.')
        return None, np.eye(4), 0.0, 0.0

    max_distance = max_correspondence_distance or max(voxel_size * 10.0, 0.02)
    init_tf = init if init is not None else np.eye(4)
    source_prepared = _prepare_for_registration(source, voxel_size)
    target_prepared = _prepare_for_registration(target, voxel_size)

    if source_prepared.is_empty() or target_prepared.is_empty():
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            source_prepared,
            target_prepared,
            max_distance,
            init_tf,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=max_iter,
            ),
        )
        if result.fitness == 0.0:
            print('[icp] colour_icp fitness=0, falling back to point-to-plane ICP')
            return point_to_plane_icp(
                source,
                target,
                max_iter=max_iter,
                voxel_size=voxel_size,
                init=init_tf,
                max_correspondence_distance=max_distance,
            )
        return result, result.transformation, result.fitness, result.inlier_rmse
    except Exception as exc:
        print(f'[icp] colour_icp failed ({exc}), falling back to point-to-plane ICP')
        return point_to_plane_icp(
            source,
            target,
            max_iter=max_iter,
            voxel_size=voxel_size,
            init=init_tf,
            max_correspondence_distance=max_distance,
        )


def point_to_plane_icp(
    source,
    target,
    max_iter=50,
    voxel_size=0.005,
    init=None,
    max_correspondence_distance=None,
):
    """
    Point-to-plane ICP fallback. Estimates normals on working copies.

    Args:
        init: 4x4 initial transform. Default identity.

    Returns: (result, transformation, fitness, inlier_rmse)
    """
    if source.is_empty() or target.is_empty():
        return None, np.eye(4), 0.0, 0.0

    max_distance = max_correspondence_distance or max(voxel_size * 10.0, 0.02)
    init_tf = init if init is not None else np.eye(4)
    source_prepared = _prepare_for_registration(source, voxel_size)
    target_prepared = _prepare_for_registration(target, voxel_size)

    if source_prepared.is_empty() or target_prepared.is_empty():
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    result = o3d.pipelines.registration.registration_icp(
        source_prepared,
        target_prepared,
        max_correspondence_distance=max_distance,
        init=init_tf,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
        ),
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
