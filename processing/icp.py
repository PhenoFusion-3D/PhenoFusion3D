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


def color_icp(source, target, max_iter=50, voxel_size=0.005, max_correspondence_distance=None):
    """
    Colour-assisted ICP registration between two point clouds.
    Falls back to point-to-plane ICP when coloured ICP is unavailable or unstable.
    """
    if source.is_empty() or target.is_empty():
        print('[icp] WARNING: Empty point cloud passed to color_icp, skipping.')
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    max_distance = max_correspondence_distance or max(voxel_size * 10.0, 0.02)
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
            np.eye(4),
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
            max_correspondence_distance=max_distance,
        )


def point_to_plane_icp(source, target, max_iter=50, voxel_size=0.005, max_correspondence_distance=None):
    """
    Point-to-plane ICP fallback. Estimates normals on working copies.
    """
    if source.is_empty() or target.is_empty():
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    max_distance = max_correspondence_distance or max(voxel_size * 10.0, 0.02)
    source_prepared = _prepare_for_registration(source, voxel_size)
    target_prepared = _prepare_for_registration(target, voxel_size)

    if source_prepared.is_empty() or target_prepared.is_empty():
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    result = o3d.pipelines.registration.registration_icp(
        source_prepared,
        target_prepared,
        max_correspondence_distance=max_distance,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
        ),
    )

    return result, result.transformation, result.fitness, result.inlier_rmse
