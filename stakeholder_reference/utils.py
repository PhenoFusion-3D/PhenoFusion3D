import numpy as np
import open3d as o3d


def clean_pcd(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=0.005):
    if pcd is None or pcd.is_empty():
        print('[utils] WARNING: clean_pcd received empty point cloud, skipping.')
        return pcd
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd


def estimate_normals(pcd, radius=0.01, max_nn=30):
    if pcd is None or pcd.is_empty():
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    return pcd


def rgbd2pcd(color, depth, K, depth_scale=1000.0, depth_trunc=1.5, bbox=None):
    """
    将 RGB 图和深度图转换为 Open3D 彩色点云。

    参数：
        color      : np.ndarray, shape (H, W, 3), RGB格式, uint8
        depth      : np.ndarray, shape (H, W), uint16, 单位毫米
        K          : np.ndarray, shape (3, 3), 相机内参矩阵
        depth_scale: 深度图数值 → 米的换算系数，D405默认1000（1mm=0.001m）
        depth_trunc: 最大有效深度（米），超过的点丢弃，俯拍植物建议1.5m
        bbox       : [x1, y1, x2, y2] 像素级裁剪框，None表示不裁剪
    """
    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 像素裁剪
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = color[y1:y2, x1:x2]
        depth = depth[y1:y2, x1:x2]
        cx -= x1
        cy -= y1
        h, w = depth.shape

    # 构建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 深度有效性掩码
    z = depth.astype(np.float64) / depth_scale
    valid = (z > 0) & (z < depth_trunc)

    z = z[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    # 颜色归一化到 [0, 1]
    colors = color[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_icp(source, target,
              voxel_size=0.005,
              max_iter=50,
              max_correspondence_distance=None):
    """
    基于颜色的 ICP 点云配准（Color ICP）。
    失败时自动降级到 Point-to-Plane ICP。

    返回：
        (success, transformation, fitness, inlier_rmse)
    """
    if max_correspondence_distance is None:
        max_correspondence_distance = voxel_size * 10

    # Color ICP 需要法线
    def prep(pcd):
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
        return pcd

    src = prep(source)
    tgt = prep(target)

    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            src, tgt,
            max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=max_iter
            )
        )
        fitness = result.fitness
        inlier_rmse = result.inlier_rmse
        transformation = result.transformation
        success = True

    except Exception as e:
        print(f'[utils] Color ICP failed ({e}), falling back to Point-to-Plane ICP')
        result = o3d.pipelines.registration.registration_icp(
            src, tgt,
            max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        fitness = result.fitness
        inlier_rmse = result.inlier_rmse
        transformation = result.transformation
        success = False

    return success, transformation, fitness, inlier_rmse


def green_filter_pcd(pcd, h_range=(35, 85), s_min=0.2, v_min=0.1):
    """
    保留点云中绿色的点（用于去掉背景，只保留植物）。

    参数：
        h_range: HSV 色相范围（OpenCV中绿色约 35-85）
        s_min  : 最小饱和度（过滤灰色/白色背景）
        v_min  : 最小亮度（过滤黑色区域）
    """
    import cv2
    colors = np.asarray(pcd.colors)  # (N, 3), float [0,1], RGB
    points = np.asarray(pcd.points)

    # 转成 uint8 BGR for OpenCV HSV 转换
    bgr = (colors[:, ::-1] * 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    h, s, v = hsv[:, 0], hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    mask = (h >= h_range[0]) & (h <= h_range[1]) & (s >= s_min) & (v >= v_min)

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points[mask])
    result.colors = o3d.utility.Vector3dVector(colors[mask])
    return result


def check_gpu():
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
    if check_gpu():
        import cupy as cp
        return cp
    return np