import numpy as np
import cv2
import open3d as o3d


def rgbd2pcd(color_img, depth_img, K, dist=None, bbox=None, depth_scale=1000.0, depth_trunc=3.0):
    """
    Convert an RGB image + depth image into an Open3D coloured PointCloud.

    Args:
        color_img   : np.ndarray (H, W, 3) in RGB order
        depth_img   : np.ndarray (H, W) uint16, depth in mm (divide by depth_scale -> metres)
        K           : 3x3 intrinsic matrix (np.ndarray or nested list)
        dist        : distortion coefficients (list of 5), or None
        bbox        : optional [x1, y1, x2, y2] crop on the colour image before projection
        depth_scale : divisor to convert raw depth to metres (1000 for RealSense mm, 1 for ICL-NUIM)
        depth_trunc : discard depth beyond this many metres (default 3.0 m)

    Returns:
        o3d.geometry.PointCloud with colour
    """
    K = np.array(K, dtype=np.float64)
    h, w = color_img.shape[:2]

    # Undistort if distortion coefficients provided and non-zero
    if dist is not None and any(d != 0.0 for d in dist):
        dist_arr = np.array(dist, dtype=np.float64)
        color_img = cv2.undistort(color_img, K, dist_arr)
        depth_img = cv2.undistort(depth_img, K, dist_arr)

    # Optional bbox crop (applied before projection)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color_img = color_img[y1:y2, x1:x2]
        depth_img = depth_img[y1:y2, x1:x2]
        # Adjust principal point for the crop
        K_crop = K.copy()
        K_crop[0, 2] -= x1
        K_crop[1, 2] -= y1
        K = K_crop
        h, w = color_img.shape[:2]

    # Ensure colour is uint8 RGB
    if color_img.dtype != np.uint8:
        color_img = (color_img * 255).astype(np.uint8)

    # Ensure depth is uint16
    if depth_img.dtype != np.uint16:
        depth_img = depth_img.astype(np.uint16)

    # Mask invalid near-field and far-field RealSense readings
    # Near: <1500mm = machine body / sensor noise (dead zone 0.5-2.0m confirms)
    # Far:  >3200mm = floor and sensor overflow (RealSense D405 max ~5m, but noise starts at 3.5m+)
    depth_min_mm = int(2000)   # dead zone confirmed 0–2.0m (depth histogram)
    depth_max_mm = int(depth_trunc * depth_scale)  # e.g. 3.2m * 1000 = 3200
    invalid_mask = (depth_img < depth_min_mm) | (depth_img > depth_max_mm)
    depth_img = depth_img.copy()
    depth_img[invalid_mask] = 0

    # Fill holes caused by leaf IR scatter + the masking above
    depth_float = depth_img.astype(np.float32)
    zero_mask = (depth_img == 0).astype(np.uint8)
    if zero_mask.sum() > 0:
        depth_float = cv2.inpaint(
            depth_float, zero_mask,
            inpaintRadius=5, flags=cv2.INPAINT_NS
        )
        depth_img = depth_float.astype(np.uint16)

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_img)
    o3d_depth = o3d.geometry.Image(depth_img)

    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    # Build camera intrinsics from K matrix
    fx = float(K[0, 0])
    fy = float(abs(K[1, 1]))   # abs handles ICL-NUIM negative fy convention
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Project to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd