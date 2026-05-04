import os
import json
import numpy as np
from natsort import natsorted
import glob
import re


def _filter_numbered_pngs(paths, prefix):
    pattern = re.compile(rf'{re.escape(prefix)}_(\d+)\.png$')
    matched = []
    for path in paths:
        name = os.path.basename(path)
        found = pattern.match(name)
        if found:
            matched.append((int(found.group(1)), path))
    matched.sort(key=lambda item: item[0])
    return [path for _, path in matched]


def load_image_pairs(rgb_dir, depth_dir, step=1):
    """
    Load sorted RGB + depth image path pairs from two directories.
    Returns a list of (rgb_path, depth_path) tuples, sampled at 'step' interval.
    """
    rgb_files = _filter_numbered_pngs(glob.glob(os.path.join(rgb_dir, 'rgb_*.png')), 'rgb')
    depth_files = _filter_numbered_pngs(glob.glob(os.path.join(depth_dir, 'depth_*.png')), 'depth')

    # Fallback: if no rgb_*.png found, try any PNG
    if not rgb_files:
        rgb_files = natsorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    if not depth_files:
        depth_files = natsorted(glob.glob(os.path.join(depth_dir, '*.png')))

    if len(rgb_files) == 0:
        raise FileNotFoundError(f'No PNG files found in RGB directory: {rgb_dir}')
    if len(depth_files) == 0:
        raise FileNotFoundError(f'No PNG files found in depth directory: {depth_dir}')
    if len(rgb_files) != len(depth_files):
        raise ValueError(
            f'RGB and depth image counts do not match: '
            f'{len(rgb_files)} RGB vs {len(depth_files)} depth'
        )

    pairs = list(zip(rgb_files, depth_files))
    return pairs[::step]


def load_intrinsics(json_path):
    """
    Parse a kdc_intrinsics.txt JSON file in the stakeholder format.
    Returns: K (np.ndarray 3x3), dist (list), width (int), height (int)
    Returns None if file is missing or malformed.
    """
    if not json_path or not os.path.exists(json_path):
        print(f'[loader] WARNING: Intrinsics file not found: {json_path}. Using defaults.')
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        K = np.array(data['K'], dtype=np.float64)
        dist = data.get('dist', [0, 0, 0, 0, 0])
        width = int(data.get('width', 640))
        height = int(data.get('height', 480))
        print(f'[loader] Loaded intrinsics: {width}x{height}, fx={K[0,0]:.2f}, fy={K[1,1]:.2f}')
        return K, dist, width, height
    except Exception as e:
        print(f'[loader] WARNING: Failed to parse intrinsics file: {e}. Using defaults.')
        return None


def get_default_intrinsics(width=640, height=480, fov_deg=60.0):
    """
    Build a reasonable pinhole camera intrinsics matrix when no file is available.
    Returns: K (np.ndarray 3x3), dist (list of 5 zeros)
    """
    import math
    fx = width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    dist = [0.0, 0.0, 0.0, 0.0, 0.0]
    print(f'[loader] Using default intrinsics: {width}x{height}, fx=fy={fx:.2f}')
    return K, dist
