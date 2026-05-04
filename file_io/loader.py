import os
import json
import math
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


GANTRY_CONFIG_FILENAME = 'gantry_config.json'
SESSION_FILENAME = 'session.json'


def load_image_pairs(rgb_dir, depth_dir, step=1):
    """
    Load sorted RGB + depth image path pairs from two directories.
    Handles both naming conventions:
      - Stakeholder format: rgb_XXXXXX.png / depth_XXXXXX.png
      - ICL-NUIM format:    0.png, 1.png, 2.png ...
    Returns a list of (rgb_path, depth_path) tuples sampled at 'step' interval.
    """
    # Prefer strict rgb_N / depth_N pairing when filenames are numeric tokens.
    rgb_files = _filter_numbered_pngs(glob.glob(os.path.join(rgb_dir, 'rgb_*.png')), 'rgb')
    depth_files = _filter_numbered_pngs(glob.glob(os.path.join(depth_dir, 'depth_*.png')), 'depth')
    if not rgb_files:
        rgb_files = natsorted(glob.glob(os.path.join(rgb_dir, 'rgb_*.png')))
    if not depth_files:
        depth_files = natsorted(glob.glob(os.path.join(depth_dir, 'depth_*.png')))

    # Fall back to plain numbered PNGs (ICL-NUIM convention)
    if not rgb_files:
        rgb_files = natsorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    if not depth_files:
        depth_files = natsorted(glob.glob(os.path.join(depth_dir, '*.png')))

    if not rgb_files:
        raise FileNotFoundError(f'No PNG files found in RGB directory: {rgb_dir}')
    if not depth_files:
        raise FileNotFoundError(f'No PNG files found in depth directory: {depth_dir}')
    if len(rgb_files) != len(depth_files):
        raise ValueError(
            f'RGB and depth image counts do not match: '
            f'{len(rgb_files)} RGB vs {len(depth_files)} depth'
        )

    pairs = list(zip(rgb_files, depth_files))
    sampled = pairs[::step]
    print(f'[loader] Found {len(pairs)} pairs, using {len(sampled)} at step={step}')
    return sampled


def load_intrinsics(json_path):
    """
    Parse a kdc_intrinsics.txt JSON file in the stakeholder format.
    Returns: (K np.ndarray 3x3, dist list, width int, height int)
    Returns None if file is missing or malformed.
    """
    if not json_path or not os.path.exists(json_path):
        print(f'[loader] WARNING: Intrinsics file not found: {json_path}. Using defaults.')
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        K      = np.array(data['K'], dtype=np.float64)
        dist   = data.get('dist', [0, 0, 0, 0, 0])
        width  = int(data.get('width',  640))
        height = int(data.get('height', 480))
        print(f'[loader] Loaded intrinsics: {width}x{height}, fx={K[0,0]:.2f}, fy={K[1,1]:.2f}')
        return K, dist, width, height
    except Exception as e:
        print(f'[loader] WARNING: Failed to parse intrinsics: {e}. Using defaults.')
        return None


def get_default_intrinsics(width=640, height=480, fov_deg=60.0):
    """
    Build a pinhole intrinsics matrix when no file is available.
    Returns: (K np.ndarray 3x3, dist list of 5 zeros)
    """
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


def _dataset_dir_from_path(path):
    """Return the dataset root from either a root, rgb, or depth directory."""
    if not path:
        return None
    norm = os.path.abspath(path)
    base = os.path.basename(norm).lower()
    if base in ('rgb', 'depth'):
        return os.path.dirname(norm)
    return norm


def load_session_json(dataset_dir):
    """
    Load capture session metadata saved next to a dataset.

    Args:
        dataset_dir: dataset root, rgb folder, or depth folder.

    Returns:
        Parsed session dictionary, or None when absent/malformed.
    """
    root = _dataset_dir_from_path(dataset_dir)
    if not root:
        return None
    path = os.path.join(root, SESSION_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f'[loader] Loaded session metadata: {path}')
        return data
    except Exception as e:
        print(f'[loader] WARNING: Failed to parse session metadata: {e}')
        return None


def save_gantry_config(dataset_dir, step_m, axis):
    """
    Persist gantry calibration next to a dataset.

    Args:
        dataset_dir: dataset root, rgb folder, or depth folder.
        step_m: gantry travel per original captured frame, in metres.
        axis: camera-space gantry axis, 0=X or 1=Y.
    """
    root = _dataset_dir_from_path(dataset_dir)
    if not root:
        raise ValueError('dataset_dir is required')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, GANTRY_CONFIG_FILENAME)
    data = {
        'gantry_step_m_per_frame': float(step_m),
        'gantry_axis': int(axis),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'[loader] Saved gantry config: {path}')
    return path


def load_gantry_config(dataset_dir):
    """
    Load persisted gantry calibration.

    Returns (step_m_per_frame, axis) or None when absent/malformed.
    """
    root = _dataset_dir_from_path(dataset_dir)
    if not root:
        return None
    path = os.path.join(root, GANTRY_CONFIG_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        step_m = float(data['gantry_step_m_per_frame'])
        axis = int(data['gantry_axis'])
        if step_m <= 0 or axis not in (0, 1):
            raise ValueError('gantry step must be >0 and axis must be 0 or 1')
        print(f'[loader] Loaded gantry config: step={step_m:.6f}m, axis={axis}')
        return step_m, axis
    except Exception as e:
        print(f'[loader] WARNING: Failed to parse gantry config: {e}')
        return None
