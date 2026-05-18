import sys
sys.path.insert(0, ".")

import cv2
import glob
import os
import numpy as np
import open3d as o3d
from natsort import natsorted

from file_io.loader import load_intrinsics
from processing.rgbd import rgbd2pcd

SEQ_ROOT   = "data/main/test_plant_rs13_1"
INTRINSICS = os.path.join(SEQ_ROOT, "kdc_intrinsics.txt")
rgb_dir    = os.path.join(SEQ_ROOT, "rgb")
depth_dir  = os.path.join(SEQ_ROOT, "depth")

rgb_files   = natsorted(glob.glob(os.path.join(rgb_dir,   "*.png")))
depth_files = natsorted(glob.glob(os.path.join(depth_dir, "*.png")))

n = len(rgb_files)
if n == 0:
    raise SystemExit(f"No PNG files in {rgb_dir!r}")
if len(depth_files) != n:
    raise SystemExit(f"RGB count {n} != depth count {len(depth_files)}")

i = n // 2
print(f"Using frame index {i}/{n - 1} ({os.path.basename(rgb_files[i])})")
color_bgr = cv2.imread(rgb_files[i])
depth     = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
if color_bgr is None or depth is None:
    raise SystemExit(f"Failed to read image pair at index {i}")
color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

K, dist, w, h = load_intrinsics(INTRINSICS)

# --------------------------------------------------------------------------
# Depth image diagnostics (always shown regardless of mode)
# --------------------------------------------------------------------------
valid_px = depth[depth > 0]
if len(valid_px) == 0:
    print("WARNING: No valid depth pixels in this frame at all!")
else:
    pct = 100.0 * len(valid_px) / depth.size
    print(
        f"Raw depth  --  valid: {len(valid_px):,}/{depth.size:,} ({pct:.1f}%)  "
        f"min={int(valid_px.min())}mm  median={int(np.median(valid_px))}mm  "
        f"max={int(valid_px.max())}mm"
    )

# --------------------------------------------------------------------------
# PARAMETER MODES -- run ONE at a time to pinpoint what hurts point counts.
# Uncomment the block you want; keep the rest commented out.
# --------------------------------------------------------------------------

# MODE 1: Absolute minimum -- mirrors stakeholder approach (no extras)
MODE = 1
pcd = rgbd2pcd(color, depth, K, depth_scale=1000.0, depth_trunc=4.0)
title = "MODE 1: minimal (no dist, no bbox, trunc=4m)"

# MODE 2: Add distortion correction
# MODE = 2
# pcd = rgbd2pcd(color, depth, K, dist=dist, depth_scale=1000.0, depth_trunc=4.0)
# title = "MODE 2: +dist undistort"

# MODE 3: Add bbox crop
# MODE = 3
# pcd = rgbd2pcd(color, depth, K, dist=dist, bbox=[150, 100, 1130, 680],
#                depth_scale=1000.0, depth_trunc=4.0)
# title = "MODE 3: +dist +bbox"

# MODE 4: Add near-clip (depth_min_mm=300)
# MODE = 4
# pcd = rgbd2pcd(color, depth, K, dist=dist, bbox=[150, 100, 1130, 680],
#                depth_scale=1000.0, depth_trunc=3.2, depth_min_mm=300)
# title = "MODE 4: +dist +bbox +near-clip"

# --------------------------------------------------------------------------

print(f"\n[{title}]")
print(f"Points: {len(pcd.points):,}")
if pcd.is_empty():
    raise SystemExit("Point cloud is empty. Check depth data and parameters.")

o3d.visualization.draw_geometries([pcd], window_name=title)
