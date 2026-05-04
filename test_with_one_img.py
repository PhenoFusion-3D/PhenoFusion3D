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

SEQ_ROOT   = "data/main/test_plant_20230809133659"  # best new dataset (1.3M pts)
INTRINSICS = os.path.join(SEQ_ROOT, "kdc_intrinsics.txt")
rgb_dir    = os.path.join(SEQ_ROOT, "rgb")
depth_dir  = os.path.join(SEQ_ROOT, "depth")
# Single-frame PLY is written here (same convention as test_with_whole_seq.py):
#   <SEQ_ROOT>/output/single_frame_<rgb_basename>_mode<N>.ply
OUT_DIR    = os.path.join(SEQ_ROOT, "output")
# Set to an integer (for example 280) to test a specific sorted frame.
# Leave as None to use the middle frame.
FRAME_IDX  = 80

rgb_files   = natsorted(glob.glob(os.path.join(rgb_dir,   "*.png")))
depth_files = natsorted(glob.glob(os.path.join(depth_dir, "*.png")))

n = len(rgb_files)
if n == 0:
    raise SystemExit(f"No PNG files in {rgb_dir!r}")
if len(depth_files) != n:
    raise SystemExit(f"RGB count {n} != depth count {len(depth_files)}")

if FRAME_IDX is None:
    i = n // 2
else:
    if FRAME_IDX < 0 or FRAME_IDX >= n:
        raise SystemExit(f"FRAME_IDX={FRAME_IDX} is outside valid range 0..{n - 1}")
    i = FRAME_IDX
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
    buckets = np.arange(0, 4200, 100)
    hist, _ = np.histogram(valid_px, bins=buckets)
    print("Depth histogram (mm buckets, non-zero only):")
    for lo, count in zip(buckets, hist):
        if count > 0:
            print(f"  {lo:4d}-{lo + 100:4d} mm : {count:6,} px")

# --------------------------------------------------------------------------
# PARAMETER MODES -- run ONE at a time to pinpoint what hurts point counts.
# Uncomment the block you want; keep the rest commented out.
# --------------------------------------------------------------------------

MODE = 5

if MODE == 1:
    # Absolute minimum -- mirrors stakeholder approach (no extras)
    pcd = rgbd2pcd(color, depth, K, depth_scale=1000.0, depth_trunc=4.0)
    title = "MODE 1: minimal (no dist, no bbox, trunc=4m)"

# MODE 2: Add distortion correction
elif MODE == 2:
    pcd = rgbd2pcd(color, depth, K, dist=dist, depth_scale=1000.0, depth_trunc=4.0)
    title = "MODE 2: +dist undistort"

# MODE 3: Add bbox crop
elif MODE == 3:
    pcd = rgbd2pcd(color, depth, K, dist=dist, bbox=[150, 100, 1130, 680],
                   depth_scale=1000.0, depth_trunc=4.0)
    title = "MODE 3: +dist +bbox"

# MODE 4: Add near-clip (depth_min_mm=300)
elif MODE == 4:
    pcd = rgbd2pcd(color, depth, K, dist=dist, bbox=[150, 100, 1130, 680],
                   depth_scale=1000.0, depth_trunc=3.2, depth_min_mm=300)
    title = "MODE 4: +dist +bbox +near-clip"

# MODE 5: Full production parameters (matches test_with_whole_seq.py for new datasets)
# For rs13_1: depth_trunc=3.1, depth_min_mm=2000
# For 20230809133659/133757: depth_trunc=3.0, depth_min_mm=1900
elif MODE == 5:
    pcd = rgbd2pcd(color, depth, K, dist=dist,
                   bbox=[300, 100, 980, 670],
                   depth_scale=1000.0, depth_trunc=3.0, depth_min_mm=1900,
                   mask_background=True, bg_sat_thresh=40)
    title = "MODE 5: production params + bg mask (new datasets)"

else:
    raise SystemExit(f"Unknown MODE={MODE}")

# --------------------------------------------------------------------------

print(f"\n[{title}]")
print(f"Points: {len(pcd.points):,}")
if pcd.is_empty():
    raise SystemExit("Point cloud is empty. Check depth data and parameters.")

os.makedirs(OUT_DIR, exist_ok=True)
rgb_stem = os.path.splitext(os.path.basename(rgb_files[i]))[0]
out_ply = os.path.join(OUT_DIR, f"single_frame_{rgb_stem}_mode{MODE}.ply")
o3d.io.write_point_cloud(out_ply, pcd)
print(f"PLY written to: {os.path.abspath(out_ply)}")

out_rgb = os.path.join(OUT_DIR, f"single_frame_{rgb_stem}_rgb.png")
cv2.imwrite(out_rgb, color_bgr)
print(f"RGB frame written to: {os.path.abspath(out_rgb)}")

o3d.visualization.draw_geometries([pcd], window_name=title)
