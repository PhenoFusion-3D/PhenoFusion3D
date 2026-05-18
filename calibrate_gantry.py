"""
calibrate_gantry.py
-------------------
Determines gantry motion axis and per-frame displacement from the image data.

Uses OpenCV phase correlation between two frames (FRAME_GAP frames apart) to
find how many pixels the scene shifts between frames. Combined with the median
scene depth and camera focal length, this converts to a metric (metre) step.

Usage:
    python calibrate_gantry.py

Outputs:
    - Which camera axis the gantry moves along (X or Y)
    - Per-frame pixel shift
    - Estimated per-frame metric step in metres
    - Copy-paste values for test_with_whole_seq.py and controller.py
"""

import sys
sys.path.insert(0, ".")

import cv2
import glob
import numpy as np
import os
from natsort import natsorted

from file_io.loader import load_intrinsics

# ---------------------------------------------------------------------------
# Configuration -- adjust paths if your data is elsewhere
# ---------------------------------------------------------------------------
SEQ_ROOT   = "data/main/test_plant_rs13_1"
rgb_dir    = os.path.join(SEQ_ROOT, "rgb")
depth_dir  = os.path.join(SEQ_ROOT, "depth")
INTRINSICS = os.path.join(SEQ_ROOT, "kdc_intrinsics.txt")

# Compare frame[0] vs frame[FRAME_GAP].
# Larger gap = stronger signal, but risk of too little overlap.
# 50 frames at ~1.3mm/frame = ~65mm of travel -- safe for a 1280px-wide scene.
FRAME_GAP = 50


def _phase_shift(img_a_grey: np.ndarray, img_b_grey: np.ndarray):
    """Return (shift_x, shift_y, response) via OpenCV phase correlation."""
    a = img_a_grey.astype(np.float32)
    b = img_b_grey.astype(np.float32)
    (sx, sy), response = cv2.phaseCorrelate(a, b)
    return float(sx), float(sy), float(response)


def main():
    rgb_files   = natsorted(glob.glob(os.path.join(rgb_dir,   "*.png")))
    depth_files = natsorted(glob.glob(os.path.join(depth_dir, "*.png")))

    n = len(rgb_files)
    if n < FRAME_GAP + 1:
        raise SystemExit(
            f"Need at least {FRAME_GAP + 1} frames, found {n}. "
            f"Lower FRAME_GAP in this script."
        )

    intr = load_intrinsics(INTRINSICS)
    if intr is None:
        raise SystemExit(f"Cannot load intrinsics from {INTRINSICS!r}")
    K, dist, img_w, img_h = intr
    fx, fy = float(K[0, 0]), float(K[1, 1])

    print(f"Intrinsics: fx={fx:.1f}  fy={fy:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  ({img_w}x{img_h})")
    print(f"Comparing frame 0 vs frame {FRAME_GAP} ...")

    # Load both frames as greyscale for phase correlation
    img_a = cv2.imread(rgb_files[0],         cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(rgb_files[FRAME_GAP], cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        raise SystemExit("Failed to read comparison frames.")

    shift_x, shift_y, response = _phase_shift(img_a, img_b)

    print(f"\nPhase correlation result:")
    print(f"  shift_x = {shift_x:+.3f} px  (positive = scene moved right in image)")
    print(f"  shift_y = {shift_y:+.3f} px  (positive = scene moved down in image)")
    print(f"  response = {response:.4f}  (higher = more reliable)")
    print(f"  per-frame: dx={shift_x/FRAME_GAP:+.4f} px/frame  "
          f"dy={shift_y/FRAME_GAP:+.4f} px/frame")

    # Dominant axis
    if abs(shift_x) >= abs(shift_y):
        axis      = 0
        axis_name = "X (horizontal in image)"
        shift_ppf = shift_x / FRAME_GAP   # pixels per frame, signed
        focal     = fx
    else:
        axis      = 1
        axis_name = "Y (vertical in image)"
        shift_ppf = shift_y / FRAME_GAP
        focal     = fy

    print(f"\nDominant motion axis: camera {axis_name} (axis index {axis})")

    # Median scene depth for metric conversion
    depth_frame = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid_d = depth_frame[(depth_frame > 100) & (depth_frame < 5000)]
    if len(valid_d) == 0:
        print("\nWARNING: No valid depth in frame 0. Cannot estimate metric step.")
        print("Set gantry_step_m manually based on gantry velocity / fps.")
        return

    median_depth_mm = float(np.median(valid_d))
    median_depth_m  = median_depth_mm / 1000.0
    print(f"\nMedian scene depth (frame 0): {median_depth_m:.3f} m  ({int(median_depth_mm)} mm)")

    # step = shift_px * depth / focal_length  (pinhole back-projection)
    step_m_signed = shift_ppf * median_depth_m / focal
    step_m_abs    = abs(step_m_signed)

    print(f"\n--- Estimated per-frame gantry step ---")
    print(f"  {step_m_abs * 1000:.3f} mm/frame  =  {step_m_abs:.6f} m/frame")
    direction = "+" if step_m_signed >= 0 else "-"
    print(f"  direction: camera moves in {direction}camera-{axis_name.split()[0]} direction")

    # Sanity check against known gantry speed
    known_speed_mps = 0.038   # m/s from rospy_thread_fin_1.py
    known_fps       = 30.0
    known_step_m    = known_speed_mps / known_fps
    print(f"\nSanity check vs. known gantry speed ({known_speed_mps}m/s @ {known_fps}fps):")
    print(f"  Expected step = {known_step_m*1000:.3f} mm/frame")
    print(f"  Measured step = {step_m_abs*1000:.3f} mm/frame")
    ratio = step_m_abs / known_step_m if known_step_m > 0 else float('nan')
    print(f"  Ratio = {ratio:.2f}  (1.0 = perfect match; >1.5 or <0.5 = suspect)")

    print("\n" + "=" * 60)
    print("Copy-paste these into test_with_whole_seq.py and controller.py:")
    print("=" * 60)
    print(f"gantry_axis   = {axis}        # 0=X (horizontal), 1=Y (vertical)")
    print(f"gantry_step_m = {step_m_abs:.6f}  # metres per ORIGINAL frame")
    print(f"# For step=N sampling: pass gantry_step_m * N to Reconstructor")
    print("=" * 60)


if __name__ == "__main__":
    main()
