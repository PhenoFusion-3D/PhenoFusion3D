#!/usr/bin/env python3
"""
Generate a 2D top-down mosaic of the plant canopy.

Each RGB-D frame is projected into a world XY plane using the saved pose
translations from reconstruct_icp_sequence.py (pure-translation mode).

Usage:
    python create_topdown_mosaic.py [--data DATA_DIR] [--resolution 0.001]

Output:
    <data>/output/topdown_mosaic.png

Prerequisite:
    Run reconstruct_icp_sequence.py first to generate output/poses/frame_N.txt files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_intrinsics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float64)
    dist = np.array(d.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
    return {"K": K, "dist": dist, "width": int(d["width"]), "height": int(d["height"])}


def pair_frames(data_dir: Path) -> list[tuple[Path, Path]]:
    rgb_dir = data_dir / "rgb"
    depth_dir = data_dir / "depth"
    rgb_files = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))
    pairs = []
    for r in rgb_files:
        d = depth_dir / r.name
        if d.exists():
            pairs.append((r, d))
    return pairs


def erode_depth(depth_u16: np.ndarray, threshold_mm: float = 200.0) -> np.ndarray:
    """Zero out pixels near depth discontinuities (flying pixels)."""
    depth_f32 = depth_u16.astype(np.float32)
    sx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
    disc = ((np.abs(sx) + np.abs(sy)) > threshold_mm).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    disc = cv2.dilate(disc, kernel, iterations=1)
    out = depth_u16.copy()
    out[disc > 0] = 0
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Create a 2-D top-down mosaic of the plant canopy from RGB-D frames and saved poses."
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root (must contain rgb/, depth/, kdc_intrinsics.txt, output/poses/).",
    )
    ap.add_argument(
        "--resolution",
        type=float,
        default=0.001,
        help="Output image resolution in metres per pixel (default 0.001 = 1 mm/px).",
    )
    ap.add_argument(
        "--depth-trunc",
        type=float,
        default=2.5,
        help="Depth truncation in metres (points beyond this are ignored).",
    )
    ap.add_argument(
        "--depth-min",
        type=float,
        default=0.5,
        help="Minimum depth in metres (points closer than this are ignored, e.g. gantry frame).",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame subsampling step — must match the step used when running ICP so poses align.",
    )
    ap.add_argument(
        "--no-erode",
        action="store_true",
        help="Disable depth-edge erosion.",
    )
    ap.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help=(
            "Minimum number of frames that must contribute to a canvas pixel for it to be kept. "
            "The gantry moves with the camera so any world-XY position of the gantry is seen "
            "by only 1-2 frames; the plant is fixed and is seen by 20-60 frames. "
            "Pixels below this threshold are zeroed out, effectively suppressing the gantry stripe. "
            "Set to 1 to disable (keep all pixels)."
        ),
    )
    ap.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort frames before projecting.",
    )
    args = ap.parse_args()

    data_dir: Path = args.data.expanduser().resolve()
    output_dir = data_dir / "output"
    pose_dir = output_dir / "poses"
    out_path = output_dir / "topdown_mosaic.png"

    intr = load_intrinsics(data_dir / "kdc_intrinsics.txt")
    K: np.ndarray = intr["K"]
    dist: np.ndarray = intr["dist"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pairs_all = pair_frames(data_dir)
    num_pairs = len(pairs_all)
    ds_imgs = num_pairs // args.step
    if ds_imgs < 1:
        raise SystemExit(f"Not enough frames for step={args.step}.")

    # --- Collect all world XY coordinates (first pass) to compute canvas bounds ---
    print("[mosaic] Pass 1 — computing world bounds …")
    all_tx, all_ty = [], []
    used_frames: list[tuple[Path, Path, np.ndarray]] = []

    for i in tqdm(range(ds_imgs), desc="bounds"):
        idx = i * args.step
        pose_file = pose_dir / f"frame_{idx}.txt"
        if not pose_file.exists():
            continue
        pose = np.loadtxt(str(pose_file))
        tx, ty = float(pose[0, 3]), float(pose[1, 3])
        all_tx.append(tx)
        all_ty.append(ty)
        rgb_p, depth_p = pairs_all[idx]
        used_frames.append((rgb_p, depth_p, pose))

    if not used_frames:
        raise SystemExit("No pose files found. Run reconstruct_icp_sequence.py first.")

    # Camera FOV at median depth (~1.96 m): half-width ≈ cx/fx * depth
    median_depth = 1.96
    half_w = (cx / fx) * median_depth
    half_h = (cy / fy) * median_depth

    world_xmin = min(all_tx) - half_w - 0.05
    world_xmax = max(all_tx) + half_w + 0.05
    world_ymin = min(all_ty) - half_h - 0.05
    world_ymax = max(all_ty) + half_h + 0.05

    res = args.resolution
    canvas_w = int(np.ceil((world_xmax - world_xmin) / res)) + 1
    canvas_h = int(np.ceil((world_ymax - world_ymin) / res)) + 1

    print(f"[mosaic] World X: [{world_xmin:.3f}, {world_xmax:.3f}]  span={world_xmax - world_xmin:.3f} m")
    print(f"[mosaic] World Y: [{world_ymin:.3f}, {world_ymax:.3f}]  span={world_ymax - world_ymin:.3f} m")
    print(f"[mosaic] Canvas: {canvas_w} x {canvas_h} px  ({res*1000:.1f} mm/px)")

    # Accumulator arrays for colour-averaging blending
    canvas_acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    canvas_cnt = np.zeros((canvas_h, canvas_w), dtype=np.int32)

    # --- Pass 2 — paint pixels ---
    print("[mosaic] Pass 2 — painting frames …")
    for rgb_p, depth_p, pose in tqdm(used_frames, desc="paint"):
        color_bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_u16 is None:
            continue

        K_use = K.copy()
        if args.undistort:
            color_bgr = cv2.undistort(color_bgr, K_use, dist, None, K_use)
            depth_f32 = depth_u16.astype(np.float32)
            depth_remapped = cv2.undistort(depth_f32, K_use, dist, None, K_use)
            depth_u16 = np.round(depth_remapped).astype(depth_u16.dtype)

        if not args.no_erode:
            depth_u16 = erode_depth(depth_u16)

        depth_m = depth_u16.astype(np.float32) / 1000.0  # mm → m

        # Build pixel grid
        h, w = depth_m.shape
        us = np.arange(w, dtype=np.float32)
        vs = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        # Depth mask
        valid = (depth_m > args.depth_min) & (depth_m < args.depth_trunc)

        d_valid = depth_m[valid]
        u_valid = uu[valid]
        v_valid = vv[valid]

        # Unproject to world coordinates using pose translation
        tx = float(pose[0, 3])
        ty = float(pose[1, 3])

        X_world = (u_valid - cx) / fx * d_valid + tx
        # Flip Y: image v increases downward, world Y should increase along gantry direction
        Y_world = -((v_valid - cy) / fy * d_valid) + ty

        # Map world XY to canvas pixel
        px = np.round((X_world - world_xmin) / res).astype(np.int32)
        py = np.round((Y_world - world_ymin) / res).astype(np.int32)

        # Clip to canvas
        in_canvas = (px >= 0) & (px < canvas_w) & (py >= 0) & (py < canvas_h)
        px = px[in_canvas]
        py = py[in_canvas]

        # Get colours (BGR)
        rows = np.where(valid)[0][in_canvas]
        cols = np.where(valid)[1][in_canvas]
        colors = color_bgr[rows, cols].astype(np.float64)  # shape (N, 3)

        np.add.at(canvas_acc[:, :, 0], (py, px), colors[:, 0])
        np.add.at(canvas_acc[:, :, 1], (py, px), colors[:, 1])
        np.add.at(canvas_acc[:, :, 2], (py, px), colors[:, 2])
        np.add.at(canvas_cnt, (py, px), 1)

    # Convert accumulated colours to uint8 average
    filled_mask = canvas_cnt > 0
    canvas_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for c in range(3):
        canvas_rgb[:, :, c][filled_mask] = np.round(
            canvas_acc[:, :, c][filled_mask] / canvas_cnt[filled_mask]
        ).astype(np.uint8)

    # Frame-count filter: zero out pixels seen by fewer than --min-frames frames.
    if args.min_frames > 1:
        low_count_mask = filled_mask & (canvas_cnt < args.min_frames)
        canvas_rgb[low_count_mask] = 0
        filled_mask = filled_mask & (canvas_cnt >= args.min_frames)
        print(
            f"[mosaic] min_frames={args.min_frames}: removed {low_count_mask.sum()} low-count pixels"
        )

    # Flip vertically so that increasing gantry Y is towards top of image
    canvas_rgb = cv2.flip(canvas_rgb, 0)

    cv2.imwrite(str(out_path), canvas_rgb)
    print(f"[mosaic] Saved -> {out_path}")
    filled = int(filled_mask.sum())
    total = canvas_w * canvas_h
    print(f"[mosaic] Coverage: {filled}/{total} px  ({100.0 * filled / total:.1f}%)")


if __name__ == "__main__":
    main()
