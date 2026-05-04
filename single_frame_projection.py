#!/usr/bin/env python3
"""Project one RGB-D frame to a colored point cloud (.ply)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from utils_rgbd import load_intrinsics, pair_frames, rgbd_to_pcd


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Single-frame RGB-D to Open3D point cloud.")
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root",
    )
    ap.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index into the sorted paired list (0-based).",
    )
    ap.add_argument(
        "--undistort",
        action="store_true",
        help="Apply cv2.undistort to RGB and depth (same image size).",
    )
    ap.add_argument(
        "--no-erode",
        action="store_true",
        help="Disable depth-edge erosion (flying-pixel suppression is on by default).",
    )
    ap.add_argument(
        "--depth-trunc",
        type=float,
        default=2.5,
        help="Max depth kept by Open3D in metres (dataset uses mm; defaults to 2.5m).",
    )
    args = ap.parse_args()
    data_dir: Path = args.data.expanduser().resolve()
    intrinsic_path = data_dir / "kdc_intrinsics.txt"
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_frames(data_dir)
    if args.index < 0 or args.index >= len(pairs):
        raise SystemExit(f"--index must be in [0, {len(pairs)-1}], got {args.index}")

    intr = load_intrinsics(intrinsic_path)
    K_orig = intr["K"]
    dist = intr["dist"]

    rgb_p, depth_p = pairs[args.index]
    color_bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
    depth_u16 = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)

    if color_bgr is None or depth_u16 is None:
        raise SystemExit(f"Failed to load {rgb_p} or {depth_p}")

    print(f"[single_frame] Frame index {args.index}: {rgb_p.name} / {depth_p.name}")

    K_use = np.array(K_orig, copy=True)

    if args.undistort:
        color_bgr = cv2.undistort(color_bgr, K_use, dist, None, K_use)
        depth_f32 = depth_u16.astype(np.float32)
        depth_remapped = cv2.undistort(depth_f32, K_use, dist, None, K_use)
        depth_u16 = np.round(depth_remapped).astype(depth_u16.dtype)

    intrinsic_wh = (intr["width"], intr["height"])

    pcd = rgbd_to_pcd(
        color_bgr,
        depth_u16,
        K_use,
        depth_scale=1000.0,
        depth_trunc=args.depth_trunc,
        erode_depth_edges=not args.no_erode,
        bbox=None,
        intrinsic_width_height=intrinsic_wh,
    )

    n_pts = len(pcd.points)
    valid_mm = depth_u16[depth_u16 > 0].astype(np.float64)

    print(f"[single_frame] point count after projection: {n_pts}")
    if valid_mm.size:
        print(f"[single_frame] depth valid pixels {valid_mm.size}, min/med/max mm "
              f"{valid_mm.min():.1f} / {np.median(valid_mm):.1f} / {valid_mm.max():.1f}")

    stem = output_dir / f"single_frame_{args.index}"
    ply_path = stem.with_suffix(".ply")
    rgb_out = stem.parent / f"single_frame_{args.index}_rgb.png"

    o3d.io.write_point_cloud(str(ply_path), pcd)
    cv2.imwrite(str(rgb_out), color_bgr)

    log_path = output_dir / f"single_frame_{args.index}_log.txt"
    lines = [
        f"rgb={rgb_p}",
        f"depth={depth_p}",
        f"points={n_pts}",
        f"rgb_shape={color_bgr.shape}",
        f"depth_shape={depth_u16.shape}",
        f"depth_dtype={depth_u16.dtype}",
        f"undistort={args.undistort}",
        f"depth_trunc_m={args.depth_trunc}",
    ]
    if valid_mm.size:
        lines.extend(
            [
                f"valid_pixels={valid_mm.size}",
                f"min_mm={valid_mm.min()}",
                f"median_mm={np.median(valid_mm)}",
                f"max_mm={valid_mm.max()}",
            ]
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved PLY  -> {ply_path}")
    print(f"Saved RGB  -> {rgb_out}")
    print(f"Saved log  -> {log_path}")


if __name__ == "__main__":
    main()
