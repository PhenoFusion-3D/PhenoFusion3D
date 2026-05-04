#!/usr/bin/env python3
"""
Export ICP camera poses + intrinsics to a Nerfstudio transforms.json suitable
for running depth-nerfacto or splatfacto with depth supervision.

Coordinate-frame note:
    Our ICP poses are camera→world in the FLIPPED frame:
        +x right, +y UP, +z BACKWARD
    This is exactly OpenGL convention, which Nerfstudio's transform_matrix expects.
    No axis swap is required — poses are used as-is.

Depth note:
    Depth PNGs are uint16 in millimetres.
    Nerfstudio reads depth with depth_unit_scale_factor = 0.001 (mm → m).

Output:
    <data>/output/nerfstudio/transforms.json
    Image/depth paths in the JSON are relative to that directory.

Running depth-nerfacto after export:
    pip install nerfstudio
    cd <PhenoFusion3D root>
    ns-train depth-nerfacto --data <data>/output/nerfstudio/
    # After training (~30-60 min on a GPU):
    ns-export tsdf \\
        --load-config outputs/<run>/depth-nerfacto/config.yml \\
        --output-dir <data>/output/nerf_mesh/

Alternatively for 3D Gaussian Splatting with depth supervision (DN-Splatter):
    pip install dn-splatter
    dn-splat train --data <data>/output/nerfstudio/ \\
                   --output-dir outputs/dn_splatter/

Prerequisite:
    Run reconstruct_icp_sequence.py first to generate output/poses/frame_N.txt files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from utils_rgbd import load_intrinsics, pair_frames


def _plant_mask_bgr(color_bgr: np.ndarray) -> np.ndarray:
    """HSV plant mask used to suppress gantry/background pixels for NeRF training."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(
        hsv,
        np.array([30, 35, 25], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    stem_mask = cv2.inRange(
        hsv,
        np.array([8, 45, 25], dtype=np.uint8),
        np.array([32, 255, 210], dtype=np.uint8),
    )
    mask = cv2.bitwise_or(leaf_mask, stem_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Export ICP poses + intrinsics → Nerfstudio transforms.json."
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root (same as used for ICP).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for transforms.json.  "
             "Defaults to <data>/output/nerfstudio/.",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame subsampling step — must match the step used when ICP was run.",
    )
    ap.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not generate plant masks or add mask_path entries to transforms.json.",
    )
    args = ap.parse_args()

    data_dir = args.data.expanduser().resolve()
    output_dir = data_dir / "output"
    pose_dir = output_dir / "poses"
    intrinsic_path = data_dir / "kdc_intrinsics.txt"
    out_dir = args.out.expanduser().resolve() if args.out else output_dir / "nerfstudio"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    if not args.no_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    if not pose_dir.exists():
        raise SystemExit(
            f"Pose directory not found: {pose_dir}\n"
            "Run reconstruct_icp_sequence.py first to generate poses."
        )

    intr = load_intrinsics(intrinsic_path)
    K = intr["K"]
    dist = intr["dist"]
    W, H = intr["width"], intr["height"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Nerfstudio OpenCV distortion coefficients (k1,k2,p1,p2,k3)
    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    p1 = float(dist[2]) if len(dist) > 2 else 0.0
    p2 = float(dist[3]) if len(dist) > 3 else 0.0
    k3 = float(dist[4]) if len(dist) > 4 else 0.0

    # Diagonal field-of-view helpers (kept for compatibility with older ns versions)
    camera_angle_x = 2.0 * math.atan(W / (2.0 * fx))
    camera_angle_y = 2.0 * math.atan(H / (2.0 * fy))

    pairs_all = pair_frames(data_dir)

    # Collect (pair_idx, frame_idx, pose_path) sorted by frame_idx
    frame_entries: list[tuple[int, int, Path]] = []
    for pose_path in sorted(pose_dir.glob("frame_*.txt")):
        frame_idx = int(pose_path.stem.split("_")[1])
        pair_idx = frame_idx  # step=1 → frame_idx == pair_idx
        if pair_idx >= len(pairs_all):
            continue
        frame_entries.append((pair_idx, frame_idx, pose_path))
    frame_entries.sort(key=lambda x: x[1])

    if not frame_entries:
        raise SystemExit("No pose files found in " + str(pose_dir))

    # Image paths in transforms.json are relative to the out_dir.
    try:
        rgb_rel = Path("..") / ".." / "rgb"
        depth_rel = Path("..") / ".." / "depth"
        _ = (out_dir / rgb_rel).resolve()
    except Exception:
        rgb_rel = data_dir / "rgb"
        depth_rel = data_dir / "depth"

    frames: list[dict] = []
    for pair_idx, frame_idx, pose_path in frame_entries:
        rgb_path, depth_path = pairs_all[pair_idx]

        pose = np.loadtxt(str(pose_path))  # 4×4 camera→world in OpenGL convention

        rgb_rel_path = (rgb_rel / rgb_path.name).as_posix()
        depth_rel_path = (depth_rel / depth_path.name).as_posix()

        frame: dict = {
            "file_path": rgb_rel_path,
            "depth_file_path": depth_rel_path,
            "transform_matrix": pose.tolist(),
        }
        if not args.no_masks:
            color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if color_bgr is None:
                raise RuntimeError(f"Cannot read RGB image for mask generation: {rgb_path}")
            mask = _plant_mask_bgr(color_bgr)
            mask_path = mask_dir / rgb_path.name
            cv2.imwrite(str(mask_path), mask)
            frame["mask_path"] = (Path("masks") / mask_path.name).as_posix()
        frames.append(frame)

    transforms = {
        # Camera model with distortion — depth-nerfacto supports OPENCV
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": W,
        "h": H,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        # field-of-view helpers for older Nerfstudio versions
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        # Depth images are uint16 PNG in millimetres → scale to metres
        "depth_unit_scale_factor": 0.001,
        "frames": frames,
    }

    out_path = out_dir / "transforms.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    print(f"[nerf-export] Wrote {len(frames)} frames -> {out_path}")
    if not args.no_masks:
        print(f"[nerf-export] Wrote plant masks -> {mask_dir}")
    print()
    print("To train depth-nerfacto:")
    print("  pip install nerfstudio")
    print(f"  ns-train depth-nerfacto --data \"{out_dir}\"")
    print()
    print("To train 3D Gaussian Splatting with depth (DN-Splatter):")
    print("  pip install dn-splatter")
    print(f"  dn-splat train --data \"{out_dir}\" --output-dir outputs/dn_splatter/")
    print()
    print("After training, export mesh:")
    print("  ns-export tsdf --load-config outputs/<run>/depth-nerfacto/config.yml")
    print(f"              --output-dir \"{output_dir / 'nerf_mesh'}\"")


if __name__ == "__main__":
    main()
