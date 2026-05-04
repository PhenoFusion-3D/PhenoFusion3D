#!/usr/bin/env python3
"""Render quick orthographic preview images for an Open3D point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def _rasterize(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    colors: np.ndarray,
    *,
    resolution: float,
    margin: float = 0.05,
) -> np.ndarray:
    amin, amax = float(a_vals.min() - margin), float(a_vals.max() + margin)
    bmin, bmax = float(b_vals.min() - margin), float(b_vals.max() + margin)
    width = max(1, int((amax - amin) / resolution) + 1)
    height = max(1, int((bmax - bmin) / resolution) + 1)

    ai = np.clip(((a_vals - amin) / resolution).astype(np.int32), 0, width - 1)
    bi = np.clip(((b_vals - bmin) / resolution).astype(np.int32), 0, height - 1)

    acc = np.zeros((height, width, 3), dtype=np.float64)
    cnt = np.zeros((height, width), dtype=np.int32)
    for channel in range(3):
        np.add.at(acc[:, :, channel], (bi, ai), colors[:, channel])
    np.add.at(cnt, (bi, ai), 1)

    mask = cnt > 0
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        vals = np.zeros((height, width), dtype=np.float64)
        vals[mask] = acc[:, :, channel][mask] / cnt[mask]
        img[:, :, channel] = np.clip(vals * 255.0, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.flip(img, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render front/side/top PCD previews.")
    ap.add_argument("pcd", type=Path)
    ap.add_argument("--out-prefix", type=Path, default=None)
    ap.add_argument("--resolution", type=float, default=0.002, help="metres per pixel")
    args = ap.parse_args()

    pcd_path = args.pcd.expanduser().resolve()
    out_prefix = (
        args.out_prefix.expanduser().resolve()
        if args.out_prefix
        else pcd_path.with_suffix("")
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise SystemExit(f"No points in {pcd_path}")
    colors = np.asarray(pcd.colors)
    if len(colors) != len(pts):
        colors = np.ones((len(pts), 3), dtype=np.float64)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    views = {
        "front": (x, y),
        "side": (z, y),
        "top": (x, z),
    }
    for name, (a_vals, b_vals) in views.items():
        img = _rasterize(a_vals, b_vals, colors, resolution=args.resolution)
        out_path = out_prefix.parent / f"{out_prefix.name}_{name}.png"
        cv2.imwrite(str(out_path), img)
        print(f"[render] {name}: {img.shape[1]}x{img.shape[0]} -> {out_path}")


if __name__ == "__main__":
    main()
