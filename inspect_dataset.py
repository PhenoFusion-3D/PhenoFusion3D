#!/usr/bin/env python3
"""Inspect RGB-D dataset: structure, intrinsics, shapes, depth stats, diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils_rgbd import load_intrinsics, pair_frames


def _depth_histogram_mm(valid_mm: np.ndarray, bucket_mm: float = 100.0) -> None:
    """Histogram in mm with fixed bucket width over the observed data range."""
    total = len(valid_mm)
    if total == 0:
        print("  (no samples)")
        return
    vmin_b = np.floor(valid_mm.min() / bucket_mm) * bucket_mm
    vmax_b = np.ceil(valid_mm.max() / bucket_mm) * bucket_mm
    if vmax_b <= vmin_b + 1e-9:
        vmax_b = vmin_b + bucket_mm
    edges = np.arange(vmin_b, vmax_b + bucket_mm * 1e-6 + bucket_mm, bucket_mm)
    counts, _ = np.histogram(valid_mm, bins=edges)
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        pct = 100.0 * count / total if total else 0.0
        print(f"  [{low:8.0f} .. {high:8.0f}) mm : {int(count):10d}  ({pct:6.2f}%)")


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Inspect RGB-D dataset structure and statistics.")
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root (contains rgb/, depth/, kdc_intrinsics.txt)",
    )
    args = ap.parse_args()
    data_dir: Path = args.data.expanduser().resolve()

    intrinsic_path = data_dir / "kdc_intrinsics.txt"
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Dataset paths ===")
    print(f"  data_dir       : {data_dir}")
    print(f"  intrinsics     : {intrinsic_path}")
    print(f"  output_dir     : {output_dir}")

    print("\n=== Frame pairs ===")
    pairs = pair_frames(data_dir)
    n = len(pairs)
    print(f"  paired frames  : {n}")
    if n >= 5:
        for i in range(5):
            r, d = pairs[i]
            print(f"    first[{i}]    : rgb={r.name}, depth={d.name}")
        print("    ...")
        for i in range(n - 5, n):
            r, d = pairs[i]
            print(f"    last[{i}]     : rgb={r.name}, depth={d.name}")
    elif n > 0:
        for i, (r, d) in enumerate(pairs):
            print(f"    [{i}]         : rgb={r.name}, depth={d.name}")
    else:
        print("  (no pairs)")

    print("\n=== Intrinsics (kdc_intrinsics.txt) ===")
    intr = load_intrinsics(intrinsic_path)
    K = intr["K"]
    print(f"  width x height : {intr['width']} x {intr['height']}")
    print("  K:\n", K)
    print(f"  dist           : {intr['dist']}")

    rgb_path, depth_path = pairs[0]
    color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    print("\n=== First frame image properties ===")
    print(f"  rgb_path       : {rgb_path.relative_to(data_dir)}")
    print(f"  depth_path     : {depth_path.relative_to(data_dir)}")
    if color_bgr is None or depth is None:
        raise SystemExit("Failed to read first RGB or depth image.")
    print(f"  RGB shape      : {color_bgr.shape}, dtype={color_bgr.dtype}")
    print(f"  depth shape    : {depth.shape}, dtype={depth.dtype}")

    z = depth.astype(np.float64)
    valid_mask = np.isfinite(z) & (z > 0)
    valid_mm = z[valid_mask]
    nz = depth.size - np.count_nonzero(valid_mask)

    print("\n=== Depth statistics (positive / valid pixels) ===")
    if valid_mm.size == 0:
        print("  No valid depth pixels!")
    else:
        print(f"  valid count    : {valid_mm.size}")
        print(f"  percent valid  : {100.0 * valid_mm.size / depth.size:.2f}%")
        print(f"  min (mm)       : {valid_mm.min():.2f}")
        print(f"  median (mm)    : {np.median(valid_mm):.2f}")
        print(f"  max (mm)       : {valid_mm.max():.2f}")
        print(f"  masked/zero    : {nz}")

        print("\n=== Depth histogram (100 mm buckets over observed mm range) ===")
        _depth_histogram_mm(valid_mm, bucket_mm=100.0)

    diag_rgb = output_dir / "inspect_rgb.png"
    cv2.imwrite(str(diag_rgb), color_bgr)
    print(f"\nSaved diagnostic RGB -> {diag_rgb.relative_to(data_dir)}")

    dvis = depth.copy().astype(np.float64)
    if valid_mm.size:
        vmin, vmax = float(valid_mm.min()), float(valid_mm.max())
        dn = np.zeros_like(dvis, dtype=np.float64)
        if vmax > vmin:
            dn[valid_mask] = (dvis[valid_mask] - vmin) / (vmax - vmin)
        else:
            dn[valid_mask] = 0.0
        dn_u8 = (np.clip(dn, 0.0, 1.0) * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(dn_u8, cv2.COLORMAP_INFERNO)
        heat[~valid_mask] = 0
        out_depth = heat
    else:
        out_depth = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    diag_d = output_dir / "inspect_depth.png"
    cv2.imwrite(str(diag_d), out_depth)
    print(f"Saved depth preview -> {diag_d.relative_to(data_dir)}")


if __name__ == "__main__":
    main()
