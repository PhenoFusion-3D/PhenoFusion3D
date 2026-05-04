#!/usr/bin/env python3
"""Run a small TSDF parameter sweep and save comparable plant-only previews.

Prerequisite: run reconstruct_icp_sequence.py first to generate poses.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import open3d as o3d


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[sweep] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Run plant-only TSDF parameter sweep.")
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root.",
    )
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-trunc", type=float, default=2.3)
    ap.add_argument("--preview-resolution", type=float, default=0.002)
    args = ap.parse_args()

    data_dir = args.data.expanduser().resolve()
    output_dir = data_dir / "output"
    sweep_dir = output_dir / "tsdf_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    voxels = [0.0015, 0.002, 0.003]
    trunc_multipliers = [3.0, 5.0, 8.0]
    rows: list[dict[str, object]] = []

    for voxel in voxels:
        for mult in trunc_multipliers:
            sdf_trunc = voxel * mult
            tag = f"v{voxel*1000:.1f}mm_t{mult:.0f}x".replace(".", "p")
            print(f"\n[sweep] === {tag}: voxel={voxel:.4f}, sdf_trunc={sdf_trunc:.4f} ===", flush=True)

            _run(
                [
                    sys.executable,
                    "reconstruct_tsdf.py",
                    "--color-mask",
                    "--plant-only-crop",
                    "--no-mesh",
                    "--voxel",
                    f"{voxel:.6f}",
                    "--sdf-trunc",
                    f"{sdf_trunc:.6f}",
                    "--depth-min",
                    f"{args.depth_min:.3f}",
                    "--depth-trunc",
                    f"{args.depth_trunc:.3f}",
                ],
                cwd=root,
            )

            src_pcd = output_dir / "tsdf_plant_only.ply"
            dst_pcd = sweep_dir / f"{tag}_plant_only.ply"
            shutil.copy2(src_pcd, dst_pcd)

            _run(
                [
                    sys.executable,
                    "render_pcd_views.py",
                    str(dst_pcd),
                    "--out-prefix",
                    str(sweep_dir / tag),
                    "--resolution",
                    f"{args.preview_resolution:.6f}",
                ],
                cwd=root,
            )

            pcd = o3d.io.read_point_cloud(str(dst_pcd))
            rows.append(
                {
                    "tag": tag,
                    "voxel_m": voxel,
                    "sdf_trunc_m": sdf_trunc,
                    "trunc_multiplier": mult,
                    "points": len(pcd.points),
                    "pcd": str(dst_pcd),
                    "front": str(sweep_dir / f"{tag}_front.png"),
                    "side": str(sweep_dir / f"{tag}_side.png"),
                    "top": str(sweep_dir / f"{tag}_top.png"),
                }
            )

    summary_path = sweep_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[sweep] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
