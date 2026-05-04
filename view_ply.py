#!/usr/bin/env python3
"""Open a .ply file in the Open3D viewer (CLI helper for pipeline outputs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser(description="View a .ply point cloud or mesh with Open3D.")
    ap.add_argument(
        "ply",
        type=Path,
        help="Path to .ply (e.g. sample_output/merge_pcd_best.ply)",
    )
    ap.add_argument(
        "--mesh",
        action="store_true",
        help="Load as triangle mesh (default: point cloud).",
    )
    args = ap.parse_args()
    path = args.ply.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    if args.mesh:
        geom = o3d.io.read_triangle_mesh(str(path))
        if geom.is_empty():
            raise SystemExit("Loaded mesh is empty.")
        geom.compute_vertex_normals()
    else:
        geom = o3d.io.read_point_cloud(str(path))
        if geom.is_empty():
            raise SystemExit("Loaded point cloud is empty. If the file has faces, use --mesh.")

    print(path)
    print(geom)
    o3d.visualization.draw_geometries([geom], window_name=path.name)


if __name__ == "__main__":
    main()
