"""
Command-line canopy reconstruction tool.

Usage examples
--------------
# Single dataset, auto-masking (recommended for new plant datasets):
python reconstruct_canopy.py --input data/main/test_plant_20230809133659

# Specify output folder and use every 20th frame for a fast preview:
python reconstruct_canopy.py \\
    --input data/main/test_plant_20230809133659 \\
    --output data/main/test_plant_20230809133659/canopy_preview \\
    --stride 20 --max-frames 9

# Batch: run on all sub-folders under data/main:
python reconstruct_canopy.py --input data/main --batch

# Use a pre-computed external mask directory:
python reconstruct_canopy.py \\
    --input data/main/test_plant_rs13_1 \\
    --mask-dir data/main/test_plant_rs13_1/masks

Notes
-----
* Works with both flat-layout datasets (``rgb_N.png`` / ``depth_N.png``)
  and ICL-style datasets (``rgb/N.png`` / ``depth/N.png``).
* Requires ``kdc_intrinsics.txt`` in the dataset root.
* The output directory receives ``canopy_points.ply``, ``canopy_mesh.ply``
  (measurement mesh), ``canopy_display_mesh.ply`` (viewer mesh),
  ``canopy_viewer.html`` (interactive WebGL), ``fused_rgb_masked.png``, and
  ``canopy_summary.json``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _has_data(path: Path) -> bool:
    """Return True if *path* looks like a dataset root (has rgb files)."""
    return (
        any(path.glob("rgb_*.png"))
        or (path / "rgb").is_dir()
    )


def _discover_datasets(input_path: Path, batch: bool) -> list[Path]:
    if batch:
        roots = sorted(p for p in input_path.iterdir() if p.is_dir() and _has_data(p))
        if not roots:
            print(f"[canopy] No dataset sub-folders found under {input_path}")
            sys.exit(1)
        return roots
    return [input_path]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Top-down canopy 3-D reconstruction for plant gantry datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Notes")[0],
    )
    p.add_argument("--input",      required=True,
                   help="Dataset root folder (or parent folder when --batch is used).")
    p.add_argument("--output",     default=None,
                   help="Output directory (default: <input>/canopy_local).")
    p.add_argument("--batch",      action="store_true",
                   help="Process every sub-folder of --input as a separate dataset.")
    p.add_argument("--mask-dir",   default=None,
                   help="Pre-computed mask directory.  Auto-masking used if omitted.")
    p.add_argument("--stride",     type=int, default=1,
                   help="Evaluate every Nth frame during candidate search (default 1 = all frames).")
    p.add_argument("--max-frames", type=int, default=15,
                   help="Maximum frames to fuse (default 15).")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="Optional post-detection candidate shortlist (default 0 = no cap).")
    p.add_argument("--min-mask-area", type=int, default=180_000,
                   help="Minimum plant-mask area in pixels (default 180000).")
    p.add_argument("--coverage",   type=int, default=1,
                   help="Minimum frame-overlap to keep a canvas pixel (default 1).")
    p.add_argument("--depth-min",  type=int, default=500,
                   help="Near-clip depth in mm (default 500).")
    p.add_argument("--depth-max",  type=int, default=4000,
                   help="Far-clip depth in mm (default 4000).")
    p.add_argument("--z-scale",    type=float, default=1.0,
                   help="Vertical exaggeration for the 3-D output (default 1.0).")
    p.add_argument("--max-hole-fill-px", type=int, default=24,
                   help="Maximum inpaint distance from real depth in pixels (default 24).")
    p.add_argument("--max-triangle-jump", type=float, default=0.025,
                   help="Maximum height jump for neighbouring mesh triangles in metres.")
    p.add_argument("--no-auto-mask", action="store_true",
                   help="Disable green-leaf auto-masking (requires --mask-dir).")
    p.add_argument("--no-leaf-thickness", action="store_true",
                   help="Disable display-only thickness/skirt geometry in the HTML viewer.")
    p.add_argument("--poisson", action="store_true",
                   help="Use experimental Poisson meshing instead of the default height-field mesh.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Import here so --help works even without dependencies installed
    from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[canopy] ERROR: input path does not exist: {input_path}")
        sys.exit(1)

    datasets = _discover_datasets(input_path, args.batch)
    print(f"[canopy] Processing {len(datasets)} dataset(s).")

    for dataset in datasets:
        cfg = CanopyReconstructionConfig(
            mask_dir=args.mask_dir,
            max_frames=args.max_frames,
            sample_stride=args.stride,
            max_candidates=args.max_candidates,
            min_mask_area=args.min_mask_area,
            coverage_threshold=args.coverage,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            z_scale=args.z_scale,
            max_hole_fill_distance_px=args.max_hole_fill_px,
            max_triangle_height_jump_m=args.max_triangle_jump,
            auto_mask=not args.no_auto_mask,
            add_leaf_thickness=not args.no_leaf_thickness,
            use_poisson_mesh=args.poisson,
            output_dir=(
                args.output
                if (args.output and not args.batch)
                else None
            ),
        )

        try:
            result = reconstruct_canopy(dataset, config=cfg)
            print(f"\n--- {dataset.name} ---")
            print(f"  Point cloud : {result.point_cloud_path}")
            print(f"  Mesh        : {result.mesh_path}")
            print(f"  HTML viewer : {result.viewer_path}")
            print(f"  Frames used : {result.frames_used}/{result.frames_available}")
            print(f"  Points      : {result.final_point_count:,}")
            print(f"  Triangles   : {result.final_triangle_count:,}")
        except Exception as exc:
            import traceback
            print(f"\n[canopy] FAILED for {dataset.name}: {exc}")
            traceback.print_exc()
            if not args.batch:
                sys.exit(1)

    print("\n[canopy] All done.")


if __name__ == "__main__":
    main()
