from __future__ import annotations

import argparse
from pathlib import Path

from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy
from processing.reconstructor import ReconstructionConfig, reconstruct_sequence


def _has_rgbd_frames(path: Path) -> bool:
    return any(path.glob("rgb_*.png")) and any(path.glob("depth_*.png"))


def _discover_records(input_path: str) -> list[Path]:
    root = Path(input_path)
    if _has_rgbd_frames(root):
        return [root]
    if not root.exists():
        return [root]
    records = sorted(path for path in root.iterdir() if path.is_dir() and _has_rgbd_frames(path))
    return records or [root]


def _batch_output_dir(base_output: str | None, input_path: str, record: Path, total_records: int, suffix: str) -> str | None:
    if total_records == 1:
        return base_output
    base = Path(base_output) if base_output else Path(input_path) / suffix
    return str(base / record.name)


def _batch_mask_dir(mask_dir: str | None, record: Path, total_records: int) -> str | None:
    if mask_dir is None or total_records == 1:
        return mask_dir
    base = Path(mask_dir)
    candidate = base / record.name
    return str(candidate if candidate.exists() else base)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run local RGB-D 3D reconstruction without the plant-scanning hardware stack."
    )
    parser.add_argument("--mode", choices=("rgbd", "canopy"), default="canopy", help="Pick the standard RGB-D stitcher or the plant-focused canopy reconstructor.")
    parser.add_argument("--input", required=True, help="Folder containing rgb_*.png, depth_*.png and intrinsics.")
    parser.add_argument("--camera-id", default="", help="Optional camera suffix if data is inside camera_<id>.")
    parser.add_argument("--step-size", type=int, default=8, help="Use every Nth frame to keep local runs practical.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on sampled frames.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this sampled frame index.")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this sampled frame index.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth divisor that converts raw depth to metres.")
    parser.add_argument("--depth-trunc", type=float, default=None, help="Discard points farther than this many metres. Defaults to auto-estimated.")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size used for cleanup and registration.")
    parser.add_argument("--min-fitness", type=float, default=0.01, help="Reject weak registrations after the warm-up frames.")
    parser.add_argument("--output-dir", default=None, help="Output folder. Defaults to <input>/reconstruction_local.")
    parser.add_argument("--green-filter", action="store_true", help="Keep only green-ish points in the final merged cloud.")
    parser.add_argument("--mask-dir", default=None, help="Optional directory containing mask_*.png for canopy mode.")
    parser.add_argument("--reference-token", type=int, default=None, help="Optional rgb/depth token to anchor canopy fusion.")
    parser.add_argument("--min-mask-area", type=int, default=180000, help="Ignore tiny masks when canopy mode selects frames.")
    parser.add_argument("--coverage-threshold", type=int, default=1, help="Canopy pixels must be supported by at least this many aligned frames.")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Vertical exaggeration for canopy mode. Keep 1.0 for metric output.")
    parser.add_argument("--no-auto-mask", action="store_true", help="Disable automatic green-leaf masks when no mask directory is present.")
    parser.add_argument("--canvas-padding", type=int, default=48, help="Extra pixels around the aligned canopy fusion canvas.")
    return parser


def main():
    args = build_parser().parse_args()
    records = _discover_records(args.input)
    if args.mode == "canopy":
        results = []
        for record in records:
            config = CanopyReconstructionConfig(
                mask_dir=_batch_mask_dir(args.mask_dir, record, len(records)),
                max_frames=args.max_frames or 9,
                min_mask_area=args.min_mask_area,
                reference_token=args.reference_token,
                coverage_threshold=args.coverage_threshold,
                z_scale=args.z_scale,
                output_dir=_batch_output_dir(args.output_dir, args.input, record, len(records), "canopy_batch"),
                auto_mask=not args.no_auto_mask,
                canvas_padding=args.canvas_padding,
            )
            results.append(reconstruct_canopy(record, config=config))

        print("Canopy reconstruction finished.")
        if len(results) > 1:
            print(f"Processed records: {len(results)}")
        for result in results:
            print(f"Input: {result.record_path}")
            print(f"Output directory: {result.output_dir}")
            print(f"Point cloud: {result.point_cloud_path}")
            print(f"Mesh: {result.mesh_path}")
            print(f"Interactive viewer: {result.viewer_path}")
            print(f"Masked RGB: {result.masked_rgb_path}")
            print(f"Summary: {result.summary_path}")
            print(f"Frames: {result.frames_used}/{result.frames_available} mask-backed frames")
            print(f"Reference frame token: {result.reference_token}")
            print(f"Final point count: {result.final_point_count}")
            print(f"Final triangle count: {result.final_triangle_count}")
        return

    results = []
    for record in records:
        config = ReconstructionConfig(
            camera_id=args.camera_id,
            step_size=args.step_size,
            max_frames=args.max_frames,
            start_index=args.start_index,
            end_index=args.end_index,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
            voxel_size=args.voxel_size,
            min_fitness=args.min_fitness,
            output_dir=_batch_output_dir(args.output_dir, args.input, record, len(records), "reconstruction_batch"),
            green_only=args.green_filter,
        )
        results.append(reconstruct_sequence(record, config=config))

    print("Reconstruction finished.")
    if len(results) > 1:
        print(f"Processed records: {len(results)}")
    for result in results:
        print(f"Input: {result.record_path}")
        print(f"Output directory: {result.output_dir}")
        print(f"Merged point cloud: {result.merged_point_cloud_path}")
        print(f"Poses: {result.pose_dir}")
        print(f"Summary: {result.summary_path}")
        print(
            "Frames: "
            f"{result.frames_registered}/{result.frames_selected} sampled "
            f"({result.frames_total} total in dataset)"
        )
        print(f"Final point count: {result.final_point_count}")


if __name__ == "__main__":
    main()
