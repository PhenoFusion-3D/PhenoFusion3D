#!/usr/bin/env python3
"""
Reorganize stakeholder-style RGB-D captures into an ICL-NUIM-like folder layout:

  <sequence>/
    rgb/0.png, 1.png, ...
    depth/0.png, 1.png, ...
    kdc_intrinsics.txt

Source layouts supported:
  - Single folder containing rgb_*.png and depth_*.png (RealSense / team convention)
  - Explicit --rgb-dir and --depth-dir
  - Optional --camera N: use <source>/camera_N/ as the frame folder

By default files are copied (originals kept). Use --move to move instead of copy.
Use --dry-run to print the plan without writing files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from natsort import natsorted


def _glob_pngs(folder: Path, pattern: str) -> list[Path]:
    return natsorted(folder.glob(pattern))


def _collect_pairs_flat(folder: Path) -> tuple[list[Path], list[Path]]:
    rgb_files = _glob_pngs(folder, "rgb_*.png")
    depth_files = _glob_pngs(folder, "depth_*.png")
    return rgb_files, depth_files


def _find_intrinsics(search_roots: list[Path]) -> Path | None:
    names = ("kdc_intrinsics.txt", "kd_intrinsics.txt", "kdc_intrinsics.json")
    for root in search_roots:
        if not root.is_dir():
            continue
        for name in names:
            p = root / name
            if p.is_file():
                return p
    return None


def _validate_intrinsics_json(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "K" not in data:
        raise ValueError(f"Intrinsics file missing 'K' matrix: {path}")


def reorganize_sequence(
    source_frames_dir: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
    move: bool = False,
    intrinsics_src: Path | None = None,
    step: int = 1,
    start_index: int = 0,
) -> tuple[int, Path | None]:
    """
    Returns (num_pairs_written, intrinsics_dest_or_none).
    """
    rgb_files, depth_files = _collect_pairs_flat(source_frames_dir)
    if not rgb_files or not depth_files:
        raise FileNotFoundError(
            f"No rgb_*.png / depth_*.png under {source_frames_dir}"
        )
    if len(rgb_files) != len(depth_files):
        raise ValueError(
            f"RGB count {len(rgb_files)} != depth count {len(depth_files)} "
            f"under {source_frames_dir}"
        )

    pairs = list(zip(rgb_files, depth_files))[:: max(1, step)]
    if not pairs:
        raise ValueError("No pairs after applying step")

    rgb_out = out_dir / "rgb"
    depth_out = out_dir / "depth"

    if not dry_run:
        rgb_out.mkdir(parents=True, exist_ok=True)
        depth_out.mkdir(parents=True, exist_ok=True)

    op = shutil.move if move else shutil.copy2
    for i, (rgb_path, depth_path) in enumerate(pairs):
        idx = start_index + i
        r_dst = rgb_out / f"{idx}.png"
        d_dst = depth_out / f"{idx}.png"
        print(f"  [{i + 1}/{len(pairs)}] {rgb_path.name} + {depth_path.name} -> rgb/{idx}.png depth/{idx}.png")
        if not dry_run:
            op(str(rgb_path), str(r_dst))
            op(str(depth_path), str(d_dst))

    intrinsics_dest = None
    if intrinsics_src and intrinsics_src.is_file():
        _validate_intrinsics_json(intrinsics_src)
        dest = out_dir / "kdc_intrinsics.txt"
        print(f"  intrinsics: {intrinsics_src} -> {dest.name}")
        intrinsics_dest = dest
        if not dry_run:
            shutil.copy2(intrinsics_src, dest)

    return len(pairs), intrinsics_dest


def process_one_capture(
    source_root: Path,
    out_dir: Path,
    *,
    camera: int | None,
    dry_run: bool,
    move: bool,
    intrinsics: Path | None,
    step: int,
) -> None:
    source_root = source_root.resolve()
    out_dir = out_dir.resolve()

    frames_dir = source_root
    if camera is not None:
        cam = source_root / f"camera_{camera}"
        if not cam.is_dir():
            raise FileNotFoundError(f"Camera folder not found: {cam}")
        frames_dir = cam

    search_roots = [frames_dir, source_root]
    intrinsics_path = intrinsics
    if intrinsics_path is None:
        intrinsics_path = _find_intrinsics(search_roots)

    print(f"\nSequence: {source_root.name}")
    print(f"  frames dir: {frames_dir}")
    print(f"  output:     {out_dir}")
    if intrinsics_path:
        print(f"  intrinsics: {intrinsics_path}")
    else:
        print("  intrinsics: (none found — add kdc_intrinsics.txt manually)")

    reorganize_sequence(
        frames_dir,
        out_dir,
        dry_run=dry_run,
        move=move,
        intrinsics_src=intrinsics_path,
        step=step,
        start_index=0,
    )


def _subdirs_with_rgbd(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        rgb, depth = _collect_pairs_flat(p)
        if rgb and depth:
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert stakeholder rgb_/depth_ PNG layout to ICL-NUIM-like rgb/N.png depth/N.png layout."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        help="Folder with rgb_*.png and depth_*.png (or parent of camera_N)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Output sequence root (default: same as --source, in-place)",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="RGB images folder (if separate from depth)",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Depth images folder (defaults to --rgb-dir)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Use <source>/camera_<N> for frames",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help="Process each immediate subfolder of this path that contains rgb_/depth_ pairs (e.g. data/main)",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="Explicit path to kdc_intrinsics.txt (or kd_intrinsics.txt JSON)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Use every Nth frame pair from the source ordering (default 1 = all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not create folders or copy files",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files into rgb/depth instead of copying (removes originals from flat layout)",
    )
    args = parser.parse_args(argv)

    if args.batch_root:
        root = args.batch_root.resolve()
        if not root.is_dir():
            print(f"ERROR: batch root not found: {root}", file=sys.stderr)
            return 1
        captures = _subdirs_with_rgbd(root)
        if not captures:
            print(f"No subfolders under {root} contain both rgb_*.png and depth_*.png")
            return 1
        print(f"Batch mode: {len(captures)} sequence(s) under {root}")
        for cap in captures:
            try:
                process_one_capture(
                    cap,
                    cap,
                    camera=args.camera,
                    dry_run=args.dry_run,
                    move=args.move,
                    intrinsics=args.intrinsics,
                    step=args.step,
                )
            except Exception as e:
                print(f"ERROR [{cap.name}]: {e}", file=sys.stderr)
                return 1
        return 0

    if args.rgb_dir is not None:
        rgb_dir = args.rgb_dir.resolve()
        depth_dir = (args.depth_dir or args.rgb_dir).resolve()
        out_dir = (args.out or rgb_dir.parent / f"{rgb_dir.name}_icl").resolve()
        if not rgb_dir.is_dir() or not depth_dir.is_dir():
            print("ERROR: --rgb-dir / --depth-dir must be existing directories", file=sys.stderr)
            return 1
        rgb_files = _glob_pngs(rgb_dir, "rgb_*.png")
        depth_files = _glob_pngs(depth_dir, "depth_*.png")
        if not rgb_files or not depth_files:
            print("ERROR: expected rgb_*.png and depth_*.png in the given dirs", file=sys.stderr)
            return 1
        if len(rgb_files) != len(depth_files):
            print("ERROR: RGB and depth file counts do not match", file=sys.stderr)
            return 1
        pairs = list(zip(rgb_files, depth_files))[:: max(1, args.step)]
        search_roots = [rgb_dir.parent, depth_dir.parent, out_dir.parent]
        intrinsics_path = args.intrinsics or _find_intrinsics(search_roots)
        if not args.dry_run:
            (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
            (out_dir / "depth").mkdir(parents=True, exist_ok=True)
        op = shutil.move if args.move else shutil.copy2
        for i, (r, d) in enumerate(pairs):
            print(f"  [{i + 1}/{len(pairs)}] -> rgb/{i}.png depth/{i}.png")
            if not args.dry_run:
                op(str(r), str(out_dir / "rgb" / f"{i}.png"))
                op(str(d), str(out_dir / "depth" / f"{i}.png"))
        if intrinsics_path and intrinsics_path.is_file():
            if not args.dry_run:
                shutil.copy2(intrinsics_path, out_dir / "kdc_intrinsics.txt")
            print(f"  intrinsics -> {out_dir / 'kdc_intrinsics.txt'}")
        print(f"\nDone. Output: {out_dir}")
        return 0

    if args.source is None:
        parser.error("Provide --source, or --batch-root, or --rgb-dir")

    source = args.source.resolve()
    out_dir = (args.out or source).resolve()

    if not source.is_dir():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        return 1

    try:
        process_one_capture(
            source,
            out_dir,
            camera=args.camera,
            dry_run=args.dry_run,
            move=args.move,
            intrinsics=args.intrinsics,
            step=args.step,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nDone. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
