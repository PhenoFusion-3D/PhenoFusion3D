"""
Whole-sequence RGB-D scene reconstruction wrapper.

This is deliberately separate from canopy reconstruction.  It preserves the
background, tray, pots, gantry rails, and all plants in one scene output.  The
canopy pipeline remains the plant-only top-surface product for trait extraction.

Typical use:
    python reconstruct_scene_sequence.py --input data/main/test_plant_20230809133757
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))

from file_io.loader import (
    get_default_intrinsics,
    load_gantry_config,
    load_image_pairs,
    load_intrinsics,
    load_session_json,
)
from processing.registration_agent import AgentConfig
from processing.reconstructor import Reconstructor
from processing.rgbd import rgbd2pcd
from visualiser.viewer import write_point_cloud_viewer


def _paths_for_dataset(root: Path, layout: str) -> tuple[str, str]:
    if layout == "subdir" or (layout == "auto" and (root / "rgb").is_dir()):
        return str(root / "rgb"), str(root / "depth")
    return str(root), str(root)


def _infer_step_from_flat_tokens(root: Path) -> float | None:
    tokens = []
    for path in sorted(root.glob("rgb_*.png")):
        try:
            tokens.append(int(path.stem.split("_", 1)[1]))
        except Exception:
            pass
    if len(tokens) < 2:
        return None
    diffs = np.diff(np.asarray(tokens, dtype=np.float64)) / 1_000_000.0
    diffs = np.abs(diffs[np.abs(diffs) > 1e-12])
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _load_scene_pcd(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")
    return pcd


def _write_preview_images(pcd: o3d.geometry.PointCloud, out_dir: Path, prefix: str) -> list[str]:
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(cols) != len(pts):
        cols = np.ones((len(pts), 3), dtype=np.float64)
    if len(pts) == 0:
        return []

    def raster(a_vals, b_vals, name, resolution=0.006):
        amin, amax = float(a_vals.min()), float(a_vals.max())
        bmin, bmax = float(b_vals.min()), float(b_vals.max())
        pad = 0.05
        amin -= pad
        amax += pad
        bmin -= pad
        bmax += pad
        w = max(1, int((amax - amin) / resolution) + 1)
        h = max(1, int((bmax - bmin) / resolution) + 1)
        w = min(w, 1800)
        h = min(h, 1800)
        ai = np.clip(((a_vals - amin) / max(amax - amin, 1e-9) * (w - 1)).astype(np.int32), 0, w - 1)
        bi = np.clip(((b_vals - bmin) / max(bmax - bmin, 1e-9) * (h - 1)).astype(np.int32), 0, h - 1)
        acc = np.zeros((h, w, 3), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.int32)
        for channel in range(3):
            np.add.at(acc[:, :, channel], (bi, ai), cols[:, channel])
        np.add.at(cnt, (bi, ai), 1)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask = cnt > 0
        for channel in range(3):
            tmp = np.zeros((h, w), dtype=np.float64)
            tmp[mask] = acc[:, :, channel][mask] / cnt[mask]
            img[:, :, channel] = np.clip(tmp * 255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 0)
        path = out_dir / f"{prefix}_{name}.png"
        cv2.imwrite(str(path), img)
        return str(path)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return [
        raster(x, y, "front"),
        raster(x, z, "top"),
        raster(z, y, "side"),
    ]


def _run_tsdf_scene(root: Path, out_dir: Path, args) -> Path:
    pairs, K, dist, step_m, axis = _load_scene_inputs(root, args)

    recon = Reconstructor(
        pairs=pairs,
        K=K,
        dist=dist,
        depth_scale=1000.0,
        depth_trunc=args.depth_trunc,
        gantry_step_m=step_m * args.step,
        gantry_axis=axis,
        depth_min_mm=args.depth_min_mm,
        erode=True,
        inpaint=False,
        use_known_poses=True,
        tsdf_voxel_m=args.voxel,
        save_path=str(out_dir),
        mask_background=False,
        agent_config=AgentConfig(
            min_depth_validity=args.min_depth_validity,
            warn_depth_validity=max(args.min_depth_validity, args.warn_depth_validity),
            min_tsdf_points=args.min_tsdf_points,
        ),
    )
    pcd, succeed, fail = recon.run()
    pcd_path = out_dir / "scene_points.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    summary = {
        "mode": "known_pose_tsdf_scene",
        "dataset": str(root),
        "frames_requested": len(pairs),
        "frames_integrated": len(succeed),
        "frames_failed": len(fail),
        "points": len(pcd.points),
        "gantry_step_m_per_sample": step_m * args.step,
        "gantry_axis": axis,
        "depth_trunc_m": args.depth_trunc,
        "depth_min_mm": args.depth_min_mm,
        "voxel_m": args.voxel,
    }
    (out_dir / "scene_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pcd_path


def _load_scene_inputs(root: Path, args):
    rgb_dir, depth_dir = _paths_for_dataset(root, args.layout)
    pairs = load_image_pairs(rgb_dir, depth_dir, step=args.step)
    if args.max_frames and args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    intr = load_intrinsics(str(root / "kdc_intrinsics.txt"))
    if intr is None:
        intr = load_intrinsics(str(root / "kd_intrinsics.txt"))
    if intr:
        K, dist, _, _ = intr
    else:
        K, dist = get_default_intrinsics(1280, 720)

    gantry_cfg = load_gantry_config(root)
    if args.gantry_step_m is not None:
        step_m = args.gantry_step_m
        axis = args.gantry_axis
    elif gantry_cfg:
        step_m, axis = gantry_cfg
    else:
        step_m = _infer_step_from_flat_tokens(root) or 0.0015
        axis = args.gantry_axis
    return pairs, K, dist, step_m, axis


def _session_position_lookup(root: Path, pairs):
    session = load_session_json(root)
    if session is None and pairs:
        session = load_session_json(Path(pairs[0][0]).parent)
    frame_positions = (session or {}).get("frame_positions", {}) or {}
    pos_0 = None

    def lookup(rgb_path):
        nonlocal pos_0
        if not frame_positions:
            return None
        stem = Path(rgb_path).stem
        candidates = [stem]
        if stem.startswith("rgb_") or stem.startswith("depth_"):
            candidates.append(stem.split("_", 1)[1])
        for key in candidates:
            if key in frame_positions:
                pos = float(frame_positions[key])
                if pos_0 is None:
                    pos_0 = pos
                return pos - pos_0
        return None

    return lookup


def _run_pointcloud_scene(root: Path, out_dir: Path, args) -> Path:
    pairs, K, dist, step_m, axis = _load_scene_inputs(root, args)
    if not pairs:
        raise RuntimeError(f"No RGB/depth pairs found in {root}")

    merged = o3d.geometry.PointCloud()
    lookup_position = _session_position_lookup(root, pairs)
    skipped = []
    integrated = 0
    per_frame_voxel = max(args.voxel, 0.001)

    print(
        f"[scene] Known-pose point cloud: {len(pairs)} frames, "
        f"step={step_m * args.step * 1000:.2f}mm, axis={axis}, "
        f"voxel={args.voxel * 1000:.1f}mm"
    )

    for i, (rgb_path, depth_path) in enumerate(pairs):
        color_bgr = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth is None:
            skipped.append({"frame": i, "reason": "imread failed"})
            continue

        valid = int(np.count_nonzero(depth))
        validity = float(valid) / float(depth.size)
        if validity < args.min_depth_validity or valid < args.min_tsdf_points:
            skipped.append({
                "frame": i,
                "reason": f"depth validity {validity:.4f}, points {valid}",
            })
            continue

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        frame_pcd = rgbd2pcd(
            color_rgb,
            depth,
            K,
            dist=dist,
            depth_scale=1000.0,
            depth_trunc=args.depth_trunc,
            depth_min_mm=args.depth_min_mm,
            erode=args.erode_edges,
            inpaint=False,
            mask_background=False,
        )
        if frame_pcd.is_empty():
            skipped.append({"frame": i, "reason": "empty point cloud"})
            continue

        if per_frame_voxel > 0:
            frame_pcd = frame_pcd.voxel_down_sample(per_frame_voxel)

        T = np.eye(4)
        pos = lookup_position(rgb_path)
        T[axis, 3] = args.gantry_sign * (pos if pos is not None else i * step_m * args.step)
        frame_pcd.transform(T)
        merged += frame_pcd
        integrated += 1

        if i % 25 == 0 or i == len(pairs) - 1:
            print(f"[scene] point cloud {i + 1:4d}/{len(pairs)}")

    before_final = len(merged.points)
    if args.final_voxel and args.final_voxel > 0:
        merged = merged.voxel_down_sample(args.final_voxel)
    after_voxel = len(merged.points)
    if args.clean:
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    after_clean = len(merged.points)

    pcd_path = out_dir / "scene_points.ply"
    o3d.io.write_point_cloud(str(pcd_path), merged)
    summary = {
        "mode": "known_pose_pointcloud_scene",
        "dataset": str(root),
        "frames_requested": len(pairs),
        "frames_integrated": integrated,
        "frames_failed": len(skipped),
        "failed_samples": skipped[:50],
        "points_before_final_voxel": before_final,
        "points_after_final_voxel": after_voxel,
        "points": after_clean,
        "gantry_step_m_per_sample": step_m * args.step,
        "gantry_axis": axis,
        "depth_trunc_m": args.depth_trunc,
        "depth_min_mm": args.depth_min_mm,
        "per_frame_voxel_m": args.voxel,
        "final_voxel_m": args.final_voxel,
        "clean": args.clean,
    }
    (out_dir / "scene_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pcd_path


def _frame_pose_offset(i: int, rgb_path: str, lookup_position, step_m: float, step: int, sign: float) -> float:
    pos = lookup_position(rgb_path)
    if pos is None:
        pos = i * step_m * step
    return sign * float(pos)


def _world_bounds_for_heightfield(pairs, K, dist, step_m, axis, args, lookup_position):
    K_mat = np.asarray(K, dtype=np.float64)
    fx, fy = float(K_mat[0, 0]), float(abs(K_mat[1, 1]))
    cx, cy = float(K_mat[0, 2]), float(K_mat[1, 2])
    xs = []
    ys = []
    stride = max(args.pixel_step * 4, 8)
    for i, (rgb_path, depth_path) in enumerate(pairs):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        if dist is not None and any(d != 0.0 for d in dist):
            h, w = depth.shape[:2]
            map1, map2 = cv2.initUndistortRectifyMap(
                K_mat, np.asarray(dist, dtype=np.float64), None, K_mat, (w, h), cv2.CV_32FC1
            )
            depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)
        d = depth[::stride, ::stride].astype(np.float32) / 1000.0
        yy, xx = np.indices(d.shape, dtype=np.float32)
        u = xx * stride
        v = yy * stride
        valid = (d > 0) & (d <= args.depth_trunc)
        if args.depth_min_mm > 0:
            valid &= d >= (args.depth_min_mm / 1000.0)
        if not np.any(valid):
            continue
        x = (u[valid] - cx) * d[valid] / fx
        y = (v[valid] - cy) * d[valid] / fy
        offset = _frame_pose_offset(i, rgb_path, lookup_position, step_m, args.step, args.gantry_sign)
        if axis == 0:
            x = x + offset
        else:
            y = y + offset
        xs.extend([float(np.min(x)), float(np.max(x))])
        ys.extend([float(np.min(y)), float(np.max(y))])
    if not xs or not ys:
        raise RuntimeError("Could not infer scene bounds: no valid depth found.")
    pad = max(args.grid_cell * 10.0, 0.03)
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _run_heightfield_scene(root: Path, out_dir: Path, args) -> Path:
    pairs, K, dist, step_m, axis = _load_scene_inputs(root, args)
    if not pairs:
        raise RuntimeError(f"No RGB/depth pairs found in {root}")

    lookup_position = _session_position_lookup(root, pairs)
    min_x, max_x, min_y, max_y = _world_bounds_for_heightfield(
        pairs, K, dist, step_m, axis, args, lookup_position
    )
    cell = args.grid_cell
    width = int(np.ceil((max_x - min_x) / cell)) + 1
    height = int(np.ceil((max_y - min_y) / cell)) + 1
    if width * height > args.max_grid_cells:
        raise RuntimeError(
            f"Heightfield grid would be {width}x{height}={width * height:,} cells. "
            f"Increase --grid-cell or --max-grid-cells."
        )

    zbuf = np.full(width * height, np.inf, dtype=np.float32)
    colors = np.zeros((width * height, 3), dtype=np.uint8)
    counts = np.zeros(width * height, dtype=np.uint16)
    skipped = []
    integrated = 0

    K_mat = np.asarray(K, dtype=np.float64)
    fx, fy = float(K_mat[0, 0]), float(abs(K_mat[1, 1]))
    cx, cy = float(K_mat[0, 2]), float(K_mat[1, 2])
    map1 = map2 = None

    print(
        f"[scene] Full-scene heightfield: {len(pairs)} frames, "
        f"grid={width}x{height}, cell={cell * 1000:.1f}mm, "
        f"step={step_m * args.step * 1000:.2f}mm, axis={axis}, sign={args.gantry_sign:g}"
    )

    for i, (rgb_path, depth_path) in enumerate(pairs):
        color_bgr = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth is None:
            skipped.append({"frame": i, "reason": "imread failed"})
            continue

        if dist is not None and any(d != 0.0 for d in dist):
            if map1 is None:
                h0, w0 = depth.shape[:2]
                map1, map2 = cv2.initUndistortRectifyMap(
                    K_mat, np.asarray(dist, dtype=np.float64), None, K_mat, (w0, h0), cv2.CV_32FC1
                )
            color_bgr = cv2.undistort(color_bgr, K_mat, np.asarray(dist, dtype=np.float64))
            depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)

        stride = max(args.pixel_step, 1)
        d = depth[::stride, ::stride].astype(np.float32) / 1000.0
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)[::stride, ::stride]
        yy, xx = np.indices(d.shape, dtype=np.float32)
        u = xx * stride
        v = yy * stride
        valid = (d > 0) & (d <= args.depth_trunc)
        if args.depth_min_mm > 0:
            valid &= d >= (args.depth_min_mm / 1000.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count < args.min_tsdf_points:
            skipped.append({"frame": i, "reason": f"valid points {valid_count}"})
            continue

        z = d[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        offset = _frame_pose_offset(i, rgb_path, lookup_position, step_m, args.step, args.gantry_sign)
        if axis == 0:
            x = x + offset
        else:
            y = y + offset

        ix = np.floor((x - min_x) / cell).astype(np.int64)
        iy = np.floor((y - min_y) / cell).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        if not np.any(in_bounds):
            skipped.append({"frame": i, "reason": "outside bounds"})
            continue

        keys = iy[in_bounds] * width + ix[in_bounds]
        z = z[in_bounds]
        c = color_rgb[valid][in_bounds]

        order = np.lexsort((z, keys))
        keys_sorted = keys[order]
        first = np.r_[True, keys_sorted[1:] != keys_sorted[:-1]]
        best_idx = order[first]
        best_keys = keys[best_idx]
        best_z = z[best_idx]
        best_c = c[best_idx]

        replace = best_z < (zbuf[best_keys] - args.z_epsilon)
        near = np.abs(best_z - zbuf[best_keys]) <= args.z_epsilon
        if np.any(replace):
            rk = best_keys[replace]
            zbuf[rk] = best_z[replace]
            colors[rk] = best_c[replace]
            counts[rk] = 1
        if np.any(near):
            nk = best_keys[near]
            old_count = np.maximum(counts[nk].astype(np.uint16), 1)
            blended = (
                colors[nk].astype(np.uint16) * old_count[:, None]
                + best_c[near].astype(np.uint16)
            ) // (old_count[:, None] + 1)
            colors[nk] = np.clip(blended, 0, 255).astype(np.uint8)
            counts[nk] = np.minimum(old_count + 1, np.iinfo(np.uint16).max)

        integrated += 1
        if i % 25 == 0 or i == len(pairs) - 1:
            print(f"[scene] heightfield {i + 1:4d}/{len(pairs)}")

    valid_cells = np.isfinite(zbuf)
    key = np.flatnonzero(valid_cells)
    iy = key // width
    ix = key - iy * width
    pts = np.column_stack([
        min_x + (ix.astype(np.float32) + 0.5) * cell,
        min_y + (iy.astype(np.float32) + 0.5) * cell,
        zbuf[key],
    ])
    cols = colors[key].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols)

    pcd_path = out_dir / "scene_points.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    summary = {
        "mode": "full_scene_heightfield",
        "dataset": str(root),
        "frames_requested": len(pairs),
        "frames_integrated": integrated,
        "frames_failed": len(skipped),
        "failed_samples": skipped[:50],
        "points": len(pcd.points),
        "grid_width": width,
        "grid_height": height,
        "grid_cell_m": cell,
        "bounds_m": [min_x, max_x, min_y, max_y],
        "gantry_step_m_per_sample": step_m * args.step,
        "gantry_axis": axis,
        "gantry_sign": args.gantry_sign,
        "pixel_step": args.pixel_step,
        "depth_trunc_m": args.depth_trunc,
        "depth_min_mm": args.depth_min_mm,
        "z_epsilon_m": args.z_epsilon,
    }
    (out_dir / "scene_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pcd_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/view whole-sequence scene reconstruction.")
    parser.add_argument("--input", required=True, help="Dataset root.")
    parser.add_argument("--output", default=None, help="Output folder, default <input>/scene_sequence_codex.")
    parser.add_argument("--layout", choices=["auto", "subdir", "flat"], default="auto")
    parser.add_argument("--rerun", action="store_true", help="Rerun known-pose TSDF instead of reusing output/merge_pcd_live.ply.")
    parser.add_argument("--method", choices=["heightfield", "pointcloud", "tsdf"], default="pointcloud", help="Full-scene rerun method.")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--voxel", type=float, default=0.006)
    parser.add_argument("--final-voxel", type=float, default=0.004)
    parser.add_argument("--grid-cell", type=float, default=0.004)
    parser.add_argument("--pixel-step", type=int, default=2)
    parser.add_argument("--z-epsilon", type=float, default=0.006)
    parser.add_argument("--max-grid-cells", type=int, default=8_000_000)
    parser.add_argument("--depth-trunc", type=float, default=3.5)
    parser.add_argument("--depth-min-mm", type=int, default=0)
    parser.add_argument("--erode-edges", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--min-depth-validity", type=float, default=0.001)
    parser.add_argument("--warn-depth-validity", type=float, default=0.01)
    parser.add_argument("--min-tsdf-points", type=int, default=500)
    parser.add_argument("--gantry-step-m", type=float, default=None)
    parser.add_argument("--gantry-axis", type=int, default=1)
    parser.add_argument("--gantry-sign", type=float, default=1.0)
    parser.add_argument("--viewer-points", type=int, default=260000)
    args = parser.parse_args()

    root = Path(args.input).resolve()
    if not root.exists():
        raise SystemExit(f"Dataset does not exist: {root}")
    out_dir = Path(args.output).resolve() if args.output else root / "scene_sequence_codex"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = root / "output" / "merge_pcd_live.ply"
    if existing.exists() and not args.rerun:
        pcd_path = out_dir / "scene_points.ply"
        shutil.copy2(existing, pcd_path)
        source = str(existing)
    else:
        if args.method == "tsdf":
            pcd_path = _run_tsdf_scene(root, out_dir, args)
        elif args.method == "heightfield":
            pcd_path = _run_heightfield_scene(root, out_dir, args)
        else:
            pcd_path = _run_pointcloud_scene(root, out_dir, args)
        source = f"rerun:{args.method}"

    pcd = _load_scene_pcd(pcd_path)
    viewer_path = out_dir / "scene_viewer.html"
    write_point_cloud_viewer(
        pcd,
        viewer_path,
        title=f"{root.name} full sequence scene",
        max_points=args.viewer_points,
    )
    previews = _write_preview_images(pcd, out_dir, "scene")

    summary_path = out_dir / "scene_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update({
        "source": source,
        "point_cloud_path": str(pcd_path),
        "viewer_path": str(viewer_path),
        "preview_paths": previews,
        "points": len(pcd.points),
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[scene] points: {len(pcd.points):,}")
    print(f"[scene] point cloud: {pcd_path}")
    print(f"[scene] viewer: {viewer_path}")
    print(f"[scene] summary: {summary_path}")


if __name__ == "__main__":
    main()
