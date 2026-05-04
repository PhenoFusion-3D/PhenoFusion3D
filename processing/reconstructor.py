from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from file_io.loader import get_default_intrinsics, load_image_pairs, load_intrinsics
from processing.icp import color_icp
from processing.rgbd import rgbd2pcd
from processing.utils import clean_pcd, green_filter_pcd


@dataclass
class ReconstructionConfig:
    camera_id: str = ""
    step_size: int = 8
    max_frames: int | None = None
    start_index: int = 0
    end_index: int | None = None
    depth_scale: float = 1000.0
    depth_trunc: float | None = None
    voxel_size: float = 0.005
    min_fitness: float = 0.01
    output_dir: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    green_only: bool = False


@dataclass
class ReconstructionResult:
    record_path: str
    output_dir: str
    merged_point_cloud_path: str
    pose_dir: str
    summary_path: str
    frames_total: int
    frames_selected: int
    frames_registered: int
    frames_failed: int
    final_point_count: int


def _resolve_record_path(record_path: Path, camera_id: str) -> Path:
    if camera_id:
        candidate = record_path / f"camera_{camera_id}"
        if candidate.exists():
            return candidate
    if any(record_path.glob("rgb_*.png")) and any(record_path.glob("depth_*.png")):
        return record_path
    camera_dirs = sorted(path for path in record_path.glob("camera_*") if path.is_dir())
    if len(camera_dirs) == 1:
        return camera_dirs[0]
    return record_path


def _frame_token(rgb_path: Path) -> str:
    return rgb_path.stem.replace("rgb_", "", 1)


def _load_frame(rgb_path: str, depth_path: str):
    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {rgb_path}")
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth frame: {depth_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb, depth


def _load_intrinsics_for_record(record_path: Path, first_rgb_path: str):
    for name in ("kdc_intrinsics.txt", "kd_intrinsics.txt"):
        intrinsics = load_intrinsics(str(record_path / name))
        if intrinsics is not None:
            return intrinsics

    sample = cv2.imread(first_rgb_path, cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(f"Failed to infer image size from {first_rgb_path}")
    height, width = sample.shape[:2]
    K, dist = get_default_intrinsics(width=width, height=height)
    return K, dist, width, height


def _estimate_depth_trunc(record_path: Path, pairs, depth_scale: float) -> float:
    sample_pairs = pairs[: min(12, len(pairs))]
    valid_depths = []
    for _, depth_path in sample_pairs:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        nz = depth[depth > 0]
        if nz.size:
            valid_depths.append(nz.astype(np.float32))
    if not valid_depths:
        return 3.0
    merged = np.concatenate(valid_depths)
    trunc = float(np.percentile(merged, 99.0) / depth_scale)
    trunc = max(1.5, min(trunc, 8.0))
    print(f"[reconstructor] Auto-selected depth_trunc={trunc:.3f} m")
    return trunc


def reconstruct_sequence(record_path: str | Path, config: ReconstructionConfig | None = None) -> ReconstructionResult:
    cfg = config or ReconstructionConfig()
    root = _resolve_record_path(Path(record_path), cfg.camera_id)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    step_size = max(1, int(cfg.step_size))
    all_pairs = load_image_pairs(str(root), str(root), step=1)
    frames_total = len(all_pairs)
    start_index = max(0, int(cfg.start_index))
    end_index = cfg.end_index if cfg.end_index is None else max(start_index, int(cfg.end_index))
    pairs = all_pairs[start_index:end_index:step_size]
    if cfg.max_frames is not None:
        pairs = pairs[: cfg.max_frames]
    if not pairs:
        raise ValueError(f"No RGB-D frame pairs found under {root}")

    K, dist, _, _ = _load_intrinsics_for_record(root, pairs[0][0])
    depth_trunc = cfg.depth_trunc if cfg.depth_trunc is not None else _estimate_depth_trunc(root, pairs, cfg.depth_scale)

    output_dir = Path(cfg.output_dir) if cfg.output_dir else root / "reconstruction_local"
    pose_dir = output_dir / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    merged = None
    target = None
    global_transform = np.eye(4)
    registered = 0
    failures = []

    for index, (rgb_path, depth_path) in enumerate(pairs):
        color_rgb, depth = _load_frame(rgb_path, depth_path)
        source = rgbd2pcd(
            color_rgb,
            depth,
            K,
            dist=dist,
            bbox=list(cfg.bbox) if cfg.bbox else None,
            depth_scale=cfg.depth_scale,
            depth_trunc=depth_trunc,
        )
        source = clean_pcd(source, voxel_size=cfg.voxel_size)
        frame_token = _frame_token(Path(rgb_path))

        if source.is_empty():
            failures.append({"frame": frame_token, "reason": "empty_point_cloud"})
            continue

        if merged is None:
            merged = copy.deepcopy(source)
            target = copy.deepcopy(source)
            registered = 1
            np.savetxt(pose_dir / f"{frame_token}_{cfg.camera_id}_pose.txt", global_transform)
            continue

        _, transform, fitness, inlier_rmse = color_icp(
            source,
            target,
            voxel_size=cfg.voxel_size,
        )
        if fitness < cfg.min_fitness and index > 2:
            failures.append(
                {
                    "frame": frame_token,
                    "reason": "low_fitness",
                    "fitness": float(fitness),
                    "inlier_rmse": float(inlier_rmse),
                }
            )
            continue

        global_transform = global_transform @ transform
        aligned = copy.deepcopy(source)
        aligned.transform(global_transform)
        merged += aligned
        target = source
        registered += 1
        np.savetxt(pose_dir / f"{frame_token}_{cfg.camera_id}_pose.txt", global_transform)

    if merged is None:
        raise RuntimeError("Reconstruction failed: every sampled frame produced an empty point cloud.")

    if cfg.green_only:
        merged = green_filter_pcd(merged)
        merged = clean_pcd(merged, voxel_size=cfg.voxel_size)

    if cfg.voxel_size > 0:
        merged = merged.voxel_down_sample(cfg.voxel_size)

    merged_path = output_dir / f"merge_pcd_cam{cfg.camera_id}.ply"
    summary_path = output_dir / "reconstruction_summary.json"
    o3d.io.write_point_cloud(str(merged_path), merged)

    summary = {
        "record_path": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "merged_point_cloud_path": str(merged_path.resolve()),
        "pose_dir": str(pose_dir.resolve()),
        "frames_total": frames_total,
        "frames_selected": len(pairs),
        "frames_registered": registered,
        "frames_failed": len(failures),
        "final_point_count": len(merged.points),
        "depth_trunc_used": depth_trunc,
        "config": asdict(cfg),
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ReconstructionResult(
        record_path=str(root.resolve()),
        output_dir=str(output_dir.resolve()),
        merged_point_cloud_path=str(merged_path.resolve()),
        pose_dir=str(pose_dir.resolve()),
        summary_path=str(summary_path.resolve()),
        frames_total=frames_total,
        frames_selected=len(pairs),
        frames_registered=registered,
        frames_failed=len(failures),
        final_point_count=len(merged.points),
    )
