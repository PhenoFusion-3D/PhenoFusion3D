"""
Reconstruct multiple plants from one long top-down gantry capture.

The regular canopy pipeline intentionally picks one high-quality local frame
window.  This wrapper chooses several well-spaced reference windows, runs the
same repaired canopy reconstruction for each plant, and also writes a combined
sequence mesh for browsing.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import open3d as o3d
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))

from file_io.loader import load_intrinsics
from processing.canopy import (
    CanopyReconstructionConfig,
    _attach_candidate_positions,
    _auto_leaf_mask,
    _clean_mask,
    _discover_image_pairs,
    _fill_depth_inside_mask,
    _fill_small_holes,
    _frame_positions_m,
    _load_auto_candidates,
    _smooth_in_mask,
    reconstruct_canopy,
)
from visualiser.viewer import write_canopy_mesh_viewer


def _component_items(mask: np.ndarray, min_area: int, border: int = 30) -> list[dict]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    height, width = mask.shape
    items: list[dict] = []
    for label in range(1, num):
        x, y, w, h, area = stats[label]
        area = int(area)
        if area < min_area:
            continue
        edge = (
            x < border
            or y < border
            or x + w > width - border
            or y + h > height - border
        )
        comp = np.zeros(mask.shape, dtype=np.uint8)
        comp[labels == label] = 255
        items.append({
            "label": int(label),
            "area": area,
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": [float(centroids[label][0]), float(centroids[label][1])],
            "edge": bool(edge),
            "mask": comp,
        })
    return items


def _component_reference_candidates(dataset: Path, args) -> tuple[list[dict], dict]:
    cfg = CanopyReconstructionConfig(
        sample_stride=args.stride,
        min_mask_area=max(1, args.component_min_area),
        min_component_area=args.component_min_area,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    pairs = _discover_image_pairs(dataset, sample_stride=args.stride)
    positions, motion_info = _frame_positions_m(dataset, pairs)
    if not pairs:
        raise RuntimeError(f"No RGB-D frames found under {dataset}")

    frame_lookup = {int(t): (r, d) for t, r, d in pairs}
    candidates: list[dict] = []
    for token, rgb_path, depth_path in pairs:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        mask = _auto_leaf_mask(rgb, cfg)
        for comp in _component_items(mask, args.component_min_area):
            cx, cy = comp["center"]
            dist_to_center = abs(cy - rgb.shape[0] * 0.5) / max(rgb.shape[0] * 0.5, 1)
            center_bonus = max(0.25, 1.0 - 0.65 * dist_to_center)
            edge_factor = args.component_edge_penalty if comp["edge"] else 1.0
            candidates.append({
                "token": int(token),
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "position_m": float(positions.get(int(token), len(candidates))),
                "score": float(comp["area"]) * float(edge_factor) * float(center_bonus),
                "mask_area": int(comp["area"]),
                "bbox": comp["bbox"],
                "center": comp["center"],
                "edge": comp["edge"],
            })

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    if not ranked:
        raise RuntimeError("No component-level plant candidates found.")

    min_score = float(ranked[0]["score"]) * float(args.min_score_ratio)
    chosen: list[dict] = []
    chosen_tracks: list[set[tuple[int, int]]] = []
    full_pairs = _discover_image_pairs(dataset, sample_stride=1)
    track_cache: dict[int, list[dict]] = {}
    for item in ranked:
        if float(item["score"]) < min_score:
            continue
        pos = float(item["position_m"])
        if any(abs(pos - float(prev["position_m"])) < args.reference_spacing_m for prev in chosen):
            continue
        track = _track_instance_components(
            dataset,
            item,
            args,
            pairs=full_pairs,
            component_cache=track_cache,
        )
        track_keys = _track_identity_keys(track)
        if any(
            _track_overlap_fraction(track_keys, prev_track) >= args.track_overlap_ratio
            for prev_track in chosen_tracks
        ):
            continue
        item["tracked_token_count"] = len(set(track.keys()))
        item["tracked_component_count"] = len(track_keys)
        chosen.append(item)
        chosen_tracks.append(track_keys)
        if args.max_instances > 0 and len(chosen) >= args.max_instances:
            break

    chosen = sorted(chosen, key=lambda item: float(item["position_m"]))
    for item in chosen:
        item["frame_lookup"] = frame_lookup
    return chosen, motion_info


def _track_instance_components(
    dataset: Path,
    reference: dict,
    args,
    *,
    pairs: list[tuple[int, Path, Path]] | None = None,
    component_cache: dict[int, list[dict]] | None = None,
) -> dict[int, dict]:
    """Track one connected leaf component through neighbouring frames."""
    cfg = CanopyReconstructionConfig(
        sample_stride=1,
        min_mask_area=max(1, args.component_min_area),
        min_component_area=args.component_min_area,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    pairs = pairs if pairs is not None else _discover_image_pairs(dataset, sample_stride=1)
    if not pairs:
        raise RuntimeError(f"No RGB-D frames found under {dataset}")
    token_to_index = {int(t): i for i, (t, _, _) in enumerate(pairs)}
    if int(reference["token"]) not in token_to_index:
        raise RuntimeError(f"Reference token not found in full pair list: {reference['token']}")

    component_cache = component_cache if component_cache is not None else {}

    def components_for(i: int) -> list[dict]:
        token, rgb_path, _ = pairs[i]
        if int(token) in component_cache:
            return component_cache[int(token)]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            component_cache[int(token)] = []
            return []
        mask = _auto_leaf_mask(rgb, cfg)
        comps = _component_items(mask, args.component_min_area)
        component_cache[int(token)] = comps
        return comps

    def choose_nearest(comps: list[dict], center: list[float], prev_area: int | None) -> dict | None:
        if not comps:
            return None
        cx, cy = float(center[0]), float(center[1])
        ranked = sorted(
            comps,
            key=lambda comp: (
                (float(comp["center"][0]) - cx) ** 2 + (float(comp["center"][1]) - cy) ** 2,
                -int(comp["area"]),
            ),
        )
        best = ranked[0]
        dist = float(
            ((float(best["center"][0]) - cx) ** 2 + (float(best["center"][1]) - cy) ** 2) ** 0.5
        )
        if dist > args.track_max_step_px:
            return None
        if prev_area is not None and prev_area > 0:
            ratio = max(int(best["area"]), prev_area) / max(min(int(best["area"]), prev_area), 1)
            if ratio > args.track_max_area_ratio and best["edge"]:
                return None
        return best

    ref_i = token_to_index[int(reference["token"])]
    ref_comp = choose_nearest(components_for(ref_i), reference["center"], None)
    if ref_comp is None:
        raise RuntimeError(f"Could not recover reference component for token {reference['token']}")

    tracked: dict[int, dict] = {int(reference["token"]): ref_comp}
    for direction in (-1, 1):
        prev = ref_comp
        i = ref_i + direction
        while 0 <= i < len(pairs):
            token = int(pairs[i][0])
            comp = choose_nearest(components_for(i), prev["center"], int(prev["area"]))
            if comp is None:
                break
            tracked[token] = comp
            prev = comp
            i += direction

    return tracked


def _track_identity_keys(track: dict[int, dict]) -> set[tuple[int, int]]:
    """Identify tracked components by frame token and connected-component label."""
    keys: set[tuple[int, int]] = set()
    for token, comp in track.items():
        keys.add((int(token), int(comp.get("label", -1))))
    return keys


def _track_overlap_fraction(
    track_keys: set[tuple[int, int]],
    previous_keys: set[tuple[int, int]],
) -> float:
    if not track_keys or not previous_keys:
        return 0.0
    return len(track_keys & previous_keys) / max(min(len(track_keys), len(previous_keys)), 1)


def _write_tracked_instance_masks(
    dataset: Path,
    out_dir: Path,
    reference: dict,
    args,
) -> tuple[Path, int]:
    """Track one connected leaf component and write masks for canopy fusion."""
    tracked = _track_instance_components(dataset, reference, args)
    mask_dir = out_dir / "instance_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for token, comp in tracked.items():
        cv2.imwrite(str(mask_dir / f"mask_{token}.png"), comp["mask"])
    return mask_dir, len(tracked)


def _choose_references(
    dataset: Path,
    args,
) -> tuple[list[dict], dict]:
    cfg = CanopyReconstructionConfig(
        sample_stride=args.stride,
        min_mask_area=args.min_mask_area,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    pairs = _discover_image_pairs(dataset, sample_stride=args.stride)
    if not pairs:
        raise RuntimeError(f"No RGB-D frames found under {dataset}")

    candidates = _load_auto_candidates(pairs, args.output / "_sequence_auto_masks", cfg)
    positions, motion_info = _frame_positions_m(dataset, pairs)
    _attach_candidate_positions(candidates, positions)
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    if not ranked:
        raise RuntimeError("No usable plant candidates found.")

    best_score = float(ranked[0]["score"])
    min_score = best_score * float(args.min_score_ratio)
    chosen: list[dict] = []
    for item in ranked:
        if float(item["score"]) < min_score:
            continue
        pos = float(item["position_m"])
        if all(abs(pos - float(prev["position_m"])) >= args.reference_spacing_m for prev in chosen):
            chosen.append(item)
        if args.max_instances > 0 and len(chosen) >= args.max_instances:
            break

    chosen = sorted(chosen, key=lambda item: float(item["position_m"]))
    return chosen, motion_info


def _read_geometry(path: str, kind: str):
    if kind == "pcd":
        return o3d.io.read_point_cloud(path)
    return o3d.io.read_triangle_mesh(path)


def _translate_xy(geom, dx: float, dy: float = 0.0):
    if geom is not None and not geom.is_empty():
        geom.translate((float(dx), float(dy), 0.0), relative=True)
    return geom


def _geometry_xy_bounds(geom) -> tuple[float, float, float, float]:
    if geom is None or geom.is_empty():
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
    else:
        pts = np.asarray(geom.vertices)
    if pts.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].min()),
        float(pts[:, 1].max()),
    )


def _non_overlapping_y_offsets(
    bounds: list[tuple[float, float, float, float]],
    targets_xy: list[tuple[float, float]],
    margin_m: float,
) -> list[tuple[float, float]]:
    if not bounds or not targets_xy:
        return []
    origin_x, origin_y = targets_xy[0]
    cursor_max_y: float | None = None
    offsets: list[tuple[float, float]] = []
    margin = max(0.0, float(margin_m))
    for (min_x, _max_x, min_y, max_y), (world_x, world_y) in zip(bounds, targets_xy):
        dx = float(world_x - origin_x)
        dy = float(world_y - origin_y)
        placed_min_y = min_y + dy
        placed_max_y = max_y + dy
        if cursor_max_y is not None and placed_min_y < cursor_max_y + margin:
            dy += (cursor_max_y + margin) - placed_min_y
            placed_min_y = min_y + dy
            placed_max_y = max_y + dy
        cursor_max_y = placed_max_y if cursor_max_y is None else max(cursor_max_y, placed_max_y)
        offsets.append((dx, dy))
    return offsets


def _reference_world_xy(reference: dict, dataset: Path) -> tuple[float, float]:
    """Approximate sequence-layout position from component centre and depth."""
    intr = load_intrinsics(str(dataset / "kdc_intrinsics.txt"))
    if intr is None:
        intr = load_intrinsics(str(dataset / "kd_intrinsics.txt"))
    if intr is None:
        return 0.0, float(reference.get("position_m", 0.0))
    K, _, _, _ = intr
    rgb = cv2.imread(str(reference["rgb_path"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(reference["depth_path"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        return 0.0, float(reference.get("position_m", 0.0))

    cfg = CanopyReconstructionConfig(min_component_area=1)
    mask = _auto_leaf_mask(rgb, cfg)
    comps = _component_items(mask, 1)
    if comps:
        ref_comp = min(
            comps,
            key=lambda comp: (
                float(comp["center"][0]) - float(reference["center"][0])
            ) ** 2
            + (
                float(comp["center"][1]) - float(reference["center"][1])
            ) ** 2,
        )
        comp_mask = ref_comp["mask"] > 0
    else:
        comp_mask = depth > 0
    vals = depth[comp_mask & (depth >= 500) & (depth <= 4000)]
    z_m = float(np.median(vals)) / 1000.0 if vals.size else 1.8
    cx, cy = reference.get("center", [float(K[0, 2]), float(K[1, 2])])
    fx, fy = float(K[0, 0]), float(abs(K[1, 1]))
    px, py = float(K[0, 2]), float(K[1, 2])
    world_x = (float(cx) - px) * z_m / fx
    world_y = float(reference.get("position_m", 0.0)) + (float(cy) - py) * z_m / fy
    return world_x, world_y


def _load_dataset_intrinsics(dataset: Path):
    intr = load_intrinsics(str(dataset / "kdc_intrinsics.txt"))
    if intr is None:
        intr = load_intrinsics(str(dataset / "kd_intrinsics.txt"))
    if intr is None:
        raise FileNotFoundError(
            f"Missing kdc_intrinsics.txt or kd_intrinsics.txt under {dataset}"
        )
    return intr


def _estimate_phase_motion(
    dataset: Path,
    pairs: list[tuple[int, Path, Path]],
    K: np.ndarray,
    args,
) -> dict | None:
    """Estimate gantry axis, sign, and metric step from image motion.

    Some legacy captures name frames with position-like tokens, but those tokens
    are not always calibrated metres.  Phase correlation over the static scene
    gives a stronger motion estimate for global orthographic fusion.
    """
    if len(pairs) < 3:
        return None

    gap = min(max(10, int(len(pairs) * 0.08)), 80, len(pairs) - 1)
    if gap <= 0:
        return None
    starts = np.linspace(0, len(pairs) - gap - 1, num=min(7, len(pairs) - gap), dtype=int)
    fx, fy = float(K[0, 0]), float(abs(K[1, 1]))
    estimates: list[dict] = []

    for start in starts:
        token_a, rgb_a_path, depth_a_path = pairs[int(start)]
        token_b, rgb_b_path, _ = pairs[int(start) + gap]
        gray_a = cv2.imread(str(rgb_a_path), cv2.IMREAD_GRAYSCALE)
        gray_b = cv2.imread(str(rgb_b_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(depth_a_path), cv2.IMREAD_UNCHANGED)
        if gray_a is None or gray_b is None or depth is None or gray_a.shape != gray_b.shape:
            continue
        try:
            (sx, sy), response = cv2.phaseCorrelate(
                gray_a.astype(np.float32), gray_b.astype(np.float32)
            )
        except cv2.error:
            continue
        if response < 0.04:
            continue
        depth_m = depth.astype(np.float32) / 1000.0
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if args.depth_min:
            valid &= depth_m >= float(args.depth_min) / 1000.0
        if args.depth_max:
            valid &= depth_m <= float(args.depth_max) / 1000.0
        if np.count_nonzero(valid) < 500:
            continue
        z_m = float(np.median(depth_m[valid]))
        axis = 0 if abs(float(sx)) >= abs(float(sy)) else 1
        signed_shift_px_per_frame = (float(sx) if axis == 0 else float(sy)) / float(gap)
        focal = fx if axis == 0 else fy
        step_m = abs(signed_shift_px_per_frame) * z_m / max(focal, 1e-6)
        if step_m <= 1e-6:
            continue
        estimates.append({
            "start_token": int(token_a),
            "end_token": int(token_b),
            "gap_frames": int(gap),
            "axis": int(axis),
            "shift_xy_px": [float(sx), float(sy)],
            "shift_axis_px_per_frame": float(signed_shift_px_per_frame),
            "response": float(response),
            "median_depth_m": z_m,
            "step_m_per_frame": float(step_m),
        })

    if not estimates:
        return None

    axes = [item["axis"] for item in estimates]
    axis = 0 if axes.count(0) >= axes.count(1) else 1
    axis_estimates = [item for item in estimates if item["axis"] == axis]
    shifts = np.asarray([item["shift_axis_px_per_frame"] for item in axis_estimates], dtype=np.float64)
    steps = np.asarray([item["step_m_per_frame"] for item in axis_estimates], dtype=np.float64)
    signed_shift = float(np.median(shifts))
    # If the scene moves positive in image coordinates, the camera moved in the
    # opposite world-image axis direction.  This sign makes static points align
    # when projected as world = camera_ray + camera_offset.
    camera_sign = -1.0 if signed_shift >= 0.0 else 1.0
    return {
        "source": "phase_correlation",
        "axis": int(axis),
        "camera_sign": float(camera_sign),
        "signed_shift_px_per_frame": signed_shift,
        "step_m_per_frame": float(np.median(steps)),
        "samples": axis_estimates,
    }


def _global_frame_offsets(
    dataset: Path,
    pairs: list[tuple[int, Path, Path]],
    K: np.ndarray,
    args,
) -> tuple[dict[int, float], dict]:
    positions, token_info = _frame_positions_m(dataset, pairs)
    phase_info = None
    if args.motion_source in ("auto", "phase"):
        phase_info = _estimate_phase_motion(dataset, pairs, K, args)
        if args.motion_source == "phase" and phase_info is None:
            raise RuntimeError("Phase motion calibration failed; try --motion-source token.")

    if phase_info is not None:
        ordered_tokens = [int(t) for t, _, _ in pairs]
        raw_positions = np.asarray(
            [float(positions.get(t, i)) for i, t in enumerate(ordered_tokens)],
            dtype=np.float64,
        )
        raw_steps = np.abs(np.diff(raw_positions))
        raw_steps = raw_steps[raw_steps > 1e-12]
        if raw_steps.size:
            raw_scale = float(phase_info["step_m_per_frame"]) / float(np.median(raw_steps))
            first = float(raw_positions[0])
            offsets = {
                int(t): float(phase_info["camera_sign"]) * (float(pos) - first) * raw_scale
                for t, pos in zip(ordered_tokens, raw_positions)
            }
        else:
            offsets = {
                int(t): float(phase_info["camera_sign"]) * i * float(phase_info["step_m_per_frame"])
                for i, t in enumerate(ordered_tokens)
            }
        return offsets, {
            "source": "phase_correlation_scaled_tokens",
            "axis": int(phase_info["axis"]),
            "camera_sign": float(phase_info["camera_sign"]),
            "step_m_per_frame": float(phase_info["step_m_per_frame"]),
            "token_position_source": token_info.get("source", "unknown"),
            "token_median_step_m": token_info.get("median_step_m", 0.0),
            "phase": phase_info,
        }

    sign = float(args.gantry_sign)
    first_token = int(pairs[0][0])
    first_pos = float(positions.get(first_token, 0.0))
    offsets = {
        int(t): sign * (float(positions.get(int(t), i)) - first_pos)
        for i, (t, _, _) in enumerate(pairs)
    }
    return offsets, {
        "source": token_info.get("source", "token_or_config"),
        "axis": int(args.gantry_axis),
        "camera_sign": sign,
        "step_m_per_frame": token_info.get("median_step_m", 0.0),
        "token": token_info,
    }


def _project_valid_pixels(
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask_u8: np.ndarray,
    K: np.ndarray,
    offset_m: float,
    axis: int,
    stride: int,
    cfg: CanopyReconstructionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    clean_mask = _clean_mask(mask_u8, keep_largest=False, min_component_area=cfg.min_component_area)
    foreground_depth, stats = _fill_depth_inside_mask(depth_mm, clean_mask, cfg)
    step = max(1, int(stride))
    d_mm = foreground_depth[::step, ::step]
    valid = d_mm > 0
    if not np.any(valid):
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            stats,
        )
    yy, xx = np.indices(d_mm.shape, dtype=np.float32)
    u = xx * float(step)
    v = yy * float(step)
    z = d_mm[valid].astype(np.float32) / 1000.0
    fx, fy = float(K[0, 0]), float(abs(K[1, 1]))
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    if int(axis) == 0:
        x = x + float(offset_m)
    else:
        y = y + float(offset_m)
    colors = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[::step, ::step][valid]
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), {
        **stats,
        "colors": colors,
    }


def _global_bounds(
    dataset: Path,
    pairs: list[tuple[int, Path, Path]],
    offsets: dict[int, float],
    motion: dict,
    K: np.ndarray,
    cfg: CanopyReconstructionConfig,
    args,
) -> tuple[float, float, float, float, list[dict]]:
    xs: list[float] = []
    ys: list[float] = []
    frame_stats: list[dict] = []
    stride = max(int(args.pixel_step) * 3, 6)
    for i, (token, rgb_path, depth_path) in enumerate(pairs):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        mask = _auto_leaf_mask(rgb, cfg)
        if int((mask > 0).sum()) < int(args.component_min_area):
            continue
        try:
            x, y, _z, stats = _project_valid_pixels(
                rgb, depth, mask, K, offsets[int(token)], int(motion["axis"]), stride, cfg
            )
        except RuntimeError:
            continue
        if x.size == 0:
            continue
        xs.extend([float(x.min()), float(x.max())])
        ys.extend([float(y.min()), float(y.max())])
        frame_stats.append({
            "token": int(token),
            "mask_area": int((mask > 0).sum()),
            "sample_points": int(x.size),
            "depth_valid_fraction": float(stats.get("depth_valid_fraction", 0.0)),
        })
        if i % 100 == 0:
            print(f"[global] bounds {i + 1:4d}/{len(pairs)}")
    if not xs or not ys:
        raise RuntimeError("Global canopy fusion found no plant depth pixels.")
    pad = max(float(args.grid_cell) * 12.0, 0.03)
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad, frame_stats


def _fill_global_grid(
    depth_grid: np.ndarray,
    color_grid: np.ndarray,
    valid_mask: np.ndarray,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(valid_mask):
        raise RuntimeError("Cannot fill an empty global grid.")
    support = _fill_small_holes(
        valid_mask.astype(np.uint8) * 255,
        max_hole_area=int(args.global_max_hole_area),
    ) > 0
    if args.max_hole_fill_px > 0:
        dist = distance_transform_edt(~valid_mask)
        support &= valid_mask | (dist <= int(args.max_hole_fill_px))
    else:
        support = valid_mask

    filled_depth = depth_grid.copy()
    filled_color = color_grid.copy()
    missing = support & ~valid_mask
    if np.any(missing):
        _, nearest = distance_transform_edt(~valid_mask, return_indices=True)
        filled_depth[missing] = filled_depth[nearest[0][missing], nearest[1][missing]]
        filled_color[missing] = filled_color[nearest[0][missing], nearest[1][missing]]
    filled_depth[~support] = np.inf
    filled_color[~support] = 0
    return filled_depth, filled_color, support


def _mesh_from_global_grid(
    depth_grid: np.ndarray,
    color_grid: np.ndarray,
    mask: np.ndarray,
    min_x: float,
    min_y: float,
    cell: float,
    args,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, np.ndarray]:
    valid = mask & np.isfinite(depth_grid)
    if not np.any(valid):
        raise RuntimeError("Global grid has no valid mesh cells.")

    depth_values = depth_grid[valid]
    baseline = float(np.percentile(depth_values, 95.0))
    height_map = np.zeros(depth_grid.shape, dtype=np.float32)
    height_map[valid] = np.maximum(0.0, baseline - depth_grid[valid])
    if np.any(valid):
        upper = float(np.percentile(height_map[valid], 99.5))
        height_map = np.clip(height_map, 0.0, max(upper, 1e-4))
        if float(args.smooth_sigma) > 0:
            height_map = _smooth_in_mask(height_map, valid, sigma=float(args.smooth_sigma))
            height_map[~valid] = 0.0

    ys, xs = np.where(valid)
    points = np.column_stack([
        min_x + (xs.astype(np.float32) + 0.5) * float(cell),
        min_y + (ys.astype(np.float32) + 0.5) * float(cell),
        height_map[ys, xs],
    ]).astype(np.float64)
    colors = color_grid[ys, xs].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    id_map = -np.ones(valid.shape, dtype=np.int32)
    id_map[ys, xs] = np.arange(len(xs), dtype=np.int32)
    triangles: list[list[int]] = []
    h, w = valid.shape
    max_jump = float(args.max_triangle_jump)
    for yy in range(h - 1):
        row0 = id_map[yy]
        row1 = id_map[yy + 1]
        for xx in range(w - 1):
            ids = [row0[xx], row0[xx + 1], row1[xx], row1[xx + 1]]
            if ids[0] >= 0 and ids[1] >= 0 and ids[2] >= 0:
                hs = [height_map[yy, xx], height_map[yy, xx + 1], height_map[yy + 1, xx]]
                if float(max(hs) - min(hs)) <= max_jump:
                    triangles.append([int(ids[0]), int(ids[2]), int(ids[1])])
            if ids[1] >= 0 and ids[2] >= 0 and ids[3] >= 0:
                hs = [height_map[yy, xx + 1], height_map[yy + 1, xx], height_map[yy + 1, xx + 1]]
                if float(max(hs) - min(hs)) <= max_jump:
                    triangles.append([int(ids[1]), int(ids[2]), int(ids[3])])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    tri_arr = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(tri_arr)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return pcd, mesh, height_map


def _submesh_from_component(
    depth_grid: np.ndarray,
    color_grid: np.ndarray,
    labels: np.ndarray,
    label: int,
    min_x: float,
    min_y: float,
    cell: float,
    args,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    comp_mask = labels == int(label)
    return _mesh_from_global_grid(depth_grid, color_grid, comp_mask, min_x, min_y, cell, args)[:2]


def _save_global_grid_images(
    out_dir: Path,
    depth_grid: np.ndarray,
    color_grid: np.ndarray,
    mask: np.ndarray,
    height_map: np.ndarray,
) -> None:
    rgb = color_grid.copy()
    rgb[~mask] = 0
    cv2.imwrite(str(out_dir / "global_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "global_mask.png"), mask.astype(np.uint8) * 255)
    depth_vis = np.zeros(mask.shape, dtype=np.uint8)
    if np.any(mask):
        vals = depth_grid[mask & np.isfinite(depth_grid)]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
            scaled = np.clip((depth_grid - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
            depth_vis[mask] = (255.0 * (1.0 - scaled[mask])).astype(np.uint8)
    cv2.imwrite(str(out_dir / "global_depth_vis.png"), depth_vis)
    height_vis = np.zeros(mask.shape, dtype=np.uint8)
    if np.any(mask):
        max_h = float(np.percentile(height_map[mask], 99.5))
        if max_h > 1e-6:
            height_vis[mask] = np.clip(255.0 * height_map[mask] / max_h, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "global_height_vis.png"), height_vis)


def _write_global_index(output: Path, rows: list[dict], combined_viewer: Path) -> Path:
    table_rows = []
    for row in rows:
        viewer = os.path.relpath(row["viewer_path"], start=output)
        mesh = os.path.relpath(row["mesh_path"], start=output)
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{row['area_cells']:,}</td>"
            f"<td>{row['points']:,}</td>"
            f"<td>{row['triangles']:,}</td>"
            f'<td><a href="{html.escape(viewer)}">viewer</a></td>'
            f'<td><a href="{html.escape(mesh)}">mesh</a></td>'
            "</tr>"
        )
    rel_combined = os.path.relpath(combined_viewer, start=output)
    css = (
        "body{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;"
        "padding:18px}a{color:#93c5fd}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #374151;padding:7px 9px;text-align:left}"
        "th{background:#1f2937}"
    )
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Global canopy sequence</title>"
        f"<style>{css}</style></head><body><h1>Global Canopy Sequence</h1>"
        f'<p><a href="{html.escape(rel_combined)}">Open combined sequence viewer</a></p>'
        "<table><thead><tr><th>Plant</th><th>Cells</th><th>Points</th>"
        "<th>Triangles</th><th>Viewer</th><th>Mesh</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></body></html>"
    )
    index = output / "sequence_index.html"
    index.write_text(doc, encoding="utf-8")
    return index


def _run_global_ortho_sequence(dataset: Path, args) -> None:
    pairs = _discover_image_pairs(dataset, sample_stride=max(1, int(args.fusion_stride)))
    if not pairs:
        raise RuntimeError(f"No RGB-D frames found under {dataset}")
    K, _dist, _w, _h = _load_dataset_intrinsics(dataset)
    offsets, motion = _global_frame_offsets(dataset, pairs, K, args)
    axis = int(motion["axis"])

    cfg = CanopyReconstructionConfig(
        sample_stride=max(1, int(args.fusion_stride)),
        min_mask_area=max(1, int(args.component_min_area)),
        min_component_area=max(1, int(args.component_min_area)),
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        smooth_sigma=args.smooth_sigma,
        max_hole_fill_distance_px=args.max_hole_fill_px,
    )
    print(
        f"[global] {dataset.name}: {len(pairs)} frames, axis={axis}, "
        f"motion={motion.get('source')}, cell={args.grid_cell * 1000:.1f}mm"
    )
    min_x, max_x, min_y, max_y, frame_stats = _global_bounds(
        dataset, pairs, offsets, motion, K, cfg, args
    )
    cell = float(args.grid_cell)
    width = int(np.ceil((max_x - min_x) / cell)) + 1
    height = int(np.ceil((max_y - min_y) / cell)) + 1
    if width * height > int(args.max_grid_cells):
        raise RuntimeError(
            f"Global canopy grid would be {width}x{height}={width * height:,} cells. "
            "Increase --grid-cell or --max-grid-cells."
        )

    zbuf = np.full(width * height, np.inf, dtype=np.float32)
    color_flat = np.zeros((width * height, 3), dtype=np.uint8)
    vote_flat = np.zeros(width * height, dtype=np.uint16)
    integrated = 0
    skipped: list[dict] = []

    for i, (token, rgb_path, depth_path) in enumerate(pairs):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            skipped.append({"token": int(token), "reason": "imread failed"})
            continue
        mask = _auto_leaf_mask(rgb, cfg)
        if int((mask > 0).sum()) < int(args.component_min_area):
            skipped.append({"token": int(token), "reason": "mask too small"})
            continue
        try:
            x, y, z, stats = _project_valid_pixels(
                rgb, depth, mask, K, offsets[int(token)], axis, args.pixel_step, cfg
            )
        except RuntimeError as exc:
            skipped.append({"token": int(token), "reason": str(exc)})
            continue
        if x.size == 0:
            skipped.append({"token": int(token), "reason": "no valid plant depth"})
            continue
        colors = stats["colors"]
        ix = np.floor((x - min_x) / cell).astype(np.int64)
        iy = np.floor((y - min_y) / cell).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        if not np.any(in_bounds):
            skipped.append({"token": int(token), "reason": "outside grid"})
            continue
        keys = iy[in_bounds] * width + ix[in_bounds]
        z = z[in_bounds]
        colors = colors[in_bounds]

        order = np.lexsort((z, keys))
        keys_sorted = keys[order]
        first = np.r_[True, keys_sorted[1:] != keys_sorted[:-1]]
        best_idx = order[first]
        best_keys = keys[best_idx]
        best_z = z[best_idx]
        best_colors = colors[best_idx]

        replace = best_z < (zbuf[best_keys] - float(args.z_epsilon))
        near = np.abs(best_z - zbuf[best_keys]) <= float(args.z_epsilon)
        if np.any(replace):
            rk = best_keys[replace]
            zbuf[rk] = best_z[replace]
            color_flat[rk] = best_colors[replace]
            vote_flat[rk] = 1
        if np.any(near):
            nk = best_keys[near]
            old_votes = np.maximum(vote_flat[nk].astype(np.uint16), 1)
            blended = (
                color_flat[nk].astype(np.uint16) * old_votes[:, None]
                + best_colors[near].astype(np.uint16)
            ) // (old_votes[:, None] + 1)
            color_flat[nk] = np.clip(blended, 0, 255).astype(np.uint8)
            vote_flat[nk] = np.minimum(old_votes + 1, np.iinfo(np.uint16).max)
        integrated += 1
        if i % 25 == 0 or i == len(pairs) - 1:
            print(f"[global] fuse {i + 1:4d}/{len(pairs)}")

    depth_grid = zbuf.reshape(height, width)
    color_grid = color_flat.reshape(height, width, 3)
    valid_grid = np.isfinite(depth_grid)
    filled_depth, filled_color, support = _fill_global_grid(depth_grid, color_grid, valid_grid, args)
    pcd, mesh, height_map = _mesh_from_global_grid(
        filled_depth, filled_color, support, min_x, min_y, cell, args
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _save_global_grid_images(output, filled_depth, filled_color, support, height_map)
    np.save(output / "global_depth_m.npy", filled_depth)
    o3d.io.write_point_cloud(str(output / "sequence_points.ply"), pcd)
    o3d.io.write_triangle_mesh(str(output / "sequence_mesh.ply"), mesh)
    viewer_path = output / "sequence_viewer.html"
    write_canopy_mesh_viewer(
        mesh,
        viewer_path,
        title=f"{dataset.name} global canopy sequence",
        point_cloud=pcd,
        metadata={
            "Model": "global metric height-field",
            "Frames": f"{integrated}/{len(pairs)}",
            "Motion": motion.get("source", "unknown"),
            "Step": f"{float(motion.get('step_m_per_frame', 0.0)) * 1000:.2f} mm/frame",
            "Grid": f"{cell * 1000:.1f} mm",
            "Note": "Top-visible surface only; hidden undersides require angled/multiview capture.",
        },
    )

    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), 8
    )
    component_rows: list[dict] = []
    plant_idx = 1
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(args.global_component_min_cells):
            continue
        name = f"plant_{plant_idx:02d}"
        plant_dir = output / name
        plant_dir.mkdir(parents=True, exist_ok=True)
        comp_pcd, comp_mesh = _submesh_from_component(
            filled_depth, filled_color, labels, label, min_x, min_y, cell, args
        )
        comp_mesh_path = plant_dir / "canopy_mesh.ply"
        comp_pcd_path = plant_dir / "canopy_points.ply"
        comp_viewer_path = plant_dir / "canopy_viewer.html"
        o3d.io.write_point_cloud(str(comp_pcd_path), comp_pcd)
        o3d.io.write_triangle_mesh(str(comp_mesh_path), comp_mesh)
        write_canopy_mesh_viewer(
            comp_mesh,
            comp_viewer_path,
            title=f"{dataset.name} {name}",
            point_cloud=comp_pcd,
            metadata={
                "Model": "global component metric mesh",
                "Area cells": area,
                "Grid": f"{cell * 1000:.1f} mm",
            },
        )
        component_rows.append({
            "name": name,
            "label": int(label),
            "area_cells": area,
            "centroid_grid_xy": [float(centroids[label][0]), float(centroids[label][1])],
            "mesh_path": str(comp_mesh_path),
            "point_cloud_path": str(comp_pcd_path),
            "viewer_path": str(comp_viewer_path),
            "points": len(comp_pcd.points),
            "triangles": len(comp_mesh.triangles),
        })
        plant_idx += 1

    index = _write_global_index(output, component_rows, viewer_path)
    summary = {
        "dataset": str(dataset),
        "output": str(output),
        "mode": "global_orthographic_canopy_sequence",
        "motion": motion,
        "frames_available": len(pairs),
        "frames_integrated": integrated,
        "frames_skipped": len(skipped),
        "skipped_samples": skipped[:60],
        "bounds_m": [float(min_x), float(max_x), float(min_y), float(max_y)],
        "grid_width": int(width),
        "grid_height": int(height),
        "grid_cell_m": cell,
        "pixel_step": int(args.pixel_step),
        "points": len(pcd.points),
        "triangles": len(mesh.triangles),
        "plants": component_rows,
        "frame_stats_sample": frame_stats[:60],
        "viewer_path": str(viewer_path),
        "index_path": str(index),
    }
    (output / "sequence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[global] plants: {len(component_rows)}")
    print(f"[global] viewer: {viewer_path}")


def _write_index(output: Path, rows: list[dict], combined_viewer: Path | None) -> Path:
    links = []
    if combined_viewer is not None:
        rel = os.path.relpath(combined_viewer, start=output)
        links.append(f'<p><a href="{html.escape(rel)}">Open combined sequence viewer</a></p>')
    for row in rows:
        viewer = os.path.relpath(row["viewer_path"], start=output)
        summary = os.path.relpath(row["summary_path"], start=output)
        links.append(
            "<tr>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{row['reference_token']}</td>"
            f"<td>{row['position_m']:.4f}</td>"
            f"<td>{row['frames_used']}/{row['frames_available']}</td>"
            f"<td>{row['points']:,}</td>"
            f"<td>{row['triangles']:,}</td>"
            f'<td><a href="{html.escape(viewer)}">viewer</a></td>'
            f'<td><a href="{html.escape(summary)}">summary</a></td>'
            "</tr>"
        )

    table = (
        "<table><thead><tr><th>Plant</th><th>Reference</th><th>Position m</th>"
        "<th>Frames</th><th>Points</th><th>Triangles</th><th>Viewer</th><th>Summary</th>"
        "</tr></thead><tbody>"
        + "".join(links[1:] if combined_viewer is not None else links)
        + "</tbody></table>"
    )
    top = links[0] if combined_viewer is not None else ""
    css = (
        "body{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;"
        "padding:18px}a{color:#93c5fd}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #374151;padding:7px 9px;text-align:left}"
        "th{background:#1f2937}"
    )
    doc = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>Canopy sequence</title>"
        f"<style>{css}</style></head><body><h1>Canopy Sequence Reconstruction</h1>"
        f"{top}{table}</body></html>"
    )
    index = output / "sequence_index.html"
    index.write_text(doc, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct multiple plant windows from one scan.")
    parser.add_argument("--input", required=True, help="Dataset root.")
    parser.add_argument("--output", default=None, help="Output folder.")
    parser.add_argument("--global-ortho", action="store_true",
                        help="Fuse all plant pixels into one global gantry-coordinate height map.")
    parser.add_argument("--motion-source", choices=["auto", "phase", "token"], default="auto",
                        help="Motion source for --global-ortho. auto prefers phase-calibrated motion.")
    parser.add_argument("--gantry-axis", type=int, default=1,
                        help="Fallback gantry axis for token/config motion: 0=X, 1=Y.")
    parser.add_argument("--gantry-sign", type=float, default=1.0,
                        help="Fallback gantry sign for token/config motion.")
    parser.add_argument("--grid-cell", type=float, default=0.0025,
                        help="Global orthographic grid cell size in metres.")
    parser.add_argument("--pixel-step", type=int, default=1,
                        help="Use every Nth pixel during global orthographic fusion.")
    parser.add_argument("--z-epsilon", type=float, default=0.006,
                        help="Depth tolerance for colour averaging in one global grid cell.")
    parser.add_argument("--max-grid-cells", type=int, default=12_000_000)
    parser.add_argument("--global-max-hole-area", type=int, default=1800,
                        help="Largest enclosed grid-cell hole to fill in global mode.")
    parser.add_argument("--global-component-min-cells", type=int, default=1200,
                        help="Minimum connected global component size exported as a plant.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fusion-stride", type=int, default=1,
                        help="Frame stride inside each canopy fusion. Keep 1 for final quality.")
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--reference-spacing-m", type=float, default=0.08)
    parser.add_argument("--min-score-ratio", type=float, default=0.35)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--min-mask-area", type=int, default=180000)
    parser.add_argument("--coverage", type=int, default=1)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--depth-min", type=int, default=500)
    parser.add_argument("--depth-max", type=int, default=4000)
    parser.add_argument("--leaf-thickness", type=float, default=0.0)
    parser.add_argument("--max-hole-fill-px", type=int, default=24)
    parser.add_argument("--max-triangle-jump", type=float, default=0.025)
    parser.add_argument("--sequence-layout-margin", type=float, default=0.08,
                        help="Minimum XY gap in metres between instances in the combined viewer.")
    parser.add_argument("--canopy-sheet", action="store_true",
                        help="Use smoothed top-textured display sheets for sequence viewers.")
    parser.add_argument("--sheet-relief", type=float, default=0.025,
                        help="Maximum display-only canopy-sheet height relief in metres.")
    parser.add_argument("--sheet-smooth-sigma", type=float, default=4.0,
                        help="Display-only canopy-sheet depth smoothing sigma.")
    parser.add_argument("--sheet-pixel-step", type=int, default=2,
                        help="Use every Nth pixel for canopy-sheet display meshes.")
    parser.add_argument("--component-instances", action="store_true",
                        help="Split green masks into tracked plant instances before canopy fusion.")
    parser.add_argument("--component-min-area", type=int, default=8_000)
    parser.add_argument("--component-edge-penalty", type=float, default=0.20)
    parser.add_argument("--track-max-step-px", type=float, default=55.0)
    parser.add_argument("--track-max-area-ratio", type=float, default=4.0)
    parser.add_argument("--track-overlap-ratio", type=float, default=0.55)
    args = parser.parse_args()

    dataset = Path(args.input).resolve()
    if not dataset.exists():
        raise SystemExit(f"Input does not exist: {dataset}")
    args.output = Path(args.output).resolve() if args.output else dataset / "canopy_sequence"
    args.output.mkdir(parents=True, exist_ok=True)

    if args.global_ortho:
        _run_global_ortho_sequence(dataset, args)
        return

    if args.component_instances:
        references, motion_info = _component_reference_candidates(dataset, args)
    else:
        references, motion_info = _choose_references(dataset, args)
    if not references:
        raise SystemExit("No plant reference windows met the spacing/score thresholds.")
    print(f"[sequence] selected {len(references)} reference windows")

    rows = []
    combined_pcd = o3d.geometry.PointCloud()
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_display = o3d.geometry.TriangleMesh()
    reference_world_xy = [_reference_world_xy(ref, dataset) for ref in references]
    instance_geoms: list[dict] = []

    for idx, ref in enumerate(references, start=1):
        token = int(ref["token"])
        name = f"plant_{idx:02d}_token_{token}"
        out_dir = args.output / name
        instance_mask_dir = None
        tracked_mask_count = 0
        if args.component_instances:
            instance_mask_dir, tracked_mask_count = _write_tracked_instance_masks(
                dataset, out_dir, ref, args
            )
        edge_clipped_instance = bool(ref.get("edge", False))
        instance_max_frames = (
            int(args.max_frames)
            if edge_clipped_instance
            else min(int(args.max_frames), 11)
        )
        instance_hole_fill_px = (
            int(args.max_hole_fill_px)
            if edge_clipped_instance
            else min(int(args.max_hole_fill_px), 8)
        )
        cfg = CanopyReconstructionConfig(
            output_dir=str(out_dir),
            mask_dir=str(instance_mask_dir) if instance_mask_dir else None,
            sample_stride=args.fusion_stride,
            max_frames=instance_max_frames,
            max_candidates=0,
            min_mask_area=args.component_min_area if args.component_instances else args.min_mask_area,
            min_component_area=args.component_min_area if args.component_instances else 12_000,
            reference_token=token,
            coverage_threshold=args.coverage,
            smooth_sigma=args.smooth_sigma,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            add_leaf_thickness=args.leaf_thickness > 0,
            leaf_thickness_m=args.leaf_thickness if args.leaf_thickness > 0 else 0.003,
            max_hole_fill_distance_px=instance_hole_fill_px,
            max_triangle_height_jump_m=args.max_triangle_jump,
            spread_reference_frames=bool(args.component_instances and edge_clipped_instance),
            display_as_canopy_sheet=bool(args.canopy_sheet),
            display_sheet_relief_m=args.sheet_relief,
            display_sheet_smooth_sigma=args.sheet_smooth_sigma,
            display_sheet_pixel_step=args.sheet_pixel_step,
        )
        print(f"[sequence] {name}")
        result = reconstruct_canopy(dataset, config=cfg)
        world_x, world_y = reference_world_xy[idx - 1]
        pcd_geom = _read_geometry(result.point_cloud_path, "pcd")
        mesh_geom = _read_geometry(result.mesh_path, "mesh")
        display_path = out_dir / "canopy_display_mesh.ply"
        display_geom = o3d.geometry.TriangleMesh()
        if display_path.exists():
            display_geom = _read_geometry(str(display_path), "mesh")
        instance_geoms.append({
            "pcd": pcd_geom,
            "mesh": mesh_geom,
            "display": display_geom,
            "world_xy": (float(world_x), float(world_y)),
        })
        rows.append({
            "name": name,
            "reference_token": token,
            "position_m": float(ref["position_m"]),
            "viewer_path": result.viewer_path,
            "summary_path": result.summary_path,
            "frames_used": result.frames_used,
            "frames_available": result.frames_available,
            "points": result.final_point_count,
            "triangles": result.final_triangle_count,
            "tracked_masks": tracked_mask_count,
            "world_xy_m": [float(world_x), float(world_y)],
        })

    layout_offsets = _non_overlapping_y_offsets(
        [
            _geometry_xy_bounds(item["display"] if not item["display"].is_empty() else item["mesh"])
            for item in instance_geoms
        ],
        [item["world_xy"] for item in instance_geoms],
        args.sequence_layout_margin,
    )
    for item, row, (dx, dy) in zip(instance_geoms, rows, layout_offsets):
        combined_pcd += _translate_xy(item["pcd"], dx, dy)
        combined_mesh += _translate_xy(item["mesh"], dx, dy)
        if not item["display"].is_empty():
            combined_display += _translate_xy(item["display"], dx, dy)
        row["layout_xy_m"] = [float(dx), float(dy)]

    combined_viewer = None
    if not combined_pcd.is_empty():
        o3d.io.write_point_cloud(str(args.output / "sequence_points.ply"), combined_pcd)
    if not combined_mesh.is_empty():
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.remove_degenerate_triangles()
        combined_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(args.output / "sequence_mesh.ply"), combined_mesh)
    viewer_mesh = combined_display if args.leaf_thickness > 0 else combined_mesh
    if not viewer_mesh.is_empty():
        viewer_mesh.remove_duplicated_vertices()
        viewer_mesh.remove_degenerate_triangles()
        viewer_mesh.compute_vertex_normals()
        display_path = args.output / ("sequence_display_mesh.ply" if args.leaf_thickness > 0 else "sequence_mesh_view.ply")
        viewer_path = args.output / "sequence_viewer.html"
        o3d.io.write_triangle_mesh(str(display_path), viewer_mesh)
        write_canopy_mesh_viewer(
            viewer_mesh,
            viewer_path,
            title=f"{dataset.name} canopy sequence",
            point_cloud=combined_pcd,
            metadata={
                "Plants": len(rows),
                "Motion": motion_info.get("source", "unknown"),
                "Note": "Combined layout preserves instance order and prevents mesh overlap.",
            },
        )
        combined_viewer = viewer_path

    summary = {
        "dataset": str(dataset),
        "output": str(args.output),
        "motion": motion_info,
        "references": rows,
        "combined_viewer": str(combined_viewer) if combined_viewer else "",
    }
    (args.output / "sequence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    index = _write_index(args.output, rows, combined_viewer)
    print(f"[sequence] index: {index}")


if __name__ == "__main__":
    main()
