"""
Canopy (top-down) 3-D reconstruction for overhead gantry plant scanning.

Algorithm
---------
1. Auto-detect plant pixels in each sampled frame using HSV + excess-green masking.
2. Align frames to a chosen reference via phase-correlation on the plant crop.
3. Warp all depth maps onto a shared canvas and fuse them with per-pixel median.
4. Fill holes with nearest-neighbour propagation + Gaussian smoothing.
5. Un-project the fused depth canvas to a 3-D point cloud and Poisson mesh.

Compared to ICP/TSDF this approach is:
* More robust for top-down gantry scans (no convergence dependency).
* Much faster when depth data is clean (no iterative registration).
* Naturally handles the wide baseline between widely-spaced frames.

Layout support
--------------
Both flat-layout datasets (``rgb_N.png`` / ``depth_N.png`` siblings) *and*
ICL-style datasets (``rgb/0.png`` / ``depth/0.png`` sub-directories) are
supported automatically.

Ported from the ``plant_construction_0504`` branch and extended with:
* ``sample_stride`` — process only every Nth frame (default 10) so that
  large 600-frame datasets complete in reasonable time.
* ``max_candidates`` — hard cap on candidates forwarded to alignment.
* Dual-layout detection via ``_discover_image_pairs``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, replace, field
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, gaussian_filter

from file_io.loader import load_intrinsics
from visualiser.viewer import write_point_cloud_viewer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass
class CanopyReconstructionConfig:
    mask_dir: str | None = None
    """External mask directory.  Auto-masking used when None."""

    max_frames: int = 9
    """Maximum frames to include in the fusion (best-score candidates)."""

    sample_stride: int = 10
    """Evaluate only every Nth original frame during candidate search.
    Increase for large (>200-frame) datasets to keep runtime practical."""

    max_candidates: int = 40
    """Hard cap on the candidate pool before final selection."""

    min_mask_area: int = 180_000
    """Minimum plant-mask area (pixels) to accept a frame."""

    reference_token: int | None = None
    """Frame token to use as the alignment reference.  Auto-selected if None."""

    coverage_threshold: int = 1
    """Minimum number of frames that must cover each canvas pixel."""

    depth_min: int | None = 500
    """Near clip for raw depth (mm)."""

    depth_max: int | None = 4_000
    """Far clip for raw depth (mm)."""

    min_valid_depth_points: int = 5_000
    smooth_sigma: float = 3.5
    z_scale: float = 1.0
    output_dir: str | None = None
    auto_mask: bool = True
    min_component_area: int = 12_000
    mask_hue_min: int = 28
    mask_hue_max: int = 95
    mask_s_min: int = 45
    mask_v_min: int = 35
    mask_exg_min: int = 20
    canvas_padding: int = 48
    crop_to_mask: bool = True

    # Post-fusion cleanup
    mesh_cleanup: bool = True
    """Remove sparse outliers before meshing and trim low-density Poisson faces."""

    mesh_density_quantile: float = 0.01
    """Poisson mesh faces below this density quantile are removed (0.01 = bottom 1%)."""

    nb_neighbors: int = 30
    """Neighbours used for statistical outlier removal."""

    outlier_std_ratio: float = 2.0
    """Points further than mean+std_ratio*sigma from their neighbours are dropped."""

    # Leaf thickness / back-face geometry
    add_leaf_thickness: bool = False
    """Duplicate top-surface points offset downward to simulate leaf thickness.
    Improves side-view appearance; no extra data capture required."""

    leaf_thickness_m: float = 0.003
    """Vertical offset (metres) for the duplicated back-face layer."""


@dataclass
class CanopyReconstructionResult:
    record_path: str
    output_dir: str
    point_cloud_path: str
    mesh_path: str
    viewer_path: str
    masked_rgb_path: str
    summary_path: str
    frames_available: int
    frames_used: int
    reference_token: int
    final_point_count: int
    final_triangle_count: int


# ---------------------------------------------------------------------------
# Layout helpers — support flat (rgb_N.png) and ICL-style (rgb/N.png)
# ---------------------------------------------------------------------------

def _discover_image_pairs(
    record_path: Path,
    sample_stride: int = 1,
) -> list[tuple[int, Path, Path]]:
    """Return ``[(token, rgb_path, depth_path), ...]`` sorted by token.

    Supports two layouts:
    * **Flat**: ``rgb_N.png`` and ``depth_N.png`` directly inside *record_path*.
    * **ICL-style**: ``rgb/N.png`` and ``depth/N.png`` sub-directories.
    """
    pairs: list[tuple[int, Path, Path]] = []

    # --- Flat layout ---
    flat_rgb = sorted(record_path.glob("rgb_*.png"), key=lambda p: int(p.stem.split("_", 1)[1]))
    if flat_rgb:
        for i, rgb_path in enumerate(flat_rgb):
            if i % sample_stride != 0:
                continue
            token = int(rgb_path.stem.split("_", 1)[1])
            depth_path = record_path / f"depth_{token}.png"
            if depth_path.exists():
                pairs.append((token, rgb_path, depth_path))
        return pairs

    # --- ICL-style layout ---
    rgb_dir   = record_path / "rgb"
    depth_dir = record_path / "depth"
    if rgb_dir.is_dir() and depth_dir.is_dir():
        all_rgb = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))
        for i, rgb_path in enumerate(all_rgb):
            if i % sample_stride != 0:
                continue
            token = int(rgb_path.stem)
            depth_path = depth_dir / f"{token}.png"
            if depth_path.exists():
                pairs.append((token, rgb_path, depth_path))

    return pairs


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _keep_largest(mask_u8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return mask_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)
    out[labels == largest] = 255
    return out


def _drop_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return mask_u8
    out = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 255
    return out


def _fill_small_holes(mask_u8: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return mask_u8
    binary = mask_u8 > 0
    inv = (~binary).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    if num_labels <= 1:
        return mask_u8
    out = mask_u8.copy()
    height, width = mask_u8.shape
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and int(area) <= max_hole_area:
            out[labels == label] = 255
    return out


def _clean_mask(
    mask_u8: np.ndarray,
    *,
    keep_largest: bool = True,
    min_component_area: int = 0,
) -> np.ndarray:
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    if keep_largest:
        mask_u8 = _keep_largest(mask_u8)
    elif min_component_area > 0:
        mask_u8 = _drop_small_components(mask_u8, min_component_area)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask_u8 = _fill_small_holes(mask_u8, max(12_000, min_component_area))

    if keep_largest:
        return _keep_largest(mask_u8)
    if min_component_area > 0:
        return _drop_small_components(mask_u8, min_component_area)
    return mask_u8


def _auto_leaf_mask(rgb_bgr: np.ndarray, cfg: CanopyReconstructionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(rgb_bgr)
    excess_green = 2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)
    raw = (
        (h >= cfg.mask_hue_min)
        & (h <= cfg.mask_hue_max)
        & (s >= cfg.mask_s_min)
        & (v >= cfg.mask_v_min)
        & (excess_green >= cfg.mask_exg_min)
    ).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (raw > 0).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return raw

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max())
    if largest_area < cfg.min_component_area:
        return np.zeros_like(raw)

    keep_threshold = max(cfg.min_component_area, int(largest_area * 0.08))
    kept = np.zeros_like(raw)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= keep_threshold:
            kept[labels == label] = 255
    return _clean_mask(kept, keep_largest=False, min_component_area=cfg.min_component_area)


def _mask_stats(mask_u8: np.ndarray, min_component_area: int) -> dict:
    mask = mask_u8 > 0
    area = int(mask.sum())
    if area == 0:
        return {"mask_area": 0, "bbox": [0, 0, 0, 0], "edge_fraction": 1.0, "score": 0.0}

    height, width = mask.shape
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    border = max(8, min(height, width) // 40)
    edge_pixels = (
        int(mask[:border, :].sum())
        + int(mask[-border:, :].sum())
        + int(mask[:, :border].sum())
        + int(mask[:, -border:].sum())
    )
    edge_fraction = edge_pixels / max(area, 1)
    score = float(area * (1.0 - min(edge_fraction * 4.0, 0.65)))

    if min_component_area > 0:
        cleaned = _drop_small_components(mask_u8, min_component_area)
        area = int((cleaned > 0).sum())

    return {
        "mask_area": area,
        "bbox": [int(x), int(y), int(w), int(h)],
        "edge_fraction": float(edge_fraction),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Candidate loading (both layouts, with stride + cap)
# ---------------------------------------------------------------------------

def _load_auto_candidates(
    pairs: list[tuple[int, Path, Path]],
    auto_mask_dir: Path,
    cfg: CanopyReconstructionConfig,
) -> list[dict]:
    """Run auto-masking on *pairs* and return candidates sorted by score.

    *pairs* is already strided — every frame in the list is evaluated.
    """
    auto_mask_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []

    preview_scale = 0.25
    preview_cfg = replace(
        cfg,
        min_component_area=max(16, int(cfg.min_component_area * preview_scale ** 2)),
    )

    for token, rgb_path, depth_path in pairs:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue

        # Fast downscale pre-filter
        preview = cv2.resize(
            rgb, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA
        )
        preview_mask = _auto_leaf_mask(preview, preview_cfg)
        preview_stats = _mask_stats(preview_mask, preview_cfg.min_component_area)
        estimated_full_area = preview_stats["mask_area"] / (preview_scale ** 2)
        if estimated_full_area < cfg.min_mask_area * 0.7:
            continue

        # Full-res mask
        mask = _auto_leaf_mask(rgb, cfg)
        stats = _mask_stats(mask, cfg.min_component_area)
        if stats["mask_area"] < cfg.min_mask_area:
            continue

        mask_path = auto_mask_dir / f"mask_{token}.png"
        cv2.imwrite(str(mask_path), mask)
        candidates.append({
            "token": token,
            "mask_path": mask_path,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "mask_area": stats["mask_area"],
            "bbox": stats["bbox"],
            "edge_fraction": stats["edge_fraction"],
            "score": stats["score"],
        })

        if len(candidates) >= cfg.max_candidates:
            print(f"[canopy] Reached max_candidates={cfg.max_candidates}, stopping search.")
            break

    return candidates


def _load_mask_candidates(
    record_path: Path,
    mask_dir: Path,
    pairs_by_token: dict[int, tuple[Path, Path]],
    cfg: CanopyReconstructionConfig,
) -> list[dict]:
    """Load candidates from an external mask directory."""
    candidates: list[dict] = []
    for mask_path in sorted(mask_dir.glob("mask_*.png"),
                            key=lambda p: int(p.stem.split("_", 1)[1])):
        token = int(mask_path.stem.split("_", 1)[1])
        if token not in pairs_by_token:
            continue
        rgb_path, depth_path = pairs_by_token[token]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        clean_mask = _clean_mask(
            mask, keep_largest=False, min_component_area=cfg.min_component_area
        )
        stats = _mask_stats(clean_mask, cfg.min_component_area)
        if stats["mask_area"] < cfg.min_mask_area:
            continue
        candidates.append({
            "token": token,
            "mask_path": mask_path,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "mask_area": stats["mask_area"],
            "bbox": stats["bbox"],
            "edge_fraction": stats["edge_fraction"],
            "score": stats["score"],
        })
    return candidates


# ---------------------------------------------------------------------------
# Frame selection and alignment
# ---------------------------------------------------------------------------

def _select_frames(
    candidates: list[dict], cfg: CanopyReconstructionConfig
) -> tuple[list[dict], int]:
    if not candidates:
        raise RuntimeError("No valid plant frames found.  Check mask/auto-mask settings.")

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    selected = candidates[: cfg.max_frames]

    if cfg.reference_token is not None:
        ref = cfg.reference_token
        if not any(c["token"] == ref for c in selected):
            match = next((c for c in candidates if c["token"] == ref), None)
            if match:
                selected[-1] = match
    else:
        ref = selected[0]["token"]

    selected = sorted(selected, key=lambda c: c["token"])
    return selected, ref


def _estimate_alignment_transforms(
    selected: list[dict],
    reference_token: int,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[float, float]], tuple[int, int]]:
    ref_item = next(c for c in selected if c["token"] == reference_token)
    ref_rgb = cv2.imread(str(ref_item["rgb_path"]))
    if ref_rgb is None:
        raise FileNotFoundError(f"Cannot read reference RGB: {ref_item['rgb_path']}")
    ref_mask = cv2.imread(str(ref_item["mask_path"]), cv2.IMREAD_GRAYSCALE)
    image_shape = ref_rgb.shape[:2]

    x, y, w, h = cv2.boundingRect((ref_mask > 0).astype(np.uint8))
    pad = max(48, int(min(image_shape) * 0.12))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image_shape[1], x + w + pad), min(image_shape[0], y + h + pad)
    ref_gray = cv2.cvtColor(ref_rgb[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)

    transforms: dict[int, np.ndarray] = {}
    predicted_centers: dict[int, tuple[float, float]] = {}

    for item in selected:
        token = item["token"]
        if token == reference_token:
            transforms[token] = np.float32([[1, 0, 0], [0, 1, 0]])
            predicted_centers[token] = (float(x + w / 2), float(y + h / 2))
            continue

        src_rgb = cv2.imread(str(item["rgb_path"]))
        if src_rgb is None:
            transforms[token] = np.float32([[1, 0, 0], [0, 1, 0]])
            predicted_centers[token] = (float(x + w / 2), float(y + h / 2))
            continue

        src_crop = src_rgb[y1:y2, x1:x2]
        src_gray = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Phase-correlation shift estimate
        shift, _ = cv2.phaseCorrelate(src_gray, ref_gray)
        dx, dy = float(shift[0]), float(shift[1])

        transforms[token] = np.float32([[1, 0, dx], [0, 1, dy]])
        predicted_centers[token] = (float(x + w / 2 - dx), float(y + h / 2 - dy))

    return transforms, predicted_centers, image_shape


# ---------------------------------------------------------------------------
# Canvas geometry
# ---------------------------------------------------------------------------

def _build_canvas_transforms(
    transforms: dict[int, np.ndarray],
    image_shape: tuple[int, int],
    padding: int = 48,
) -> tuple[dict[int, np.ndarray], tuple[int, int], tuple[float, float]]:
    height, width = image_shape
    all_shifts = [t[:, 2] for t in transforms.values()]
    min_dx = min(s[0] for s in all_shifts)
    min_dy = min(s[1] for s in all_shifts)
    max_dx = max(s[0] for s in all_shifts)
    max_dy = max(s[1] for s in all_shifts)

    offset_x = -min(0.0, min_dx) + padding
    offset_y = -min(0.0, min_dy) + padding
    canvas_w  = int(width  + abs(max_dx - min_dx) + 2 * padding)
    canvas_h  = int(height + abs(max_dy - min_dy) + 2 * padding)

    canvas_transforms: dict[int, np.ndarray] = {}
    for token, t in transforms.items():
        ct = t.copy()
        ct[0, 2] += offset_x
        ct[1, 2] += offset_y
        canvas_transforms[token] = ct

    return canvas_transforms, (canvas_w, canvas_h), (offset_x, offset_y)


# ---------------------------------------------------------------------------
# Depth filling and smoothing
# ---------------------------------------------------------------------------

def _smooth_in_mask(
    values: np.ndarray, mask: np.ndarray, sigma: float
) -> np.ndarray:
    weight = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    smooth = gaussian_filter(
        values.astype(np.float32) * mask.astype(np.float32), sigma=sigma
    )
    out = np.zeros_like(values, dtype=np.float32)
    np.divide(smooth, weight, out=out, where=weight > 1e-6)
    return out


def _nearest_fill_in_mask(
    values: np.ndarray, valid: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    if not np.any(valid):
        raise RuntimeError("No usable depth values inside the canopy mask.")
    canvas = np.zeros(values.shape, dtype=np.float32)
    canvas[valid] = values[valid].astype(np.float32)
    missing = mask & ~valid
    if np.any(missing):
        _, nearest_indices = distance_transform_edt(~valid, return_indices=True)
        canvas[missing] = canvas[
            nearest_indices[0][missing], nearest_indices[1][missing]
        ]
    canvas[~mask] = 0.0
    return canvas


def _fill_depth_inside_mask(
    depth_u16: np.ndarray,
    mask_u8: np.ndarray,
    cfg: CanopyReconstructionConfig,
) -> tuple[np.ndarray, dict]:
    mask = mask_u8 > 0
    valid_values = depth_u16[mask & (depth_u16 > 0)]
    if cfg.depth_min is not None:
        valid_values = valid_values[valid_values >= cfg.depth_min]
    if cfg.depth_max is not None:
        valid_values = valid_values[valid_values <= cfg.depth_max]
    if valid_values.size < cfg.min_valid_depth_points:
        valid_values = depth_u16[mask & (depth_u16 > 0)]
    if valid_values.size == 0:
        raise RuntimeError("No usable depth values inside the canopy mask.")

    low  = float(np.percentile(valid_values, 1.0))
    high = float(np.percentile(valid_values, 98.0))
    valid = mask & (depth_u16 >= low) & (depth_u16 <= high)
    if not np.any(valid):
        raise RuntimeError("Depth filtering removed every point inside the canopy mask.")

    canvas = _nearest_fill_in_mask(depth_u16.astype(np.float32), valid, mask)
    canvas = _smooth_in_mask(canvas, mask, sigma=cfg.smooth_sigma)

    canopy = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if canopy.max() > 0:
        canvas -= 30.0 * (canopy / canopy.max()).astype(np.float32)
    canvas[~mask] = 0.0

    return canvas, {
        "mask_area": int(mask.sum()),
        "valid_depth_points": int(valid.sum()),
        "depth_range_mm": [low, high],
    }


# ---------------------------------------------------------------------------
# 3-D reconstruction from the fused canvas
# ---------------------------------------------------------------------------

def _crop_to_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    padding: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return depth, mask, rgb, K, [0, 0, depth.shape[1], depth.shape[0]]
    x1, y1 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
    x2 = min(depth.shape[1], int(xs.max()) + padding)
    y2 = min(depth.shape[0], int(ys.max()) + padding)
    K_out = K.copy()
    K_out[0, 2] -= x1
    K_out[1, 2] -= y1
    return (
        depth[y1:y2, x1:x2],
        mask[y1:y2, x1:x2],
        rgb[y1:y2, x1:x2],
        K_out,
        [x1, y1, x2 - x1, y2 - y1],
    )


def _add_leaf_thickness_points(
    pcd: o3d.geometry.PointCloud,
    thickness_m: float,
) -> o3d.geometry.PointCloud:
    """Return a new point cloud that includes the original top-surface points
    plus a duplicate layer offset by *thickness_m* along +Z (further from camera).

    This creates a visually "thick" leaf shell that improves side-view appearance
    without requiring additional capture angles.
    """
    pts  = np.asarray(pcd.points).copy()
    cols = np.asarray(pcd.colors).copy()
    back_pts = pts.copy()
    back_pts[:, 2] += thickness_m          # positive Z = further from camera
    thick_pcd = o3d.geometry.PointCloud()
    thick_pcd.points = o3d.utility.Vector3dVector(np.vstack([pts, back_pts]))
    thick_pcd.colors = o3d.utility.Vector3dVector(np.vstack([cols, cols]))
    return thick_pcd


def _build_mesh_and_point_cloud(
    fused_depth_mm: np.ndarray,
    fused_mask: np.ndarray,
    fused_rgb_bgr: np.ndarray,
    K: np.ndarray,
    z_scale: float = 1.0,
    do_cleanup: bool = True,
    nb_neighbors: int = 30,
    outlier_std_ratio: float = 2.0,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, "o3d.utility.DoubleVector", float]:
    """Un-project the fused depth canvas to a 3-D point cloud and Poisson mesh.

    Returns
    -------
    pcd, mesh, densities, reference_depth_m
        *densities* is the per-vertex density array from Poisson reconstruction —
        use it to trim low-confidence mesh faces with a quantile threshold.
    """
    fx, fy = float(K[0, 0]), float(abs(K[1, 1]))
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ys, xs = np.where(fused_mask & (fused_depth_mm > 0))
    if len(ys) == 0:
        raise RuntimeError("Fused depth map has no valid points.")

    zs = fused_depth_mm[ys, xs].astype(np.float64) / 1000.0 * z_scale
    Xs = (xs - cx) * zs / fx
    Ys = (ys - cy) * zs / fy

    rgb_img = cv2.cvtColor(fused_rgb_bgr, cv2.COLOR_BGR2RGB)
    Rs = rgb_img[ys, xs, 0].astype(np.float64) / 255.0
    Gs = rgb_img[ys, xs, 1].astype(np.float64) / 255.0
    Bs = rgb_img[ys, xs, 2].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([Xs, Ys, zs]))
    pcd.colors = o3d.utility.Vector3dVector(np.column_stack([Rs, Gs, Bs]))

    # Statistical outlier removal
    if do_cleanup and len(pcd.points) > nb_neighbors:
        clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=outlier_std_ratio
        )
        if not clean.is_empty():
            pcd = clean

    reference_depth_m = float(np.median(np.asarray(pcd.points)[:, 2]))

    densities = o3d.utility.DoubleVector()
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, linear_fit=True
        )
        mesh.compute_vertex_normals()
    except Exception as exc:
        print(f"[canopy] Mesh generation failed ({exc}); returning empty mesh.")
        mesh = o3d.geometry.TriangleMesh()

    return pcd, mesh, densities, reference_depth_m


# ---------------------------------------------------------------------------
# Oblique preview
# ---------------------------------------------------------------------------

def _save_previews(output_dir: Path, pcd: o3d.geometry.PointCloud) -> None:
    if pcd is None or pcd.is_empty():
        return
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        return

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(points) // 40_000)
    sel = slice(None, None, step)
    ax.scatter(
        points[sel, 0], points[sel, 1], points[sel, 2],
        s=1.0, c=colors[sel], depthshade=False,
    )
    ax.view_init(elev=30, azim=-55)
    ax.set_axis_off()
    mins, maxs = points.min(0), points.max(0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max() / 2.0)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(0.0, max(center[2] + span * 0.75, 0.1))
    fig.tight_layout()
    fig.savefig(output_dir / "canopy_oblique.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconstruct_canopy(
    record_path: str | Path,
    config: CanopyReconstructionConfig | None = None,
) -> CanopyReconstructionResult:
    """Run the canopy reconstruction pipeline on *record_path*.

    Parameters
    ----------
    record_path:
        Root of the dataset.  May contain ``rgb_*.png`` / ``depth_*.png``
        (flat layout) **or** ``rgb/`` / ``depth/`` sub-directories (ICL-style).
    config:
        Pipeline parameters.  Defaults to :class:`CanopyReconstructionConfig`.
    """
    cfg  = config or CanopyReconstructionConfig()
    root = Path(record_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    # Camera intrinsics
    intrinsics = load_intrinsics(str(root / "kdc_intrinsics.txt"))
    if intrinsics is None:
        intrinsics = load_intrinsics(str(root / "kd_intrinsics.txt"))
    if intrinsics is None:
        raise FileNotFoundError(
            "Canopy reconstruction requires kdc_intrinsics.txt or kd_intrinsics.txt."
        )
    K, _, _, _ = intrinsics

    output_dir = Path(cfg.output_dir) if cfg.output_dir else root / "canopy_local"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover image pairs (stride applied here)
    all_pairs = _discover_image_pairs(root, sample_stride=cfg.sample_stride)
    if not all_pairs:
        raise RuntimeError(
            f"No RGB-D frame pairs found under {root}.  "
            "Expected flat rgb_N.png/depth_N.png or rgb//N.png/depth//N.png layout."
        )
    print(
        f"[canopy] {root.name}: {len(all_pairs)} candidate frames "
        f"(stride={cfg.sample_stride})"
    )

    pairs_by_token: dict[int, tuple[Path, Path]] = {t: (r, d) for t, r, d in all_pairs}

    # Load candidates
    mask_dir = Path(cfg.mask_dir) if cfg.mask_dir else root / "reconstruction" / "masks"
    if mask_dir.exists():
        candidates = _load_mask_candidates(root, mask_dir, pairs_by_token, cfg)
        mask_source = str(mask_dir.resolve())
    elif cfg.auto_mask:
        auto_mask_dir = output_dir / "auto_masks"
        candidates = _load_auto_candidates(all_pairs, auto_mask_dir, cfg)
        mask_source = str(auto_mask_dir.resolve())
    else:
        raise FileNotFoundError(
            f"Mask directory does not exist and auto_mask is disabled: {mask_dir}"
        )

    print(f"[canopy] {len(candidates)} candidates pass the mask area threshold.")

    selected, reference_token = _select_frames(candidates, cfg)
    print(
        f"[canopy] Selected {len(selected)} frames for fusion "
        f"(reference token={reference_token})."
    )

    transforms, predicted_centers, image_shape = _estimate_alignment_transforms(
        selected, reference_token
    )
    canvas_transforms, canvas_size, canvas_offset = _build_canvas_transforms(
        transforms, image_shape, padding=max(0, int(cfg.canvas_padding))
    )
    K_canvas = np.asarray(K, dtype=np.float64).copy()
    K_canvas[0, 2] += canvas_offset[0]
    K_canvas[1, 2] += canvas_offset[1]

    warped_depths, warped_masks, warped_rgbs = [], [], []
    reference_warped_rgb = None
    reference_warped_mask = None
    frame_info: list[dict] = []

    for item in selected:
        token = int(item["token"])
        rgb   = cv2.imread(str(item["rgb_path"]))
        depth = cv2.imread(str(item["depth_path"]), cv2.IMREAD_UNCHANGED)
        mask  = cv2.imread(str(item["mask_path"]),  cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise FileNotFoundError(f"Failed to load frame set for token {token}")

        clean_mask  = _clean_mask(
            mask, keep_largest=False, min_component_area=cfg.min_component_area
        )
        filled_depth, depth_stats = _fill_depth_inside_mask(depth, clean_mask, cfg)
        warp = canvas_transforms[token]

        warped_depth = cv2.warpAffine(
            filled_depth, warp, canvas_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            clean_mask, warp, canvas_size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        warped_rgb = cv2.warpAffine(
            rgb.astype(np.float32), warp, canvas_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        warped_mask_bool = warped_mask > 0
        warped_depth[~warped_mask_bool] = 0.0
        warped_rgb[~warped_mask_bool]   = 0.0

        warped_depths.append(warped_depth)
        warped_masks.append(warped_mask_bool)
        warped_rgbs.append(warped_rgb)
        if token == reference_token:
            reference_warped_rgb  = warped_rgb.copy()
            reference_warped_mask = warped_mask_bool.copy()

        frame_info.append({
            "token": token,
            "mask_area": int(item["mask_area"]),
            "edge_fraction": float(item.get("edge_fraction", 0.0)),
            "alignment_shift_xy": [
                float(transforms[token][0, 2]), float(transforms[token][1, 2])
            ],
            "predicted_center_xy": [
                float(v) for v in predicted_centers.get(token, (0.0, 0.0))
            ],
            **depth_stats,
        })

    # Depth fusion (median)
    coverage = np.sum(np.stack(warped_masks, axis=0), axis=0)
    stacked  = np.stack(
        [np.where(m, d, np.nan) for d, m in zip(warped_depths, warped_masks)], axis=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused_depth = np.nanmedian(stacked, axis=0).astype(np.float32)

    fused_mask   = coverage >= max(1, int(cfg.coverage_threshold))
    fused_mask_u8 = _clean_mask(
        fused_mask.astype(np.uint8) * 255,
        keep_largest=False,
        min_component_area=cfg.min_component_area,
    )
    fused_mask = fused_mask_u8 > 0

    valid = fused_mask & np.isfinite(fused_depth) & (fused_depth > 0)
    if not np.any(valid):
        raise RuntimeError("Canopy fusion produced an empty depth map.")

    fused_depth_full = _nearest_fill_in_mask(
        np.nan_to_num(fused_depth, nan=0.0), valid, fused_mask
    )
    fused_depth_full = _smooth_in_mask(
        fused_depth_full, fused_mask, sigma=cfg.smooth_sigma + 1.0
    )
    canopy_dt = cv2.distanceTransform(fused_mask.astype(np.uint8), cv2.DIST_L2, 5)
    if canopy_dt.max() > 0:
        fused_depth_full -= 45.0 * (canopy_dt / canopy_dt.max()).astype(np.float32)
    fused_depth_full[~fused_mask] = 0.0

    # Colour fusion
    canvas_h, canvas_w = coverage.shape
    color_sum  = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    for warped_rgb, warped_mask_bool in zip(warped_rgbs, warped_masks):
        color_sum += warped_rgb * warped_mask_bool[..., None].astype(np.float32)
    fused_rgb = np.zeros_like(color_sum, dtype=np.float32)
    np.divide(
        color_sum, np.maximum(coverage[..., None], 1),
        out=fused_rgb, where=coverage[..., None] > 0,
    )
    if reference_warped_rgb is not None and reference_warped_mask is not None:
        ref_valid = reference_warped_mask & fused_mask
        fused_rgb[ref_valid] = reference_warped_rgb[ref_valid]
    fused_rgb = np.clip(fused_rgb, 0, 255).astype(np.uint8)
    fused_rgb[~fused_mask] = 0

    # Optional crop
    crop_box = [0, 0, int(canvas_w), int(canvas_h)]
    if cfg.crop_to_mask:
        fused_depth_full, fused_mask, fused_rgb, K_canvas, crop_box = _crop_to_mask(
            fused_depth_full, fused_mask, fused_rgb, K_canvas,
            padding=max(8, int(cfg.canvas_padding) // 2),
        )
        fused_mask_u8 = (fused_mask.astype(np.uint8) * 255)

    # Save 2-D outputs
    masked_rgb_path = output_dir / "fused_rgb_masked.png"
    cv2.imwrite(str(masked_rgb_path), fused_rgb)
    cv2.imwrite(str(output_dir / "fused_mask.png"), fused_mask_u8)
    np.save(output_dir / "fused_depth_mm.npy", fused_depth_full)

    # Depth preview
    preview_depth = np.zeros(fused_depth_full.shape, dtype=np.uint8)
    inside = fused_depth_full[fused_mask]
    low_p, high_p = np.percentile(inside, [2, 98])
    scaled = np.clip((fused_depth_full - low_p) / max(high_p - low_p, 1e-6), 0.0, 1.0)
    preview_depth[fused_mask] = (255.0 * (1.0 - scaled[fused_mask])).astype(np.uint8)
    cv2.imwrite(str(output_dir / "fused_depth_vis.png"), preview_depth)

    # 3-D reconstruction
    pcd, mesh, densities, reference_depth_m = _build_mesh_and_point_cloud(
        fused_depth_mm=fused_depth_full,
        fused_mask=fused_mask,
        fused_rgb_bgr=fused_rgb,
        K=K_canvas,
        z_scale=cfg.z_scale,
        do_cleanup=cfg.mesh_cleanup,
        nb_neighbors=cfg.nb_neighbors,
        outlier_std_ratio=cfg.outlier_std_ratio,
    )

    # Trim low-density Poisson artefacts (floaters around the mesh boundary)
    if cfg.mesh_cleanup and len(densities) > 0:
        density_arr = np.asarray(densities)
        density_thresh = np.quantile(density_arr, max(0.0, min(1.0, cfg.mesh_density_quantile)))
        mesh.remove_vertices_by_mask(density_arr < density_thresh)
        print(
            f"[canopy] Mesh after density trim: "
            f"{len(np.asarray(mesh.triangles)):,} triangles "
            f"(quantile={cfg.mesh_density_quantile:.3f}, "
            f"threshold={density_thresh:.4f})"
        )

    # Leaf thickness: duplicate top-surface layer offset in Z for better side views
    viewer_pcd = pcd
    if cfg.add_leaf_thickness and not pcd.is_empty():
        viewer_pcd = _add_leaf_thickness_points(pcd, cfg.leaf_thickness_m)
        thick_path = output_dir / "canopy_points_thick.ply"
        o3d.io.write_point_cloud(str(thick_path), viewer_pcd)
        print(
            f"[canopy] Leaf thickness layer added "
            f"({cfg.leaf_thickness_m * 1000:.1f} mm offset, "
            f"{len(np.asarray(viewer_pcd.points)):,} points total)."
        )

    point_cloud_path = output_dir / "canopy_points.ply"
    mesh_path        = output_dir / "canopy_mesh.ply"
    viewer_path      = output_dir / "canopy_viewer.html"
    o3d.io.write_point_cloud(str(point_cloud_path), pcd)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    write_point_cloud_viewer(viewer_pcd, viewer_path, title=f"{root.name} canopy")
    _save_previews(output_dir, viewer_pcd)

    summary = {
        "record_path": str(root.resolve()),
        "output_dir":  str(output_dir.resolve()),
        "point_cloud_path": str(point_cloud_path.resolve()),
        "mesh_path":   str(mesh_path.resolve()),
        "viewer_path": str(viewer_path.resolve()),
        "masked_rgb_path": str(masked_rgb_path.resolve()),
        "mask_source": mask_source,
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "canvas_offset_xy": [float(canvas_offset[0]), float(canvas_offset[1])],
        "crop_box_xywh": crop_box,
        "frames_available": len(all_pairs),
        "frames_used": len(selected),
        "reference_token": reference_token,
        "reference_depth_m": reference_depth_m,
        "final_point_count": len(np.asarray(pcd.points)),
        "viewer_point_count": len(np.asarray(viewer_pcd.points)),
        "final_triangle_count": len(np.asarray(mesh.triangles)),
        "config": asdict(cfg),
        "frames": frame_info,
    }
    summary_path = output_dir / "canopy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[canopy] Done.  {len(selected)} frames fused -> "
        f"{len(np.asarray(pcd.points)):,} points, "
        f"{len(np.asarray(mesh.triangles)):,} triangles."
    )

    return CanopyReconstructionResult(
        record_path=str(root.resolve()),
        output_dir=str(output_dir.resolve()),
        point_cloud_path=str(point_cloud_path.resolve()),
        mesh_path=str(mesh_path.resolve()),
        viewer_path=str(viewer_path.resolve()),
        masked_rgb_path=str(masked_rgb_path.resolve()),
        summary_path=str(summary_path.resolve()),
        frames_available=len(all_pairs),
        frames_used=len(selected),
        reference_token=reference_token,
        final_point_count=len(np.asarray(pcd.points)),
        final_triangle_count=len(np.asarray(mesh.triangles)),
    )
