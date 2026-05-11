"""
Canopy (top-down) 3-D reconstruction for overhead gantry plant scanning.

Algorithm
---------
1. Auto-detect plant pixels in each frame using HSV + excess-green masking.
2. Select a high-quality local frame window around a reference canopy view.
3. Align frames from gantry/session motion, with bounded image refinement.
4. Gate foreground depth before fusion so far background returns do not fill leaves.
5. Fuse valid depth on a shared canvas and build a metric height-field mesh.

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, gaussian_filter

from file_io.loader import load_gantry_config, load_intrinsics, load_session_json
from visualiser.viewer import write_canopy_mesh_viewer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass
class CanopyReconstructionConfig:
    mask_dir: str | None = None
    """External mask directory.  Auto-masking used when None."""

    max_frames: int = 15
    """Maximum frames to include in the fusion (best-score candidates)."""

    sample_stride: int = 1
    """Evaluate every Nth original frame during candidate search."""

    max_candidates: int = 0
    """Optional post-detection candidate shortlist size. 0 means no cap."""

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
    smooth_sigma: float = 2.0
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

    foreground_percentile: float = 90.0
    """Per-frame depth percentile used to reject far background leakage."""

    foreground_margin_mm: float = 80.0
    """Extra depth margin beyond the foreground percentile."""

    max_alignment_refine_px: float = 5.0
    """Maximum accepted phase-correlation refinement after gantry alignment."""

    min_alignment_response: float = 0.08
    """Minimum phase-correlation response for accepting an image refinement."""

    max_triangle_height_jump_m: float = 0.025
    """Do not connect neighbouring height-field vertices across larger jumps."""

    max_hole_fill_distance_px: int = 24
    """Only inpaint RGB-supported plant holes this many pixels from real depth."""

    use_poisson_mesh: bool = False
    """Experimental: run Poisson reconstruction instead of height-field meshing."""

    # Post-fusion cleanup
    mesh_cleanup: bool = True
    """Remove sparse outliers before meshing and trim low-density Poisson faces."""

    mesh_density_quantile: float = 0.01
    """Poisson mesh faces below this density quantile are removed (0.01 = bottom 1%)."""

    nb_neighbors: int = 30
    """Neighbours used for statistical outlier removal."""

    outlier_std_ratio: float = 2.0
    """Points further than mean+std_ratio*sigma from their neighbours are dropped."""

    # Display-only leaf thickness / back-face geometry
    add_leaf_thickness: bool = True
    """Add a thin display-only back-face/skirt mesh for side-view readability."""

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

def _mask_centroid(mask_u8: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments((mask_u8 > 0).astype(np.uint8))
    if moments["m00"] <= 0:
        return 0.0, 0.0
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _frame_positions_m(
    root: Path,
    pairs: list[tuple[int, Path, Path]],
) -> tuple[dict[int, float], dict]:
    """Return token -> gantry position in metres plus source diagnostics.

    Priority:
    1. ``session.json`` frame_positions from in-app capture.
    2. Stakeholder flat filename tokens, which are gantry metres * 1e6.
    3. ``gantry_config.json`` step for numeric rgb/depth subfolders.
    4. Ordinal frame index fallback for old/partial datasets.
    """
    tokens = [int(t) for t, _, _ in pairs]
    positions: dict[int, float] = {}
    info = {"source": "ordinal", "median_step_m": 0.0, "warning": ""}

    session = load_session_json(root)
    frame_positions = (session or {}).get("frame_positions", {}) or {}
    if frame_positions:
        for token, rgb_path, _ in pairs:
            stem = rgb_path.stem
            keys = [str(token), stem]
            if stem.startswith("rgb_"):
                keys.append(stem[4:])
            for key in keys:
                if key in frame_positions:
                    positions[int(token)] = float(frame_positions[key])
                    break
        if len(positions) >= max(2, len(pairs) // 2):
            info["source"] = "session.json"

    if not positions and tokens:
        # Stakeholder ROS captures name flat frames as int(position_m * 1e6).
        token_range = max(tokens) - min(tokens)
        looks_like_position_token = (
            max(tokens) > 10_000
            and max(tokens) < 10_000_000
            and token_range > max(50, len(tokens) * 10)
        )
        if looks_like_position_token:
            positions = {int(t): float(t) / 1_000_000.0 for t in tokens}
            info["source"] = "filename_position_token"

    if not positions:
        gantry_cfg = load_gantry_config(root)
        if gantry_cfg:
            step_m, axis = gantry_cfg
            positions = {int(t): i * float(step_m) for i, t in enumerate(tokens)}
            info["source"] = "gantry_config"
            info["gantry_axis"] = int(axis)

    if not positions:
        positions = {int(t): float(i) for i, t in enumerate(tokens)}

    ordered = [positions[int(t)] for t in tokens if int(t) in positions]
    if len(ordered) >= 2:
        steps = np.diff(np.asarray(ordered, dtype=np.float64))
        nonzero = np.abs(steps[np.abs(steps) > 1e-12])
        if nonzero.size:
            median_step = float(np.median(nonzero))
            info["median_step_m"] = median_step
            if info["source"] != "ordinal" and median_step > 0.002:
                info["warning"] = (
                    f"Median frame spacing is {median_step * 1000:.2f} mm; "
                    "dense canopy fusion works best below about 2 mm/frame."
                )

    return positions, info


def _attach_candidate_positions(
    candidates: list[dict],
    positions_m: dict[int, float],
) -> None:
    for idx, item in enumerate(sorted(candidates, key=lambda c: int(c["token"]))):
        token = int(item["token"])
        item["position_m"] = float(positions_m.get(token, idx))

def _select_frames(
    candidates: list[dict], cfg: CanopyReconstructionConfig
) -> tuple[list[dict], int]:
    if not candidates:
        raise RuntimeError("No valid plant frames found.  Check mask/auto-mask settings.")

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    if cfg.max_candidates and cfg.max_candidates > 0:
        ranked = ranked[: max(int(cfg.max_candidates), int(cfg.max_frames))]

    if cfg.reference_token is not None:
        reference = min(ranked, key=lambda c: abs(int(c["token"]) - int(cfg.reference_token)))
    else:
        reference = ranked[0]

    ref = int(reference["token"])
    ref_pos = float(reference.get("position_m", ref))
    selected = sorted(
        ranked,
        key=lambda c: (
            abs(float(c.get("position_m", c["token"])) - ref_pos),
            -float(c.get("score", 0.0)),
        ),
    )[: max(1, int(cfg.max_frames))]

    selected = sorted(selected, key=lambda c: c["token"])
    return selected, ref


def _estimate_alignment_transforms(
    selected: list[dict],
    reference_token: int,
    cfg: CanopyReconstructionConfig,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, tuple[float, float]],
    tuple[int, int],
    dict[int, dict],
]:
    ref_item = next(c for c in selected if c["token"] == reference_token)
    ref_rgb = cv2.imread(str(ref_item["rgb_path"]))
    if ref_rgb is None:
        raise FileNotFoundError(f"Cannot read reference RGB: {ref_item['rgb_path']}")
    ref_mask = cv2.imread(str(ref_item["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if ref_mask is None:
        raise FileNotFoundError(f"Cannot read reference mask: {ref_item['mask_path']}")
    image_shape = ref_rgb.shape[:2]

    x, y, w, h = cv2.boundingRect((ref_mask > 0).astype(np.uint8))
    pad = max(48, int(min(image_shape) * 0.12))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image_shape[1], x + w + pad), min(image_shape[0], y + h + pad)
    ref_gray = cv2.cvtColor(ref_rgb[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)

    tokens: list[int] = []
    positions: list[float] = []
    centers: list[tuple[float, float]] = []
    weights: list[float] = []
    for item in selected:
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        tokens.append(int(item["token"]))
        positions.append(float(item.get("position_m", item["token"])))
        centers.append(_mask_centroid(mask))
        weights.append(max(float(item.get("mask_area", 1.0)), 1.0) ** 0.5)

    pos_arr = np.asarray(positions, dtype=np.float64)
    center_arr = np.asarray(centers, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    ref_pos = float(ref_item.get("position_m", reference_token))
    center_by_token = {t: tuple(c) for t, c in zip(tokens, centers)}

    def predict_center(token: int, position: float) -> tuple[float, float]:
        if len(pos_arr) >= 3 and float(np.ptp(pos_arr)) > 1e-12:
            scaled = (pos_arr - pos_arr.mean()) / float(np.ptp(pos_arr))
            coeff_x = np.polyfit(scaled, center_arr[:, 0], deg=1, w=weight_arr)
            coeff_y = np.polyfit(scaled, center_arr[:, 1], deg=1, w=weight_arr)
            x_scaled = (float(position) - float(pos_arr.mean())) / float(np.ptp(pos_arr))
            return float(np.polyval(coeff_x, x_scaled)), float(np.polyval(coeff_y, x_scaled))
        return tuple(center_by_token.get(token, _mask_centroid(ref_mask)))

    ref_center = predict_center(reference_token, ref_pos)
    transforms: dict[int, np.ndarray] = {}
    predicted_centers: dict[int, tuple[float, float]] = {}
    diagnostics: dict[int, dict] = {}

    for item in selected:
        token = int(item["token"])
        pos = float(item.get("position_m", token))
        predicted = predict_center(token, pos)
        base_dx = float(ref_center[0] - predicted[0])
        base_dy = float(ref_center[1] - predicted[1])
        if token == reference_token:
            transforms[token] = np.float32([[1, 0, 0], [0, 1, 0]])
            predicted_centers[token] = ref_center
            diagnostics[token] = {
                "position_m": pos,
                "base_shift_xy": [0.0, 0.0],
                "refine_shift_xy": [0.0, 0.0],
                "phase_response": 1.0,
                "alignment_method": "reference",
            }
            continue

        src_rgb = cv2.imread(str(item["rgb_path"]))
        if src_rgb is None:
            transforms[token] = np.float32([[1, 0, base_dx], [0, 1, base_dy]])
            predicted_centers[token] = predicted
            diagnostics[token] = {
                "position_m": pos,
                "base_shift_xy": [base_dx, base_dy],
                "refine_shift_xy": [0.0, 0.0],
                "phase_response": 0.0,
                "alignment_method": "motion_model",
            }
            continue

        base_transform = np.float32([[1, 0, base_dx], [0, 1, base_dy]])
        aligned_src = cv2.warpAffine(
            src_rgb, base_transform, (image_shape[1], image_shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        src_crop = aligned_src[y1:y2, x1:x2]
        src_gray = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        refine = (0.0, 0.0)
        response = 0.0
        try:
            shift, response = cv2.phaseCorrelate(src_gray, ref_gray)
            shift_mag = float(np.hypot(float(shift[0]), float(shift[1])))
            if response >= cfg.min_alignment_response and shift_mag <= cfg.max_alignment_refine_px:
                refine = (float(shift[0]), float(shift[1]))
        except cv2.error:
            refine = (0.0, 0.0)

        dx = base_dx + refine[0]
        dy = base_dy + refine[1]
        transforms[token] = np.float32([[1, 0, dx], [0, 1, dy]])
        predicted_centers[token] = predicted
        diagnostics[token] = {
            "position_m": pos,
            "base_shift_xy": [base_dx, base_dy],
            "refine_shift_xy": [float(refine[0]), float(refine[1])],
            "phase_response": float(response),
            "alignment_method": "motion_model+bounded_phase"
            if refine != (0.0, 0.0) else "motion_model",
        }

    return transforms, predicted_centers, image_shape, diagnostics


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
    """Return foreground-only raw depth and diagnostics.

    This intentionally does not fill holes per frame.  Filling before fusion
    lets far background values inside green masks vote as if they were leaf
    surface.  We gate each frame to the nearest coherent foreground band, fuse
    only those valid depths, and fill remaining holes after multi-frame fusion.
    """
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

    low = float(np.percentile(valid_values, 1.0))
    raw_high = float(np.percentile(valid_values, 98.0))
    foreground_anchor = float(
        np.percentile(valid_values, np.clip(cfg.foreground_percentile, 50.0, 99.0))
    )
    high = min(raw_high, foreground_anchor + float(cfg.foreground_margin_mm))
    valid = mask & (depth_u16 >= low) & (depth_u16 <= high)
    if not np.any(valid):
        raise RuntimeError("Depth filtering removed every point inside the canopy mask.")

    canvas = np.zeros(depth_u16.shape, dtype=np.float32)
    canvas[valid] = depth_u16[valid].astype(np.float32)
    canvas[~mask] = 0.0

    return canvas, {
        "mask_area": int(mask.sum()),
        "valid_depth_points": int(valid.sum()),
        "depth_valid_fraction": float(valid.sum() / max(int(mask.sum()), 1)),
        "depth_range_mm": [low, high],
        "raw_depth_range_mm": [float(np.min(valid_values)), raw_high],
        "far_depth_rejected": int(mask.sum() - valid.sum()),
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


def _build_mesh_and_point_cloud(
    fused_depth_mm: np.ndarray,
    fused_mask: np.ndarray,
    fused_rgb_bgr: np.ndarray,
    K: np.ndarray,
    z_scale: float = 1.0,
    do_cleanup: bool = True,
    nb_neighbors: int = 30,
    outlier_std_ratio: float = 2.0,
    max_height_jump_m: float = 0.08,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, float]:
    """Build a metric canopy height-field point cloud and triangle mesh."""
    fx, fy = float(K[0, 0]), float(abs(K[1, 1]))
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ys, xs = np.where(fused_mask & (fused_depth_mm > 0))
    if len(ys) == 0:
        raise RuntimeError("Fused depth map has no valid points.")

    depth_values_mm = fused_depth_mm[fused_mask & (fused_depth_mm > 0)]
    reference_depth_m = float(np.median(depth_values_mm) / 1000.0)
    baseline_depth_m = float(np.percentile(depth_values_mm, 95.0) / 1000.0)

    depth_m = (fused_depth_mm / 1000.0).astype(np.float32)
    height_map = (baseline_depth_m - depth_m).astype(np.float32)
    height_map[~fused_mask] = 0.0
    if np.any(fused_mask):
        h_clip = float(np.percentile(height_map[fused_mask], 99.5))
        height_map = np.clip(height_map, 0.0, max(h_clip, 1e-3))
        height_map = _smooth_in_mask(height_map, fused_mask, sigma=1.0)
        height_map[~fused_mask] = 0.0

    z_depth = depth_m[ys, xs].astype(np.float32)
    raw_x = (xs.astype(np.float32) - cx) * z_depth / fx
    raw_y = (ys.astype(np.float32) - cy) * z_depth / fy
    points = np.column_stack([
        raw_x - float(raw_x.mean()),
        -(raw_y - float(raw_y.mean())),
        height_map[ys, xs].astype(np.float32) * float(z_scale),
    ]).astype(np.float64)
    points[:, 2] -= float(points[:, 2].min())

    rgb_img = cv2.cvtColor(fused_rgb_bgr, cv2.COLOR_BGR2RGB)
    colors = rgb_img[ys, xs].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if do_cleanup and len(pcd.points) > nb_neighbors:
        clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=outlier_std_ratio
        )
        if not clean.is_empty():
            pcd = clean

    x0, y0, w, h = cv2.boundingRect(fused_mask.astype(np.uint8))
    id_map = -np.ones((h, w), dtype=np.int32)
    vertices: list[list[float]] = []
    vertex_colors: list[np.ndarray] = []
    x_center = float(raw_x.mean())
    y_center = float(raw_y.mean())

    for yy in range(y0, y0 + h):
        for xx in range(x0, x0 + w):
            if not fused_mask[yy, xx] or fused_depth_mm[yy, xx] <= 0:
                continue
            pixel_depth_m = float(depth_m[yy, xx])
            idx = len(vertices)
            vertices.append([
                (xx - cx) * pixel_depth_m / fx - x_center,
                -((yy - cy) * pixel_depth_m / fy - y_center),
                float(height_map[yy, xx] * z_scale),
            ])
            vertex_colors.append(rgb_img[yy, xx].astype(np.float64) / 255.0)
            id_map[yy - y0, xx - x0] = idx

    triangles: list[list[int]] = []
    for yy in range(h - 1):
        for xx in range(w - 1):
            ids = [
                id_map[yy, xx],
                id_map[yy, xx + 1],
                id_map[yy + 1, xx],
                id_map[yy + 1, xx + 1],
            ]
            if ids[0] >= 0 and ids[1] >= 0 and ids[2] >= 0:
                heights = [vertices[ids[0]][2], vertices[ids[1]][2], vertices[ids[2]][2]]
                if max(heights) - min(heights) <= max_height_jump_m:
                    triangles.append([ids[0], ids[2], ids[1]])
            if ids[1] >= 0 and ids[2] >= 0 and ids[3] >= 0:
                heights = [vertices[ids[1]][2], vertices[ids[2]][2], vertices[ids[3]][2]]
                if max(heights) - min(heights) <= max_height_jump_m:
                    triangles.append([ids[1], ids[2], ids[3]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return pcd, mesh, reference_depth_m


def _poisson_mesh_from_point_cloud(
    pcd: o3d.geometry.PointCloud,
    cfg: CanopyReconstructionConfig,
) -> o3d.geometry.TriangleMesh:
    if pcd.is_empty():
        return o3d.geometry.TriangleMesh()
    try:
        work = o3d.geometry.PointCloud(pcd)
        work.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        work.orient_normals_consistent_tangent_plane(k=15)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            work, depth=8, linear_fit=True
        )
        if cfg.mesh_cleanup and len(densities) > 0:
            density_arr = np.asarray(densities)
            threshold = np.quantile(
                density_arr, max(0.0, min(1.0, cfg.mesh_density_quantile))
            )
            mesh.remove_vertices_by_mask(density_arr < threshold)
        mesh.compute_vertex_normals()
        return mesh
    except Exception as exc:
        print(f"[canopy] Experimental Poisson mesh failed ({exc}); using height-field mesh.")
        return o3d.geometry.TriangleMesh()


def _display_mesh_with_thickness(
    mesh: o3d.geometry.TriangleMesh,
    thickness_m: float,
) -> o3d.geometry.TriangleMesh:
    if mesh is None or mesh.is_empty() or thickness_m <= 0:
        return mesh

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = (
        np.asarray(mesh.vertex_colors)
        if mesh.has_vertex_colors()
        else np.full((len(vertices), 3), 0.45, dtype=np.float64)
    )
    if len(vertices) == 0 or len(triangles) == 0:
        return mesh

    lower_vertices = vertices.copy()
    lower_vertices[:, 2] = np.maximum(lower_vertices[:, 2] - thickness_m, 0.0)
    lower_colors = np.clip(colors * 0.82, 0.0, 1.0)

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    n = len(vertices)
    out_tris = triangles.tolist()
    out_tris.extend([[int(c) + n, int(b) + n, int(a) + n] for a, b, c in triangles])
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        out_tris.append([a, b, b + n])
        out_tris.append([a, b + n, a + n])

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.vstack([vertices, lower_vertices]))
    out.vertex_colors = o3d.utility.Vector3dVector(np.vstack([colors, lower_colors]))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(out_tris, dtype=np.int32))
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


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


def _save_selected_frame_mosaic(
    output_dir: Path,
    selected: list[dict],
    max_tiles: int = 16,
) -> None:
    tiles: list[np.ndarray] = []
    for item in selected[:max(1, int(max_tiles))]:
        rgb = cv2.imread(str(item["rgb_path"]))
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if rgb is None:
            continue
        rgb = cv2.resize(rgb, (240, 135), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (240, 135), interpolation=cv2.INTER_NEAREST) > 0
            overlay = rgb.copy()
            overlay[mask] = (0, 220, 80)
            rgb = cv2.addWeighted(overlay, 0.28, rgb, 0.72, 0)
        label = f"token {item['token']}"
        cv2.putText(
            rgb, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            rgb, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (20, 40, 20), 1, cv2.LINE_AA,
        )
        tiles.append(rgb)
    if not tiles:
        return

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * 135, cols * 240, 3), dtype=np.uint8)
    canvas[:] = (16, 19, 20)
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, cols)
        canvas[row * 135:(row + 1) * 135, col * 240:(col + 1) * 240] = tile
    cv2.imwrite(str(output_dir / "selected_frames_mosaic.jpg"), canvas)


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
    positions_m, motion_info = _frame_positions_m(root, all_pairs)
    if motion_info.get("warning"):
        print(f"[canopy] WARNING: {motion_info['warning']}")

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
    _attach_candidate_positions(candidates, positions_m)

    selected, reference_token = _select_frames(candidates, cfg)
    print(
        f"[canopy] Selected {len(selected)} frames for fusion "
        f"(reference token={reference_token})."
    )

    transforms, predicted_centers, image_shape, alignment_info = _estimate_alignment_transforms(
        selected, reference_token, cfg
    )
    canvas_transforms, canvas_size, canvas_offset = _build_canvas_transforms(
        transforms, image_shape, padding=max(0, int(cfg.canvas_padding))
    )
    K_canvas = np.asarray(K, dtype=np.float64).copy()
    K_canvas[0, 2] += canvas_offset[0]
    K_canvas[1, 2] += canvas_offset[1]

    warped_depths, warped_depth_valids, warped_masks, warped_rgbs = [], [], [], []
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
        foreground_depth, depth_stats = _fill_depth_inside_mask(depth, clean_mask, cfg)
        foreground_valid = foreground_depth > 0
        warp = canvas_transforms[token]

        warped_depth = cv2.warpAffine(
            foreground_depth, warp, canvas_size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        warped_depth_valid = cv2.warpAffine(
            foreground_valid.astype(np.uint8) * 255, warp, canvas_size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
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
        warped_depth_valid_bool = warped_depth_valid > 0
        warped_depth[~warped_depth_valid_bool] = 0.0
        warped_rgb[~warped_mask_bool]   = 0.0

        warped_depths.append(warped_depth)
        warped_depth_valids.append(warped_depth_valid_bool)
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
            **alignment_info.get(token, {}),
            **depth_stats,
        })

    # Depth fusion (median)
    depth_coverage = np.sum(np.stack(warped_depth_valids, axis=0), axis=0)
    plant_coverage = np.sum(np.stack(warped_masks, axis=0), axis=0)
    stacked  = np.stack(
        [np.where(m, d, np.nan) for d, m in zip(warped_depths, warped_depth_valids)], axis=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused_depth = np.nanmedian(stacked, axis=0).astype(np.float32)

    min_votes = max(1, int(cfg.coverage_threshold))
    depth_vote_mask = depth_coverage >= min_votes
    depth_vote_mask_u8 = _clean_mask(
        depth_vote_mask.astype(np.uint8) * 255,
        keep_largest=False,
        min_component_area=cfg.min_component_area,
    )
    depth_vote_mask = depth_vote_mask_u8 > 0

    support_mask_u8 = _clean_mask(
        (plant_coverage >= min_votes).astype(np.uint8) * 255,
        keep_largest=False,
        min_component_area=cfg.min_component_area,
    )
    fused_mask = support_mask_u8 > 0

    valid = depth_vote_mask & fused_mask & np.isfinite(fused_depth) & (fused_depth > 0)
    if not np.any(valid):
        raise RuntimeError("Canopy fusion produced an empty depth map.")

    if cfg.max_hole_fill_distance_px > 0:
        fill_distance = distance_transform_edt(~valid)
        fused_mask = fused_mask & (
            depth_vote_mask | (fill_distance <= int(cfg.max_hole_fill_distance_px))
        )
    else:
        fused_mask = depth_vote_mask
    valid = valid & fused_mask
    if not np.any(valid):
        raise RuntimeError("Canopy fusion produced no depth after hole-fill gating.")

    fused_depth_full = _nearest_fill_in_mask(
        np.nan_to_num(fused_depth, nan=0.0), valid, fused_mask
    )
    fused_depth_full = _smooth_in_mask(
        fused_depth_full, fused_mask, sigma=cfg.smooth_sigma
    )
    fused_depth_full[~fused_mask] = 0.0

    # Colour fusion
    canvas_h, canvas_w = plant_coverage.shape
    color_sum  = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    for warped_rgb, warped_mask_bool in zip(warped_rgbs, warped_masks):
        color_sum += warped_rgb * warped_mask_bool[..., None].astype(np.float32)
    fused_rgb = np.zeros_like(color_sum, dtype=np.float32)
    np.divide(
        color_sum, np.maximum(plant_coverage[..., None], 1),
        out=fused_rgb, where=plant_coverage[..., None] > 0,
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
        crop_x, crop_y, crop_w, crop_h = crop_box
        depth_coverage = depth_coverage[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        plant_coverage = plant_coverage[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        fused_mask_u8 = (fused_mask.astype(np.uint8) * 255)
    else:
        fused_mask_u8 = (fused_mask.astype(np.uint8) * 255)

    # Save 2-D outputs
    masked_rgb_path = output_dir / "fused_rgb_masked.png"
    cv2.imwrite(str(masked_rgb_path), fused_rgb)
    cv2.imwrite(str(output_dir / "fused_mask.png"), fused_mask_u8)
    np.save(output_dir / "fused_depth_mm.npy", fused_depth_full)
    confidence = np.zeros_like(depth_coverage, dtype=np.uint8)
    if depth_coverage.max() > 0:
        confidence = np.clip(
            255.0 * depth_coverage.astype(np.float32) / float(depth_coverage.max()),
            0, 255,
        ).astype(np.uint8)
    cv2.imwrite(str(output_dir / "fused_confidence.png"), confidence)

    # Depth preview
    preview_depth = np.zeros(fused_depth_full.shape, dtype=np.uint8)
    inside = fused_depth_full[fused_mask]
    low_p, high_p = np.percentile(inside, [2, 98])
    scaled = np.clip((fused_depth_full - low_p) / max(high_p - low_p, 1e-6), 0.0, 1.0)
    preview_depth[fused_mask] = (255.0 * (1.0 - scaled[fused_mask])).astype(np.uint8)
    cv2.imwrite(str(output_dir / "fused_depth_vis.png"), preview_depth)

    # 3-D reconstruction
    pcd, mesh, reference_depth_m = _build_mesh_and_point_cloud(
        fused_depth_mm=fused_depth_full,
        fused_mask=fused_mask,
        fused_rgb_bgr=fused_rgb,
        K=K_canvas,
        z_scale=cfg.z_scale,
        do_cleanup=cfg.mesh_cleanup,
        nb_neighbors=cfg.nb_neighbors,
        outlier_std_ratio=cfg.outlier_std_ratio,
        max_height_jump_m=cfg.max_triangle_height_jump_m,
    )

    mesh_method = "heightfield"
    if cfg.use_poisson_mesh:
        poisson_mesh = _poisson_mesh_from_point_cloud(pcd, cfg)
        if not poisson_mesh.is_empty():
            mesh = poisson_mesh
            mesh_method = "poisson_experimental"

    display_mesh = _display_mesh_with_thickness(
        mesh, cfg.leaf_thickness_m if cfg.add_leaf_thickness else 0.0
    )

    point_cloud_path = output_dir / "canopy_points.ply"
    mesh_path        = output_dir / "canopy_mesh.ply"
    display_mesh_path = output_dir / "canopy_display_mesh.ply"
    viewer_path      = output_dir / "canopy_viewer.html"
    o3d.io.write_point_cloud(str(point_cloud_path), pcd)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_triangle_mesh(str(display_mesh_path), display_mesh)
    _save_previews(output_dir, pcd)
    _save_selected_frame_mosaic(output_dir, selected)

    summary = {
        "record_path": str(root.resolve()),
        "output_dir":  str(output_dir.resolve()),
        "point_cloud_path": str(point_cloud_path.resolve()),
        "mesh_path":   str(mesh_path.resolve()),
        "display_mesh_path": str(display_mesh_path.resolve()),
        "viewer_path": str(viewer_path.resolve()),
        "masked_rgb_path": str(masked_rgb_path.resolve()),
        "selected_mosaic_path": str((output_dir / "selected_frames_mosaic.jpg").resolve()),
        "mask_source": mask_source,
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "canvas_offset_xy": [float(canvas_offset[0]), float(canvas_offset[1])],
        "crop_box_xywh": crop_box,
        "motion": motion_info,
        "frames_available": len(all_pairs),
        "frames_used": len(selected),
        "reference_token": reference_token,
        "reference_depth_m": reference_depth_m,
        "final_point_count": len(np.asarray(pcd.points)),
        "final_triangle_count": len(np.asarray(mesh.triangles)),
        "display_triangle_count": len(np.asarray(display_mesh.triangles)),
        "mesh_method": mesh_method,
        "depth_coverage_max": int(depth_coverage.max()),
        "depth_coverage_mean_on_mask": float(depth_coverage[fused_mask].mean())
        if np.any(fused_mask) else 0.0,
        "config": asdict(cfg),
        "frames": frame_info,
    }
    summary_path = output_dir / "canopy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_canopy_mesh_viewer(
        display_mesh,
        viewer_path,
        title=f"{root.name} canopy",
        point_cloud=pcd,
        metadata={
            "Model": "display mesh" if cfg.add_leaf_thickness else "metric mesh",
            "Metric mesh": str(mesh_path.name),
            "Display mesh": str(display_mesh_path.name),
            "Frames": f"{len(selected)}/{len(all_pairs)}",
            "Motion": str(motion_info.get("source", "unknown")),
            "Motion warning": str(motion_info.get("warning", "")),
            "Depth coverage": f"{int(depth_coverage.max())} max votes",
            "Note": "Thickness/skirt is display-only; use canopy_mesh.ply for traits.",
        },
    )

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
