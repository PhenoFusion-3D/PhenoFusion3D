from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from natsort import natsorted


TOKEN_RE = re.compile(r"(?:rgb|depth)_(\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class FramePair:
    rgb_path: Path
    depth_path: Path
    token: int
    frame_index: int


@dataclass
class FrameCandidate:
    pair: FramePair
    mask: np.ndarray
    score: float
    mask_area_px: int
    valid_depth_fraction: float
    centroid_x_px: float = 0.0
    centroid_y_px: float = 0.0
    border_fraction: float = 0.0


@dataclass
class ReferenceTraitResult:
    plant_id: int
    frame_index: int
    frame_token: int
    rgb_path: str
    depth_path: str
    mask_area_px: int
    valid_depth_fraction: float
    projected_canopy_area_m2: float
    projected_convex_hull_area_m2: float
    canopy_major_span_m: float
    canopy_minor_span_m: float
    visible_depth_relief_m: float
    height_above_local_support_m: float | None
    support_depth_m: float | None
    visible_leaf_count: int
    visible_leaf_area_sum_m2: float
    leaf_width_mean_m: float | None
    leaf_width_max_m: float | None
    leaf_area_mean_m2: float | None
    confidence: str
    height_status: str
    leaf_traits_status: str
    warning: str
    overlay_path: str
    mask_path: str
    leaf_overlay_path: str


def _token(path: Path, fallback: int) -> int:
    match = TOKEN_RE.search(path.name)
    return int(match.group(1)) if match else fallback


def discover_frame_pairs(dataset_dir: str | Path) -> list[FramePair]:
    from .rgb_recovery.dataset import paired_images
    return [FramePair(rgb,depth,token,index) for index,(token,rgb,depth) in enumerate(paired_images(dataset_dir))]


def load_intrinsics(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    path = Path(dataset_dir) / "kdc_intrinsics.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing RGB intrinsics: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["K"], dtype=np.float64)
    distortion = np.asarray(data.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or matrix[0,0]<=0 or matrix[1,1]<=0 or not np.allclose(matrix[2],[0,0,1]):
        raise ValueError('Invalid calibrated camera matrix.')
    if distortion.size not in (4,5,8,12,14) or not np.isfinite(distortion).all():
        raise ValueError('Invalid camera distortion coefficients.')
    return matrix, distortion, int(data["width"]), int(data["height"])


def plant_mask(rgb_bgr: np.ndarray, min_component_area: int = 500) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(rgb_bgr.astype(np.int16))
    excess_green = 2 * g - r - b
    green = (cv2.inRange(hsv, (24, 28, 28), (105, 255, 255)) > 0) & (excess_green > 8)
    magenta = cv2.inRange(hsv, (125, 45, 25), (179, 255, 255)) > 0
    pale_green = (cv2.inRange(hsv, (18, 25, 35), (110, 160, 255)) > 0) & (excess_green > 6)
    mask = (green | magenta | pale_green).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[float, int]] = []
    height, width = mask.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        center_x = float(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] / 2.0)
        center_y = float(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] / 2.0)
        dx = (center_x - width / 2.0) / (width / 2.0)
        dy = (center_y - height / 2.0) / (height / 2.0)
        # During a gantry pass adjacent plants often share a frame.  Prefer the
        # plant crossing the optical centre; area alone otherwise selects a
        # large, clipped neighbour at the image edge.
        components.append((math.sqrt(area) * math.exp(-4.0 * (dx * dx + dy * dy)), label))
    cleaned = np.zeros_like(mask)
    if components:
        _, selected_label = max(components)
        cleaned[labels == selected_label] = 255
    return cleaned


def valid_foreground_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale: float = 1000.0,
    depth_min_m: float = 0.05,
    depth_max_m: float = 2.0,
) -> np.ndarray:
    min_raw = int(round(depth_min_m * depth_scale))
    max_raw = int(round(depth_max_m * depth_scale))
    valid = (mask > 0) & (depth >= min_raw) & (depth <= max_raw)
    values = depth[valid]
    if len(values) < 100:
        return valid.astype(np.uint8) * 255
    upper = min(float(max_raw), float(np.percentile(values, 97)) + 0.08 * depth_scale)
    return ((mask > 0) & (depth >= min_raw) & (depth <= upper)).astype(np.uint8) * 255


def _candidate_score(
    mask: np.ndarray, depth: np.ndarray, depth_scale: float = 1000.0
) -> tuple[float, float, float, float, float]:
    foreground = valid_foreground_mask(mask, depth, depth_scale)
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0
    mask_area = max(1, int(np.count_nonzero(mask)))
    valid_fraction = float(len(xs) / mask_area)
    height, width = mask.shape
    dx = (float(np.mean(xs)) - width / 2.0) / (width / 2.0)
    dy = (float(np.mean(ys)) - height / 2.0) / (height / 2.0)
    centrality = math.exp(-3.5 * (dx * dx + dy * dy))
    border = np.zeros_like(foreground, dtype=bool)
    border_width = max(8, int(round(min(height, width) * 0.025)))
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border_fraction = float(np.count_nonzero((foreground > 0) & border) / len(xs))
    edge_penalty = math.exp(-12.0 * border_fraction)
    return (
        float(len(xs) * valid_fraction * centrality * edge_penalty),
        valid_fraction,
        float(np.mean(xs)),
        float(np.mean(ys)),
        border_fraction,
    )


def inspect_candidates(
    pairs: Iterable[FramePair],
    discovery_stride: int = 3,
    depth_scale: float = 1000.0,
) -> list[FrameCandidate]:
    candidates: list[FrameCandidate] = []
    for pair in list(pairs)[:: max(1, discovery_stride)]:
        rgb = cv2.imread(str(pair.rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(pair.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape[:2]:
            continue
        mask = plant_mask(rgb)
        area = int(np.count_nonzero(mask))
        score, valid_fraction, center_x, center_y, border_fraction = _candidate_score(
            mask, depth, depth_scale
        )
        if area >= 2_000 and valid_fraction >= 0.1:
            candidates.append(
                FrameCandidate(pair, mask, score, area, valid_fraction, center_x, center_y, border_fraction)
            )
    return candidates


def select_plant_candidates(
    candidates: list[FrameCandidate],
    expected_plants: int | None = None,
    min_separation_frames: int = 110,
) -> list[FrameCandidate]:
    if not candidates:
        return []
    def crossing_score(item: FrameCandidate) -> float:
        height, width = item.mask.shape
        dx = (item.centroid_x_px - width / 2.0) / (width / 2.0)
        dy = (item.centroid_y_px - height / 2.0) / (height / 2.0)
        return (
            item.valid_depth_fraction
            * math.exp(-5.0 * (dx * dx + dy * dy))
            * math.exp(-12.0 * item.border_fraction)
        )

    # Plant passages are identified by their optical-centre crossing, not by
    # canopy area. This prevents one large plant from consuming several slots.
    ranked = sorted(candidates, key=crossing_score, reverse=True)
    selected: list[FrameCandidate] = []
    target = expected_plants if expected_plants and expected_plants > 0 else 12
    for candidate in ranked:
        if all(abs(candidate.pair.frame_index - other.pair.frame_index) >= min_separation_frames for other in selected):
            selected.append(candidate)
            if len(selected) >= target:
                break

    if expected_plants is None:
        best_crossing = crossing_score(ranked[0])
        strong = [item for item in selected if crossing_score(item) >= best_crossing * 0.35]
        selected = strong or selected[:1]
    return sorted(selected, key=lambda item: item.pair.frame_index)


def select_candidates_by_frame(
    candidates: list[FrameCandidate], frame_indices: Iterable[int]
) -> list[FrameCandidate]:
    selected: list[FrameCandidate] = []
    for requested in frame_indices:
        if not candidates:
            break
        nearest = min(candidates, key=lambda item: abs(item.pair.frame_index - int(requested)))
        if nearest not in selected:
            selected.append(nearest)
    return sorted(selected, key=lambda item: item.pair.frame_index)


def write_discovery_diagnostics(
    candidates: list[FrameCandidate], selected: list[FrameCandidate], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = {item.pair.frame_index for item in selected}
    fields = [
        "frame_index", "frame_token", "score", "mask_area_px", "valid_depth_fraction",
        "centroid_x_px", "centroid_y_px", "border_fraction", "selected",
    ]
    with (output_dir / "candidate_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "frame_index": item.pair.frame_index,
                    "frame_token": item.pair.token,
                    "score": item.score,
                    "mask_area_px": item.mask_area_px,
                    "valid_depth_fraction": item.valid_depth_fraction,
                    "centroid_x_px": item.centroid_x_px,
                    "centroid_y_px": item.centroid_y_px,
                    "border_fraction": item.border_fraction,
                    "selected": item.pair.frame_index in selected_indices,
                }
            )

    if candidates:
        sample_count = min(30, len(candidates))
        sample_indices = np.linspace(0, len(candidates) - 1, sample_count, dtype=int)
        contact_tiles: list[np.ndarray] = []
        for candidate_index in sample_indices:
            item = candidates[int(candidate_index)]
            rgb = cv2.imread(str(item.pair.rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                continue
            tile = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
            tile_mask = cv2.resize(item.mask, (320, 180), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tile, contours, -1, (0, 255, 255), 2)
            cv2.putText(
                tile,
                f"frame {item.pair.frame_index}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                tile,
                f"frame {item.pair.frame_index}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
            )
            contact_tiles.append(tile)
        if contact_tiles:
            columns = 5
            blank = np.zeros_like(contact_tiles[0])
            rows = []
            for start in range(0, len(contact_tiles), columns):
                row = contact_tiles[start : start + columns]
                row.extend([blank] * (columns - len(row)))
                rows.append(np.hstack(row))
            cv2.imwrite(str(output_dir / "discovery_contact_sheet.jpg"), np.vstack(rows))

    if not selected:
        return
    tiles: list[np.ndarray] = []
    for item in selected:
        rgb = cv2.imread(str(item.pair.rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        preview = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
        preview_mask = cv2.resize(item.mask, (480, 270), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(preview_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
        cv2.putText(
            preview,
            f"frame {item.pair.frame_index}  token {item.pair.token}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            preview,
            f"frame {item.pair.frame_index}  token {item.pair.token}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
        )
        tiles.append(preview)
    if tiles:
        cv2.imwrite(str(output_dir / "selected_frames.jpg"), np.vstack(tiles))


def _metric_points(
    mask: np.ndarray,
    depth: np.ndarray,
    matrix: np.ndarray,
    depth_scale: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero((mask > 0) & (depth > 0))
    z = depth[ys, xs].astype(np.float64) / depth_scale
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y)), z, np.column_stack((xs, ys))


def _hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def _oriented_spans(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    (_, _), (width, height), _ = cv2.minAreaRect(points.astype(np.float32))
    return float(max(width, height)), float(min(width, height))


def _pixel_area_sum(
    mask: np.ndarray,
    depth: np.ndarray,
    matrix: np.ndarray,
    depth_scale: float = 1000.0,
) -> float:
    z = depth[(mask > 0) & (depth > 0)].astype(np.float64) / depth_scale
    return float(np.sum((z * z) / (float(matrix[0, 0]) * float(matrix[1, 1]))))


def _support_depth(
    mask: np.ndarray, depth: np.ndarray, depth_scale: float = 1000.0
) -> float | None:
    outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    inner = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    ring = (
        (outer > 0)
        & (inner == 0)
        & (depth >= int(round(0.05 * depth_scale)))
        & (depth <= int(round(2.0 * depth_scale)))
    )
    values = depth[ring]
    if len(values) < 1_000:
        return None
    median = float(np.median(values)) / depth_scale
    mad = float(np.median(np.abs(values.astype(np.float64) / depth_scale - median)))
    return median if mad <= 0.08 else None


def segment_visible_leaves(rgb: np.ndarray, mask: np.ndarray, min_leaf_area_px: int = 1_200) -> list[np.ndarray]:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(distance.max()) <= 0:
        return []
    sure_foreground = (distance > max(5.0, 0.24 * float(distance.max()))).astype(np.uint8)
    count, markers = cv2.connectedComponents(sure_foreground)
    if count <= 2:
        return []
    markers = markers + 1
    unknown = (mask > 0) & (sure_foreground == 0)
    markers[unknown] = 0
    watershed = cv2.watershed(rgb.copy(), markers.astype(np.int32))
    leaves: list[np.ndarray] = []
    for label in range(2, int(watershed.max()) + 1):
        leaf = ((watershed == label) & (mask > 0)).astype(np.uint8) * 255
        if int(np.count_nonzero(leaf)) >= min_leaf_area_px:
            leaves.append(leaf)
    return leaves


def _leaf_traits(
    leaves: list[np.ndarray],
    depth: np.ndarray,
    matrix: np.ndarray,
    depth_scale: float = 1000.0,
) -> tuple[list[dict[str, float]], float | None, float | None, float | None]:
    rows: list[dict[str, float]] = []
    for leaf in leaves:
        points, _, _ = _metric_points(leaf, depth, matrix, depth_scale)
        if len(points) < 3:
            continue
        length, width = _oriented_spans(points)
        area = _pixel_area_sum(leaf, depth, matrix, depth_scale)
        rows.append({"projected_area_m2": area, "length_m": length, "width_m": width})
    if not rows:
        return rows, None, None, None
    widths = np.asarray([row["width_m"] for row in rows], dtype=float)
    areas = np.asarray([row["projected_area_m2"] for row in rows], dtype=float)
    return rows, float(np.mean(widths)), float(np.max(widths)), float(np.mean(areas))


def extract_candidate_traits(
    candidate: FrameCandidate,
    matrix: np.ndarray,
    output_dir: Path,
    plant_id: int,
    depth_scale: float = 1000.0,
    distortion: np.ndarray | None = None,
) -> tuple[ReferenceTraitResult, list[dict[str, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(candidate.pair.rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(candidate.pair.depth_path), cv2.IMREAD_UNCHANGED)
    outline = candidate.mask
    if distortion is not None and np.any(distortion):
        height,width=depth.shape
        maps=cv2.initUndistortRectifyMap(matrix,distortion,None,matrix,(width,height),cv2.CV_32FC1)
        rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR)
        depth=cv2.remap(depth,*maps,cv2.INTER_NEAREST)
        outline=cv2.remap(outline,*maps,cv2.INTER_NEAREST)
    mask = valid_foreground_mask(outline, depth, depth_scale)
    points, z, pixels = _metric_points(mask, depth, matrix, depth_scale)
    if len(points) < 3:
        raise RuntimeError(f"Insufficient valid plant depth in {candidate.pair.rgb_path}")

    projected_area = _pixel_area_sum(mask, depth, matrix, depth_scale)
    hull_area = _hull_area(points)
    major, minor = _oriented_spans(points)
    low_depth = float(np.percentile(z, 5))
    high_depth = float(np.percentile(z, 95))
    relief = max(0.0, high_depth - low_depth)
    support = _support_depth(mask, depth, depth_scale)
    height = max(0.0, support - low_depth) if support is not None else None

    leaves = segment_visible_leaves(rgb, mask)
    leaf_rows, width_mean, width_max, area_mean = _leaf_traits(
        leaves, depth, matrix, depth_scale
    )
    leaf_area_sum = float(sum(row["projected_area_m2"] for row in leaf_rows))

    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)
    hull = cv2.convexHull(pixels.astype(np.int32))
    cv2.polylines(overlay, [hull], True, (255, 80, 20), 3)
    cv2.putText(overlay, f"Plant {plant_id}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 255), 3)

    leaf_overlay = rgb.copy()
    palette = [(255, 80, 20), (20, 180, 255), (180, 40, 220), (40, 220, 80), (220, 180, 20)]
    for index, leaf in enumerate(leaves):
        contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(leaf_overlay, contours, -1, palette[index % len(palette)], 3)
        if contours:
            moments = cv2.moments(max(contours, key=cv2.contourArea))
            if moments["m00"]:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])
                cv2.putText(leaf_overlay, str(index + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(leaf_overlay, str(index + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    overlay_path = output_dir / "reference_overlay.png"
    mask_path = output_dir / "plant_mask.png"
    leaf_overlay_path = output_dir / "leaf_segments.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(leaf_overlay_path), leaf_overlay)

    warnings: list[str] = []
    if support is None:
        warnings.append("No stable local support plane; plant height requires physical/manual reference")
        height_status = "unavailable"
    else:
        warnings.append("Height uses a nearby support surface, not a confirmed plant base plane")
        height_status = "conditional"
    warnings.append("Individual leaf traits are exploratory for overlapping canopies")
    leaf_traits_status = "exploratory"
    if candidate.valid_depth_fraction < 0.75:
        warnings.append("Low valid-depth coverage")
    if candidate.border_fraction > 0.01:
        warnings.append("Selected canopy touches the image border; projected traits may be underestimated")
    confidence = (
        "high"
        if candidate.valid_depth_fraction >= 0.8 and candidate.border_fraction <= 0.005
        else ("medium" if candidate.valid_depth_fraction >= 0.7 and candidate.border_fraction <= 0.02 else "low")
    )

    result = ReferenceTraitResult(
        plant_id=plant_id,
        frame_index=candidate.pair.frame_index,
        frame_token=candidate.pair.token,
        rgb_path=str(candidate.pair.rgb_path),
        depth_path=str(candidate.pair.depth_path),
        mask_area_px=candidate.mask_area_px,
        valid_depth_fraction=candidate.valid_depth_fraction,
        projected_canopy_area_m2=projected_area,
        projected_convex_hull_area_m2=hull_area,
        canopy_major_span_m=major,
        canopy_minor_span_m=minor,
        visible_depth_relief_m=relief,
        height_above_local_support_m=height,
        support_depth_m=support,
        visible_leaf_count=len(leaf_rows),
        visible_leaf_area_sum_m2=leaf_area_sum,
        leaf_width_mean_m=width_mean,
        leaf_width_max_m=width_max,
        leaf_area_mean_m2=area_mean,
        confidence=confidence,
        height_status=height_status,
        leaf_traits_status=leaf_traits_status,
        warning="; ".join(warnings),
        overlay_path=str(overlay_path),
        mask_path=str(mask_path),
        leaf_overlay_path=str(leaf_overlay_path),
    )
    (output_dir / "visible_leaves.json").write_text(json.dumps(leaf_rows, indent=2), encoding="utf-8")
    return result, leaf_rows


def write_reference_outputs(results: list[ReferenceTraitResult], output_dir: Path, dataset_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(dataset_dir),
        "reference_type": "image_derived_rgbd",
        "manual_ground_truth": False,
        "notes": [
            "Projected traits use calibrated RGB pixels and aligned depth.",
            "Visible leaf traits are watershed estimates and require overlay review.",
            "Physical ruler/scanner measurements remain the validation ground truth.",
        ],
        "plants": [asdict(result) for result in results],
    }
    (output_dir / "reference_traits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if results:
        with (output_dir / "reference_traits.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)

    manual_fields = [
        "plant_id",
        "physical_plant_height_m",
        "physical_canopy_major_span_m",
        "physical_canopy_minor_span_m",
        "physical_projected_canopy_area_m2",
        "physical_projected_convex_hull_area_m2",
        "physical_leaf_width_mean_m",
        "physical_leaf_area_mean_m2",
        "operator",
        "measurement_date",
        "notes",
    ]
    with (output_dir / "manual_measurements_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manual_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"plant_id": result.plant_id})


def extract_dataset_reference_traits(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    expected_plants: int | None = None,
    discovery_stride: int = 3,
    min_separation_frames: int = 55,
    frame_indices: Iterable[int] | None = None,
    depth_scale: float = 1000.0,
) -> list[ReferenceTraitResult]:
    dataset = Path(dataset_dir)
    if not np.isfinite(depth_scale) or depth_scale<=0:
        raise ValueError('Depth scale must be finite positive raw units per metre.')
    destination = Path(output_dir) if output_dir else dataset / "reference_traits"
    pairs = discover_frame_pairs(dataset)
    matrix, distortion, width, height = load_intrinsics(dataset)
    candidates = inspect_candidates(
        pairs, discovery_stride=discovery_stride, depth_scale=depth_scale
    )
    selected = (
        select_candidates_by_frame(candidates, frame_indices)
        if frame_indices is not None
        else select_plant_candidates(candidates, expected_plants, min_separation_frames)
    )
    if not selected:
        raise RuntimeError(f"No usable plant frames detected in {dataset}")
    write_discovery_diagnostics(candidates, selected, destination)
    results: list[ReferenceTraitResult] = []
    for plant_id, candidate in enumerate(selected, start=1):
        rgb=cv2.imread(str(candidate.pair.rgb_path))
        depth=cv2.imread(str(candidate.pair.depth_path),cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2]!=(height,width) or depth.shape!=(height,width) or depth.dtype!=np.uint16:
            raise ValueError('Reference RGB and aligned uint16 depth must match calibration resolution.')
        result, _ = extract_candidate_traits(
            candidate,
            matrix,
            destination / f"plant_{plant_id:02d}",
            plant_id,
            depth_scale,
            distortion,
        )
        results.append(result)
    write_reference_outputs(results, destination, dataset)
    return results
