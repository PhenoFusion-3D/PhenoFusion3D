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

from file_io.loader import load_intrinsics
from visualiser.viewer import write_point_cloud_viewer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class CanopyReconstructionConfig:
    mask_dir: str | None = None
    max_frames: int = 9
    min_mask_area: int = 180000
    reference_token: int | None = None
    coverage_threshold: int = 1
    depth_min: int | None = 500
    depth_max: int | None = 4000
    min_valid_depth_points: int = 5000
    smooth_sigma: float = 3.5
    z_scale: float = 1.0
    output_dir: str | None = None
    auto_mask: bool = True
    min_component_area: int = 12000
    mask_hue_min: int = 28
    mask_hue_max: int = 95
    mask_s_min: int = 45
    mask_v_min: int = 35
    mask_exg_min: int = 20
    canvas_padding: int = 48
    crop_to_mask: bool = True


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


def _frame_token(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _keep_largest(mask_u8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_u8)
    out[labels == largest] = 255
    return out


def _drop_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
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


def _clean_mask(mask_u8: np.ndarray, *, keep_largest: bool = True, min_component_area: int = 0) -> np.ndarray:
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    if keep_largest:
        mask_u8 = _keep_largest(mask_u8)
    elif min_component_area > 0:
        mask_u8 = _drop_small_components(mask_u8, min_component_area)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask_u8 = _fill_small_holes(mask_u8, max(12000, min_component_area))

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
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((raw > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return raw

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max())
    if largest_area < cfg.min_component_area:
        return np.zeros_like(raw)

    # The target plant can be split by highlights/shadows, but small calibration
    # patches and background pots should not join the target. Keep only sizeable
    # green islands relative to the dominant one.
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
        return {
            "mask_area": 0,
            "bbox": [0, 0, 0, 0],
            "edge_fraction": 1.0,
            "score": 0.0,
        }

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


def _smooth_in_mask(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weight = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    smooth = gaussian_filter(values.astype(np.float32) * mask.astype(np.float32), sigma=sigma)
    out = np.zeros_like(values, dtype=np.float32)
    np.divide(smooth, weight, out=out, where=weight > 1e-6)
    return out


def _nearest_fill_in_mask(values: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise RuntimeError("No usable depth values inside the canopy mask.")

    canvas = np.zeros(values.shape, dtype=np.float32)
    canvas[valid] = values[valid].astype(np.float32)
    missing = mask & ~valid
    if np.any(missing):
        _, nearest_indices = distance_transform_edt(~valid, return_indices=True)
        canvas[missing] = canvas[nearest_indices[0][missing], nearest_indices[1][missing]]
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

    low = float(np.percentile(valid_values, 1.0))
    high = float(np.percentile(valid_values, 98.0))
    valid = mask & (depth_u16 >= low) & (depth_u16 <= high)
    if not np.any(valid):
        raise RuntimeError("Depth filtering removed every point inside the canopy mask.")

    canvas = _nearest_fill_in_mask(depth_u16.astype(np.float32), valid, mask)
    canvas = _smooth_in_mask(canvas, mask, sigma=cfg.smooth_sigma)

    # A light distance-transform prior helps broad leaves read as continuous surfaces.
    canopy = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if canopy.max() > 0:
        canvas -= 30.0 * (canopy / canopy.max()).astype(np.float32)
    canvas[~mask] = 0.0

    return canvas, {
        "mask_area": int(mask.sum()),
        "valid_depth_points": int(valid.sum()),
        "depth_range_mm": [low, high],
    }


def _load_mask_candidates(record_path: Path, mask_dir: Path, cfg: CanopyReconstructionConfig) -> list[dict]:
    candidates = []
    for mask_path in sorted(mask_dir.glob("mask_*.png"), key=_frame_token):
        token = _frame_token(mask_path)
        rgb_path = record_path / f"rgb_{token}.png"
        depth_path = record_path / f"depth_{token}.png"
        if not rgb_path.exists() or not depth_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        clean_mask = _clean_mask(mask, keep_largest=False, min_component_area=cfg.min_component_area)
        stats = _mask_stats(clean_mask, cfg.min_component_area)
        if stats["mask_area"] < cfg.min_mask_area:
            continue
        candidates.append(
            {
                "token": token,
                "mask_path": mask_path,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "mask_area": stats["mask_area"],
                "bbox": stats["bbox"],
                "edge_fraction": stats["edge_fraction"],
                "score": stats["score"],
            }
        )
    return candidates


def _load_auto_candidates(record_path: Path, auto_mask_dir: Path, cfg: CanopyReconstructionConfig) -> list[dict]:
    auto_mask_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    preview_scale = 0.25
    preview_cfg = replace(
        cfg,
        min_component_area=max(16, int(cfg.min_component_area * preview_scale * preview_scale)),
    )
    for rgb_path in sorted(record_path.glob("rgb_*.png"), key=_frame_token):
        token = _frame_token(rgb_path)
        depth_path = record_path / f"depth_{token}.png"
        if not depth_path.exists():
            continue
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue

        preview = cv2.resize(rgb, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
        preview_mask = _auto_leaf_mask(preview, preview_cfg)
        preview_stats = _mask_stats(preview_mask, preview_cfg.min_component_area)
        estimated_full_area = preview_stats["mask_area"] / (preview_scale * preview_scale)
        if estimated_full_area < cfg.min_mask_area * 0.7:
            continue

        mask = _auto_leaf_mask(rgb, cfg)
        stats = _mask_stats(mask, cfg.min_component_area)
        if stats["mask_area"] < cfg.min_mask_area:
            continue
        mask_path = auto_mask_dir / f"mask_{token}.png"
        cv2.imwrite(str(mask_path), mask)
        candidates.append(
            {
                "token": token,
                "mask_path": mask_path,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "mask_area": stats["mask_area"],
                "bbox": stats["bbox"],
                "edge_fraction": stats["edge_fraction"],
                "score": stats["score"],
            }
        )
    return candidates


def _select_frames(candidates: list[dict], cfg: CanopyReconstructionConfig) -> tuple[list[dict], int]:
    if not candidates:
        raise ValueError("No mask-backed plant frames matched the requested canopy settings.")

    if cfg.reference_token is None:
        reference = max(candidates, key=lambda item: item.get("score", item["mask_area"]))
        reference_token = int(reference["token"])
    else:
        requested = int(cfg.reference_token)
        reference = min(candidates, key=lambda item: abs(int(item["token"]) - requested))
        reference_token = int(reference["token"])

    ordered = sorted(candidates, key=lambda item: (abs(int(item["token"]) - reference_token), -int(item["mask_area"])))
    if cfg.max_frames is not None:
        ordered = ordered[: max(1, int(cfg.max_frames))]
    selected = sorted(ordered, key=lambda item: item["token"])
    return selected, reference_token


def _mask_centroid(mask_u8: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments((mask_u8 > 0).astype(np.uint8))
    if moments["m00"] <= 0:
        return 0.0, 0.0
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _estimate_alignment_transforms(
    selected: list[dict],
    reference_token: int,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[float, float]], tuple[int, int]]:
    tokens = []
    centers = []
    weights = []
    image_shape = None

    for item in selected:
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read canopy mask: {item['mask_path']}")
        if image_shape is None:
            image_shape = mask.shape[:2]
        cx, cy = _mask_centroid(mask)
        tokens.append(float(item["token"]))
        centers.append([cx, cy])
        weights.append(max(float(item["mask_area"]), 1.0) ** 0.5)

    if image_shape is None:
        raise RuntimeError("No selected masks were readable.")

    token_arr = np.asarray(tokens, dtype=np.float64)
    center_arr = np.asarray(centers, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    transforms: dict[int, np.ndarray] = {}
    predicted_centers: dict[int, tuple[float, float]] = {}

    if len(selected) >= 3 and float(np.ptp(token_arr)) > 0:
        token_mean = float(token_arr.mean())
        token_scale = float(np.ptp(token_arr))
        x = (token_arr - token_mean) / token_scale
        coeff_x = np.polyfit(x, center_arr[:, 0], deg=1, w=weight_arr)
        coeff_y = np.polyfit(x, center_arr[:, 1], deg=1, w=weight_arr)

        def predict(token: float) -> tuple[float, float]:
            scaled = (float(token) - token_mean) / token_scale
            return float(np.polyval(coeff_x, scaled)), float(np.polyval(coeff_y, scaled))

        reference_center = predict(reference_token)
        for item in selected:
            token = int(item["token"])
            predicted = predict(token)
            predicted_centers[token] = predicted
            dx = reference_center[0] - predicted[0]
            dy = reference_center[1] - predicted[1]
            transforms[token] = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    else:
        center_by_token = {int(item["token"]): tuple(center) for item, center in zip(selected, center_arr)}
        reference_center = center_by_token.get(reference_token, tuple(center_arr[len(center_arr) // 2]))
        for item in selected:
            token = int(item["token"])
            center = center_by_token[token]
            predicted_centers[token] = (float(center[0]), float(center[1]))
            dx = float(reference_center[0] - center[0])
            dy = float(reference_center[1] - center[1])
            transforms[token] = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    transforms[reference_token] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return transforms, predicted_centers, image_shape


def _build_canvas_transforms(
    transforms: dict[int, np.ndarray],
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[dict[int, np.ndarray], tuple[int, int], tuple[float, float]]:
    height, width = image_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width), 0.0, 1.0],
            [0.0, float(height), 1.0],
            [float(width), float(height), 1.0],
        ],
        dtype=np.float32,
    )
    warped_corners = []
    for transform in transforms.values():
        warped_corners.append((transform @ corners.T).T)
    all_corners = np.vstack(warped_corners)
    min_xy = all_corners[:, :2].min(axis=0)
    max_xy = all_corners[:, :2].max(axis=0)
    offset_x = float(padding - min_xy[0])
    offset_y = float(padding - min_xy[1])
    canvas_width = int(np.ceil(max_xy[0] - min_xy[0] + padding * 2))
    canvas_height = int(np.ceil(max_xy[1] - min_xy[1] + padding * 2))

    canvas_transforms = {}
    for token, transform in transforms.items():
        adjusted = transform.copy()
        adjusted[0, 2] += offset_x
        adjusted[1, 2] += offset_y
        canvas_transforms[token] = adjusted
    return canvas_transforms, (canvas_width, canvas_height), (offset_x, offset_y)


def _crop_to_mask(
    fused_depth: np.ndarray,
    fused_mask: np.ndarray,
    fused_rgb: np.ndarray,
    K: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if not np.any(fused_mask):
        return fused_depth, fused_mask, fused_rgb, K, [0, 0, fused_mask.shape[1], fused_mask.shape[0]]
    x, y, w, h = cv2.boundingRect(fused_mask.astype(np.uint8))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(fused_mask.shape[1], x + w + padding)
    y1 = min(fused_mask.shape[0], y + h + padding)
    cropped_k = K.copy()
    cropped_k[0, 2] -= x0
    cropped_k[1, 2] -= y0
    return (
        fused_depth[y0:y1, x0:x1],
        fused_mask[y0:y1, x0:x1],
        fused_rgb[y0:y1, x0:x1],
        cropped_k,
        [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
    )


def _build_mesh_and_point_cloud(
    fused_depth_mm: np.ndarray,
    fused_mask: np.ndarray,
    fused_rgb_bgr: np.ndarray,
    K: np.ndarray,
    z_scale: float,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, float]:
    ys, xs = np.where(fused_mask)
    depth_values_mm = fused_depth_mm[fused_mask]
    reference_depth_m = float(np.median(depth_values_mm) / 1000.0)
    baseline_depth_m = float(np.percentile(depth_values_mm, 95.0) / 1000.0)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    depth_m = (fused_depth_mm / 1000.0).astype(np.float32)
    height_map = (baseline_depth_m - depth_m).astype(np.float32)
    height_map[~fused_mask] = 0.0
    height_map = np.clip(height_map, 0.0, float(np.percentile(height_map[fused_mask], 99.0)))
    height_map = _smooth_in_mask(height_map, fused_mask, sigma=2.5)
    height_map[~fused_mask] = 0.0

    z_depth = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z_depth / fx
    y = (ys.astype(np.float32) - cy) * z_depth / fy
    z = height_map[ys, xs].astype(np.float32) * z_scale
    x -= float(x.mean())
    y -= float(y.mean())
    y *= -1.0
    z -= float(z.min())

    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = fused_rgb_bgr[ys, xs][:, ::-1].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd = pcd.voxel_down_sample(0.002)

    x0, y0, w, h = cv2.boundingRect((fused_mask.astype(np.uint8) * 255))
    id_map = -np.ones((h, w), dtype=np.int32)
    vertices = []
    vertex_colors = []
    for yy in range(y0, y0 + h):
        for xx in range(x0, x0 + w):
            if not fused_mask[yy, xx]:
                continue
            idx = len(vertices)
            pixel_depth_m = float(depth_m[yy, xx])
            x_coord = (xx - cx) * pixel_depth_m / fx - float(x.mean())
            y_coord = -((yy - cy) * pixel_depth_m / fy - float(y.mean()))
            z_coord = float(height_map[yy, xx] * z_scale - float(z.min()))
            vertices.append([x_coord, y_coord, z_coord])
            vertex_colors.append(fused_rgb_bgr[yy, xx][::-1] / 255.0)
            id_map[yy - y0, xx - x0] = idx

    triangles = []
    for yy in range(h - 1):
        for xx in range(w - 1):
            ids = [id_map[yy, xx], id_map[yy, xx + 1], id_map[yy + 1, xx], id_map[yy + 1, xx + 1]]
            if ids[0] >= 0 and ids[1] >= 0 and ids[2] >= 0:
                heights = [vertices[ids[0]][2], vertices[ids[1]][2], vertices[ids[2]][2]]
                if max(heights) - min(heights) < 0.06:
                    triangles.append([ids[0], ids[2], ids[1]])
            if ids[1] >= 0 and ids[2] >= 0 and ids[3] >= 0:
                heights = [vertices[ids[1]][2], vertices[ids[2]][2], vertices[ids[3]][2]]
                if max(heights) - min(heights) < 0.06:
                    triangles.append([ids[1], ids[2], ids[3]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return pcd, mesh, reference_depth_m


def _save_previews(output_dir: Path, pcd: o3d.geometry.PointCloud) -> None:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    fig = plt.figure(figsize=(12, 6), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(points[:, 0], points[:, 1], s=0.6, c=colors)
    ax1.set_title("Top View")
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(points[:, 0], points[:, 1], s=0.6, c=points[:, 2], cmap="viridis")
    ax2.set_title("Height")
    ax2.set_aspect("equal")
    ax2.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / "canopy_topdown.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(points) // 50000)
    selected = slice(None, None, step)
    ax.scatter(points[selected, 0], points[selected, 1], points[selected, 2], s=1.0, c=colors[selected], depthshade=False)
    ax.view_init(elev=30, azim=-55)
    ax.set_axis_off()
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max() / 2.0)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(0.0, max(center[2] + span * 0.75, 0.1))
    fig.tight_layout()
    fig.savefig(output_dir / "canopy_oblique.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def reconstruct_canopy(record_path: str | Path, config: CanopyReconstructionConfig | None = None) -> CanopyReconstructionResult:
    cfg = config or CanopyReconstructionConfig()
    root = Path(record_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    intrinsics = load_intrinsics(str(root / "kdc_intrinsics.txt"))
    if intrinsics is None:
        intrinsics = load_intrinsics(str(root / "kd_intrinsics.txt"))
    if intrinsics is None:
        raise FileNotFoundError("Canopy reconstruction requires kdc_intrinsics.txt or kd_intrinsics.txt.")
    K, _, _, _ = intrinsics

    output_dir = Path(cfg.output_dir) if cfg.output_dir else root / "canopy_local"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(cfg.mask_dir) if cfg.mask_dir else root / "reconstruction" / "masks"
    if mask_dir.exists():
        candidates = _load_mask_candidates(root, mask_dir, cfg)
        mask_source = str(mask_dir.resolve())
    elif cfg.auto_mask:
        auto_mask_dir = output_dir / "auto_masks"
        candidates = _load_auto_candidates(root, auto_mask_dir, cfg)
        mask_source = str(auto_mask_dir.resolve())
    else:
        raise FileNotFoundError(f"Mask directory does not exist and auto_mask is disabled: {mask_dir}")

    selected, reference_token = _select_frames(candidates, cfg)
    transforms, predicted_centers, image_shape = _estimate_alignment_transforms(selected, reference_token)
    canvas_transforms, canvas_size, canvas_offset = _build_canvas_transforms(
        transforms,
        image_shape,
        padding=max(0, int(cfg.canvas_padding)),
    )
    K_canvas = np.asarray(K, dtype=np.float64).copy()
    K_canvas[0, 2] += canvas_offset[0]
    K_canvas[1, 2] += canvas_offset[1]

    warped_depths = []
    warped_masks = []
    warped_rgbs = []
    reference_warped_rgb = None
    reference_warped_mask = None
    frame_info = []

    for item in selected:
        token = int(item["token"])
        rgb = cv2.imread(str(item["rgb_path"]))
        depth = cv2.imread(str(item["depth_path"]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise FileNotFoundError(f"Failed to load frame set for token {token}")

        clean_mask = _clean_mask(mask, keep_largest=False, min_component_area=cfg.min_component_area)
        filled_depth, depth_stats = _fill_depth_inside_mask(depth, clean_mask, cfg)
        warp = canvas_transforms[token]

        warped_depth = cv2.warpAffine(
            filled_depth,
            warp,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            clean_mask,
            warp,
            canvas_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_rgb = cv2.warpAffine(
            rgb.astype(np.float32),
            warp,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask_bool = warped_mask > 0
        warped_depth[~warped_mask_bool] = 0.0
        warped_rgb[~warped_mask_bool] = 0.0

        warped_depths.append(warped_depth)
        warped_masks.append(warped_mask_bool)
        warped_rgbs.append(warped_rgb)
        if token == reference_token:
            reference_warped_rgb = warped_rgb.copy()
            reference_warped_mask = warped_mask_bool.copy()
        frame_info.append(
            {
                "token": token,
                "mask_area": int(item["mask_area"]),
                "edge_fraction": float(item.get("edge_fraction", 0.0)),
                "alignment_shift_xy": [float(transforms[token][0, 2]), float(transforms[token][1, 2])],
                "predicted_center_xy": [float(v) for v in predicted_centers.get(token, (0.0, 0.0))],
                **depth_stats,
            }
        )

    coverage = np.sum(np.stack(warped_masks, axis=0), axis=0)
    stacked = np.stack([np.where(mask, depth, np.nan) for depth, mask in zip(warped_depths, warped_masks)], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused_depth = np.nanmedian(stacked, axis=0).astype(np.float32)
    fused_mask = coverage >= max(1, int(cfg.coverage_threshold))
    fused_mask_u8 = _clean_mask(
        fused_mask.astype(np.uint8) * 255,
        keep_largest=False,
        min_component_area=cfg.min_component_area,
    )
    fused_mask = fused_mask_u8 > 0

    valid = fused_mask & np.isfinite(fused_depth) & (fused_depth > 0)
    if not np.any(valid):
        raise RuntimeError("Canopy fusion produced an empty depth map.")
    fused_depth_full = _nearest_fill_in_mask(np.nan_to_num(fused_depth, nan=0.0), valid, fused_mask)
    fused_depth_full = _smooth_in_mask(fused_depth_full, fused_mask, sigma=cfg.smooth_sigma + 1.0)
    canopy = cv2.distanceTransform(fused_mask.astype(np.uint8), cv2.DIST_L2, 5)
    if canopy.max() > 0:
        fused_depth_full -= 45.0 * (canopy / canopy.max()).astype(np.float32)
    fused_depth_full[~fused_mask] = 0.0

    canvas_height, canvas_width = coverage.shape
    color_sum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    for warped_rgb, warped_mask in zip(warped_rgbs, warped_masks):
        color_sum += warped_rgb * warped_mask[..., None].astype(np.float32)
    fused_rgb = np.zeros_like(color_sum, dtype=np.float32)
    np.divide(color_sum, np.maximum(coverage[..., None], 1), out=fused_rgb, where=coverage[..., None] > 0)
    if reference_warped_rgb is not None and reference_warped_mask is not None:
        reference_valid = reference_warped_mask & fused_mask
        fused_rgb[reference_valid] = reference_warped_rgb[reference_valid]
    fused_rgb = np.clip(fused_rgb, 0, 255).astype(np.uint8)
    fused_rgb[~fused_mask] = 0

    crop_box = [0, 0, int(canvas_width), int(canvas_height)]
    if cfg.crop_to_mask:
        fused_depth_full, fused_mask, fused_rgb, K_canvas, crop_box = _crop_to_mask(
            fused_depth_full,
            fused_mask,
            fused_rgb,
            K_canvas,
            padding=max(8, int(cfg.canvas_padding) // 2),
        )
        fused_mask_u8 = (fused_mask.astype(np.uint8) * 255)

    masked_rgb_path = output_dir / "fused_rgb_masked.png"
    cv2.imwrite(str(masked_rgb_path), fused_rgb)
    cv2.imwrite(str(output_dir / "fused_mask.png"), fused_mask_u8)
    np.save(output_dir / "fused_depth_mm.npy", fused_depth_full)

    preview_depth = np.zeros(fused_depth_full.shape, dtype=np.uint8)
    inside = fused_depth_full[fused_mask]
    low, high = np.percentile(inside, [2, 98])
    scaled = np.clip((fused_depth_full - low) / max(high - low, 1e-6), 0.0, 1.0)
    preview_depth[fused_mask] = (255.0 * (1.0 - scaled[fused_mask])).astype(np.uint8)
    cv2.imwrite(str(output_dir / "fused_depth_vis.png"), preview_depth)

    pcd, mesh, reference_depth_m = _build_mesh_and_point_cloud(
        fused_depth_mm=fused_depth_full,
        fused_mask=fused_mask,
        fused_rgb_bgr=fused_rgb,
        K=K_canvas,
        z_scale=cfg.z_scale,
    )

    point_cloud_path = output_dir / "canopy_points.ply"
    mesh_path = output_dir / "canopy_mesh.ply"
    viewer_path = output_dir / "canopy_viewer.html"
    o3d.io.write_point_cloud(str(point_cloud_path), pcd)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    write_point_cloud_viewer(pcd, viewer_path, title=f"{root.name} canopy")
    _save_previews(output_dir, pcd)

    summary = {
        "record_path": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "point_cloud_path": str(point_cloud_path.resolve()),
        "mesh_path": str(mesh_path.resolve()),
        "viewer_path": str(viewer_path.resolve()),
        "masked_rgb_path": str(masked_rgb_path.resolve()),
        "mask_source": mask_source,
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "canvas_offset_xy": [float(canvas_offset[0]), float(canvas_offset[1])],
        "crop_box_xywh": crop_box,
        "frames_available": len(candidates),
        "frames_used": len(selected),
        "reference_token": reference_token,
        "reference_depth_m": reference_depth_m,
        "final_point_count": len(np.asarray(pcd.points)),
        "final_triangle_count": len(np.asarray(mesh.triangles)),
        "config": asdict(cfg),
        "frames": frame_info,
    }
    summary_path = output_dir / "canopy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return CanopyReconstructionResult(
        record_path=str(root.resolve()),
        output_dir=str(output_dir.resolve()),
        point_cloud_path=str(point_cloud_path.resolve()),
        mesh_path=str(mesh_path.resolve()),
        viewer_path=str(viewer_path.resolve()),
        masked_rgb_path=str(masked_rgb_path.resolve()),
        summary_path=str(summary_path.resolve()),
        frames_available=len(candidates),
        frames_used=len(selected),
        reference_token=reference_token,
        final_point_count=len(np.asarray(pcd.points)),
        final_triangle_count=len(np.asarray(mesh.triangles)),
    )
