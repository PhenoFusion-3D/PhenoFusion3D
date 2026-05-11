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

sys.path.insert(0, str(Path(__file__).parent))

from processing.canopy import (
    CanopyReconstructionConfig,
    _attach_candidate_positions,
    _discover_image_pairs,
    _frame_positions_m,
    _load_auto_candidates,
    reconstruct_canopy,
)
from visualiser.viewer import write_canopy_mesh_viewer


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


def _translate(geom, dx: float):
    if geom is not None and not geom.is_empty():
        geom.translate((float(dx), 0.0, 0.0), relative=True)
    return geom


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
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--reference-spacing-m", type=float, default=0.08)
    parser.add_argument("--min-score-ratio", type=float, default=0.35)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--min-mask-area", type=int, default=180000)
    parser.add_argument("--coverage", type=int, default=1)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--depth-min", type=int, default=500)
    parser.add_argument("--depth-max", type=int, default=4000)
    parser.add_argument("--leaf-thickness", type=float, default=0.003)
    parser.add_argument("--max-hole-fill-px", type=int, default=24)
    parser.add_argument("--max-triangle-jump", type=float, default=0.025)
    args = parser.parse_args()

    dataset = Path(args.input).resolve()
    if not dataset.exists():
        raise SystemExit(f"Input does not exist: {dataset}")
    args.output = Path(args.output).resolve() if args.output else dataset / "canopy_sequence"
    args.output.mkdir(parents=True, exist_ok=True)

    references, motion_info = _choose_references(dataset, args)
    if not references:
        raise SystemExit("No plant reference windows met the spacing/score thresholds.")
    print(f"[sequence] selected {len(references)} reference windows")

    rows = []
    combined_pcd = o3d.geometry.PointCloud()
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_display = o3d.geometry.TriangleMesh()
    origin_pos = float(references[0]["position_m"])

    for idx, ref in enumerate(references, start=1):
        token = int(ref["token"])
        name = f"plant_{idx:02d}_token_{token}"
        out_dir = args.output / name
        cfg = CanopyReconstructionConfig(
            output_dir=str(out_dir),
            sample_stride=args.stride,
            max_frames=args.max_frames,
            max_candidates=0,
            min_mask_area=args.min_mask_area,
            reference_token=token,
            coverage_threshold=args.coverage,
            smooth_sigma=args.smooth_sigma,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            add_leaf_thickness=args.leaf_thickness > 0,
            leaf_thickness_m=args.leaf_thickness if args.leaf_thickness > 0 else 0.003,
            max_hole_fill_distance_px=args.max_hole_fill_px,
            max_triangle_height_jump_m=args.max_triangle_jump,
        )
        print(f"[sequence] {name}")
        result = reconstruct_canopy(dataset, config=cfg)
        dx = float(ref["position_m"]) - origin_pos
        combined_pcd += _translate(_read_geometry(result.point_cloud_path, "pcd"), dx)
        combined_mesh += _translate(_read_geometry(result.mesh_path, "mesh"), dx)
        display_path = out_dir / "canopy_display_mesh.ply"
        if display_path.exists():
            combined_display += _translate(_read_geometry(str(display_path), "mesh"), dx)
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
        })

    combined_viewer = None
    if not combined_pcd.is_empty():
        o3d.io.write_point_cloud(str(args.output / "sequence_points.ply"), combined_pcd)
    if not combined_mesh.is_empty():
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.remove_degenerate_triangles()
        combined_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(args.output / "sequence_mesh.ply"), combined_mesh)
    if not combined_display.is_empty():
        combined_display.remove_duplicated_vertices()
        combined_display.remove_degenerate_triangles()
        combined_display.compute_vertex_normals()
        display_path = args.output / "sequence_display_mesh.ply"
        viewer_path = args.output / "sequence_viewer.html"
        o3d.io.write_triangle_mesh(str(display_path), combined_display)
        write_canopy_mesh_viewer(
            combined_display,
            viewer_path,
            title=f"{dataset.name} canopy sequence",
            point_cloud=combined_pcd,
            metadata={
                "Plants": len(rows),
                "Motion": motion_info.get("source", "unknown"),
                "Note": "Combined layout uses reference-position offsets for browsing.",
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
