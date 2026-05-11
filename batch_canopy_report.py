"""
Run the repaired canopy reconstruction once per dataset and build an HTML report.

Example:
    python batch_canopy_report.py --root data/main
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy


def _has_data(path: Path) -> bool:
    return any(path.glob("rgb_*.png")) or (path / "rgb").is_dir()


def _discover(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and _has_data(p))


def _img_uri(path: Path, max_w: int = 420, colorize: bool = False) -> str:
    if not path.exists():
        return ""
    flag = cv2.IMREAD_GRAYSCALE if colorize else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        return ""
    if colorize:
        img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _run_one(dataset: Path, out_root: Path, args) -> dict:
    out_dir = out_root / dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CanopyReconstructionConfig(
        output_dir=str(out_dir),
        sample_stride=args.stride,
        max_frames=args.max_frames,
        max_candidates=args.max_candidates,
        coverage_threshold=args.coverage,
        smooth_sigma=args.smooth_sigma,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        add_leaf_thickness=args.leaf_thickness > 0,
        leaf_thickness_m=args.leaf_thickness if args.leaf_thickness > 0 else 0.003,
        max_hole_fill_distance_px=args.max_hole_fill_px,
        max_triangle_height_jump_m=args.max_triangle_jump,
        use_poisson_mesh=args.poisson,
    )

    info = {
        "dataset": dataset.name,
        "dataset_path": str(dataset),
        "output_dir": str(out_dir),
        "success": False,
        "error": "",
        "duration_s": 0.0,
    }
    start = time.time()
    try:
        result = reconstruct_canopy(dataset, config=cfg)
        info.update({
            "success": True,
            "duration_s": round(time.time() - start, 1),
            "frames_used": result.frames_used,
            "frames_available": result.frames_available,
            "points": result.final_point_count,
            "triangles": result.final_triangle_count,
            "viewer_path": result.viewer_path,
            "summary_path": result.summary_path,
        })
    except Exception as exc:
        info["duration_s"] = round(time.time() - start, 1)
        info["error"] = str(exc)
    return info


def _write_report(out_root: Path, rows: list[dict]) -> Path:
    cards = []
    for row in rows:
        out_dir = Path(row["output_dir"])
        status = "OK" if row["success"] else "FAIL"
        viewer = ""
        if row.get("viewer_path"):
            rel = os.path.relpath(row["viewer_path"], start=out_root)
            viewer = f'<a href="{html.escape(rel)}">Open mesh viewer</a>'
        if row["success"]:
            rgb = _img_uri(out_dir / "fused_rgb_masked.png")
            depth = _img_uri(out_dir / "fused_depth_vis.png", colorize=True)
            conf = _img_uri(out_dir / "fused_confidence.png", colorize=True)
            mosaic = _img_uri(out_dir / "selected_frames_mosaic.jpg")
            oblique = _img_uri(out_dir / "canopy_oblique.png")
            images = []
            if rgb:
                images.append(f'<img src="{rgb}" alt="fused rgb">')
            if mosaic:
                images.append(f'<img src="{mosaic}" alt="selected frames">')
            if depth:
                images.append(f'<img src="{depth}" alt="depth">')
            if conf:
                images.append(f'<img src="{conf}" alt="confidence">')
            if oblique:
                images.append(f'<img src="{oblique}" alt="oblique">')
            body = (
                f'<div class="meta">{row["frames_used"]}/{row["frames_available"]} frames, '
                f'{row["points"]:,} points, {row["triangles"]:,} triangles, '
                f'{row["duration_s"]}s</div>'
                f'<div class="meta">{viewer}</div>'
                f'{"".join(images)}'
            )
        else:
            body = f'<div class="fail">{html.escape(row["error"])}</div>'
        cards.append(
            f'<section class="card">'
            f'<h2>{html.escape(row["dataset"])} <span class="{status.lower()}">{status}</span></h2>'
            f'{body}</section>'
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    css = """
body{margin:0;padding:18px;background:#111827;color:#e5e7eb;font-family:Segoe UI,Arial,sans-serif}
h1{margin:0 0 4px;color:#bfdbfe}.meta{color:#9ca3af;font-size:13px;margin:5px 0 10px}
.grid{display:flex;flex-wrap:wrap;gap:14px}.card{background:#172033;border:1px solid #2d3748;border-radius:6px;padding:12px;max-width:460px}
h2{font-size:16px;margin:0 0 8px;color:#dbeafe}.ok{color:#86efac}.fail,.error{color:#fca5a5}
img{display:block;max-width:100%;border-radius:4px;margin-top:8px}a{color:#93c5fd}
"""
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Canopy Batch Report</title>
<style>{css}</style></head><body>
<h1>Canopy Batch Report</h1>
<div class="meta">Generated {html.escape(ts)}. {sum(r["success"] for r in rows)}/{len(rows)} succeeded.</div>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    report = out_root / "batch_report.html"
    report.write_text(doc, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canopy reconstruction across all datasets.")
    parser.add_argument("--root", default="data/main", help="Parent folder containing dataset subfolders.")
    parser.add_argument("--output", default=None, help="Batch output folder.")
    parser.add_argument("--stride", type=int, default=1, help="Candidate stride, default all frames.")
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--coverage", type=int, default=1)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--depth-min", type=int, default=500)
    parser.add_argument("--depth-max", type=int, default=4000)
    parser.add_argument("--leaf-thickness", type=float, default=0.003)
    parser.add_argument("--max-hole-fill-px", type=int, default=24)
    parser.add_argument("--max-triangle-jump", type=float, default=0.025)
    parser.add_argument("--poisson", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")
    datasets = _discover(root)
    if not datasets:
        raise SystemExit(f"No datasets found under {root}")

    out_root = (
        Path(args.output).resolve()
        if args.output else root / f"canopy_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, dataset in enumerate(datasets, start=1):
        print(f"[{i}/{len(datasets)}] {dataset.name}")
        row = _run_one(dataset, out_root, args)
        rows.append(row)
        print("  OK" if row["success"] else f"  FAIL: {row['error']}")

    (out_root / "batch_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = _write_report(out_root, rows)
    print(f"\nBatch report: {report}")


if __name__ == "__main__":
    main()
