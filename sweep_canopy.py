"""
sweep_canopy.py – Canopy reconstruction parameter sweep and evaluation tool.

Runs reconstruct_canopy() with a grid of parameter combinations and generates
a self-contained HTML report with side-by-side preview images, statistics, and
a recommended best-run table.

Usage
-----
    python sweep_canopy.py --dataset data/main/test_plant_20230809133659

    # Full grid (slower):
    python sweep_canopy.py --dataset data/main/test_plant_20230809133659 --full

    # Custom sweep:
    python sweep_canopy.py --dataset data/main/test_plant_20230809133659 \
        --max-frames 5 9 15 \
        --smooth-sigma 2.0 3.5 6.0 \
        --coverage 1 2

Output
------
A folder  <dataset>/sweep_<timestamp>/  containing:
  run_*/          – per-run output from reconstruct_canopy()
  sweep_report.html  – self-contained HTML comparison

Dependencies
------------
  open3d, opencv-python, numpy, scipy, matplotlib  (all already needed by the pipeline)
"""

import argparse
import base64
import json
import os
import shutil
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).parent))

from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy


# ---------------------------------------------------------------------------
# Sweep grid definitions
# ---------------------------------------------------------------------------

QUICK_GRID = {
    "max_frames":          [5, 9, 15],
    "smooth_sigma":        [2.0, 3.5, 6.0],
    "coverage_threshold":  [1],
}

FULL_GRID = {
    "max_frames":          [5, 9, 15, 20],
    "smooth_sigma":        [2.0, 3.5, 5.0, 7.0],
    "coverage_threshold":  [1, 2],
}

# Mask sensitivity presets (applied on top of base config)
MASK_PRESETS = {
    "default": {},
    # Uncomment below to add more presets to the sweep:
    # "loose":   {"mask_s_min": 25, "mask_v_min": 20, "mask_exg_min": 10},
    # "strict":  {"mask_s_min": 65, "mask_v_min": 55, "mask_exg_min": 35},
}


# ---------------------------------------------------------------------------
# Helper: embed image as base-64 data URI
# ---------------------------------------------------------------------------

def _img_to_data_uri(path: Path, max_w: int = 480) -> str:
    if not path.exists():
        return ""
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _depth_img_to_data_uri(path: Path, max_w: int = 480) -> str:
    """Colourmap a 8-bit depth-preview PNG to a data URI."""
    if not path.exists():
        return ""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    colour = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    h, w = colour.shape[:2]
    if w > max_w:
        scale = max_w / w
        colour = cv2.resize(colour, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", colour, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one(
    dataset_root: Path,
    run_id: str,
    sweep_dir: Path,
    params: dict,
    base_cfg_overrides: dict,
) -> dict:
    """Run reconstruct_canopy for one parameter combination.

    Returns a result-info dict with statistics and image paths for the report.
    """
    run_dir = sweep_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_kwargs = dict(
        output_dir=str(run_dir),
        auto_mask=True,
        **base_cfg_overrides,
        **params,
    )
    cfg = CanopyReconstructionConfig(**cfg_kwargs)

    info: dict = {
        "run_id":   run_id,
        "params":   params,
        "run_dir":  str(run_dir),
        "success":  False,
        "error":    "",
        "duration_s": 0.0,
    }

    t0 = time.time()
    try:
        result = reconstruct_canopy(dataset_root, config=cfg)
        elapsed = time.time() - t0

        info.update({
            "success":            True,
            "duration_s":         round(elapsed, 1),
            "frames_used":        result.frames_used,
            "frames_available":   result.frames_available,
            "point_count":        result.final_point_count,
            "triangle_count":     result.final_triangle_count,
            "summary_path":       result.summary_path,
            "viewer_path":        result.viewer_path,
            "fused_rgb_path":     str(run_dir / "fused_rgb_masked.png"),
            "fused_depth_path":   str(run_dir / "fused_depth_vis.png"),
            "confidence_path":    str(run_dir / "fused_confidence.png"),
            "oblique_path":       str(run_dir / "canopy_oblique.png"),
            "mask_path":          str(run_dir / "fused_mask.png"),
            "mosaic_path":        str(run_dir / "selected_frames_mosaic.jpg"),
        })
        print(
            f"  [{run_id}] OK  {result.frames_used}/{result.frames_available} frames, "
            f"{result.final_point_count:,} pts, {elapsed:.0f}s"
        )
    except Exception as exc:
        elapsed = time.time() - t0
        info["error"] = str(exc)
        info["duration_s"] = round(elapsed, 1)
        print(f"  [{run_id}] FAIL  {exc}")

    return info


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; background:#1a1a2e; color:#e0e0e0; margin:0; padding:16px; }
h1   { color:#a0d8ef; margin-bottom:4px; }
h2   { color:#7ec8e3; margin-top:24px; }
.meta { color:#888; font-size:0.85em; margin-bottom:20px; }
table { border-collapse:collapse; width:100%; margin-top:12px; }
th, td { padding:8px 10px; border:1px solid #333; text-align:center; font-size:0.82em; }
th     { background:#162032; color:#a0d8ef; }
tr:hover td { background:#1e2d3d; }
.best  { background:#0d2e1a !important; color:#6fcf97; font-weight:bold; }
.fail  { color:#e57373; }
.card  { display:inline-block; vertical-align:top; margin:6px; background:#162032; border-radius:6px; padding:8px; max-width:520px; }
.card img { max-width:100%; border-radius:4px; margin-top:4px; }
.card-title { font-size:0.8em; color:#a0d8ef; margin-bottom:2px; }
.run-label  { font-size:0.75em; color:#888; }
.section-imgs { margin-top:8px; display:flex; flex-wrap:wrap; }
"""

_JS = """
function sortTable(n) {
  var t = document.getElementById('results-table');
  var rows = Array.from(t.querySelectorAll('tbody tr'));
  var asc = t.dataset.sortDir !== '1';
  t.dataset.sortDir = asc ? '1' : '0';
  rows.sort(function(a, b) {
    var va = a.cells[n].textContent.trim();
    var vb = b.cells[n].textContent.trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  rows.forEach(function(r) { t.querySelector('tbody').appendChild(r); });
}
"""


def _build_report(sweep_dir: Path, runs: list[dict], dataset: str, grid: dict) -> Path:
    """Generate sweep_report.html and return its path."""
    report_path = sweep_dir / "sweep_report.html"

    successes = [r for r in runs if r["success"]]

    # Best run by point count (as a simple proxy for reconstruction quality)
    best_id = None
    if successes:
        best = max(successes, key=lambda r: r.get("point_count", 0))
        best_id = best["run_id"]

    col_headers = [
        ("Run", "run_id"),
        ("max_frames", "params.max_frames"),
        ("smooth_sigma", "params.smooth_sigma"),
        ("coverage", "params.coverage_threshold"),
        ("Frames used", "frames_used"),
        ("Points", "point_count"),
        ("Triangles", "triangle_count"),
        ("Time (s)", "duration_s"),
        ("Status", "_status"),
    ]

    def _cell(run, key):
        if "." in key:
            parts = key.split(".")
            val = run
            for p in parts:
                val = val.get(p, "")
            return val
        if key == "_status":
            return "OK" if run["success"] else f"FAIL: {run['error'][:60]}"
        return run.get(key, "")

    rows_html = []
    for run in runs:
        cls = "best" if run["run_id"] == best_id else ("fail" if not run["success"] else "")
        cells = "".join(
            f'<td class="{cls}">{_cell(run, k)}</td>'
            for _, k in col_headers
        )
        rows_html.append(f"<tr>{cells}</tr>")

    table_html = (
        "<table id='results-table'>"
        "<thead><tr>"
        + "".join(
            f'<th onclick="sortTable({i})" style="cursor:pointer">{h} &#9661;</th>'
            for i, (h, _) in enumerate(col_headers)
        )
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )

    # Preview cards (only successful runs)
    cards_html = []
    for run in successes:
        fused_uri   = _img_to_data_uri(Path(run.get("fused_rgb_path", "")))
        depth_uri   = _depth_img_to_data_uri(Path(run.get("fused_depth_path", "")))
        conf_uri    = _depth_img_to_data_uri(Path(run.get("confidence_path", "")))
        oblique_uri = _img_to_data_uri(Path(run.get("oblique_path", "")))
        mask_uri    = _img_to_data_uri(Path(run.get("mask_path", "")))
        mosaic_uri  = _img_to_data_uri(Path(run.get("mosaic_path", "")))
        viewer_rel = os.path.relpath(run.get("viewer_path", ""), start=sweep_dir)

        best_badge = " &#9733; BEST" if run["run_id"] == best_id else ""
        cards_html.append(
            f'<div class="card">'
            f'<div class="card-title">{run["run_id"]}{best_badge}</div>'
            f'<div class="run-label">'
            f'max_frames={run["params"].get("max_frames")} '
            f'smooth_sigma={run["params"].get("smooth_sigma")} '
            f'coverage={run["params"].get("coverage_threshold")}'
            f'</div>'
            f'<div class="run-label">'
            f'{run["frames_used"]}/{run["frames_available"]} frames &bull; '
            f'{run["point_count"]:,} pts &bull; {run["duration_s"]}s'
            f'</div>'
            + (f'<div class="card-title" style="margin-top:6px">Fused RGB</div>'
               f'<img src="{fused_uri}" alt="fused rgb" />' if fused_uri else "")
            + (f'<div class="card-title">Selected frames</div>'
               f'<img src="{mosaic_uri}" alt="selected frames" />' if mosaic_uri else "")
            + (f'<div class="card-title">Depth preview</div>'
               f'<img src="{depth_uri}" alt="depth" />' if depth_uri else "")
            + (f'<div class="card-title">Depth confidence</div>'
               f'<img src="{conf_uri}" alt="confidence" />' if conf_uri else "")
            + (f'<div class="card-title">Oblique 3-D view</div>'
               f'<img src="{oblique_uri}" alt="oblique" />' if oblique_uri else "")
            + (f'<div class="card-title">Fused mask</div>'
               f'<img src="{mask_uri}" alt="mask" />' if mask_uri else "")
            + (f'<div class="run-label" style="margin-top:6px">'
               f'<a href="{viewer_rel}" style="color:#7ec8e3">Open mesh viewer</a></div>'
               if run.get("viewer_path") else "")
            + "</div>"
        )

    grid_str = json.dumps(grid, indent=2)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>Canopy Sweep Report – {dataset}</title>
<style>{_CSS}</style>
<script>{_JS}</script>
</head><body>
<h1>Canopy Reconstruction Sweep</h1>
<div class="meta">
  Dataset: <strong>{dataset}</strong> &bull;
  Generated: {ts} &bull;
  {len(runs)} runs ({len(successes)} succeeded)
</div>

<h2>Parameter Grid</h2>
<pre style="background:#162032;padding:10px;border-radius:4px;font-size:0.8em;">{grid_str}</pre>

<h2>Results (click column header to sort)</h2>
{table_html}

<h2>Preview Cards</h2>
<div class="section-imgs">
{"".join(cards_html) or "<p>No successful runs to preview.</p>"}
</div>

</body></html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sweep canopy reconstruction parameters and generate an HTML report."
    )
    ap.add_argument(
        "--dataset", required=True,
        help="Path to the dataset root (contains rgb_*.png or rgb/ subfolder)."
    )
    ap.add_argument(
        "--output", default=None,
        help="Output directory for the sweep (default: <dataset>/sweep_<timestamp>/)."
    )
    ap.add_argument(
        "--full", action="store_true",
        help="Run the full parameter grid instead of the quick grid."
    )
    ap.add_argument(
        "--max-frames", nargs="+", type=int, default=None, metavar="N",
        help="Override max_frames values to sweep (e.g. --max-frames 5 9 15)."
    )
    ap.add_argument(
        "--smooth-sigma", nargs="+", type=float, default=None, metavar="S",
        help="Override smooth_sigma values to sweep (e.g. --smooth-sigma 2.0 3.5 6.0)."
    )
    ap.add_argument(
        "--coverage", nargs="+", type=int, default=None, metavar="C",
        help="Override coverage_threshold values (e.g. --coverage 1 2)."
    )
    ap.add_argument(
        "--stride", type=int, default=1, metavar="N",
        help="Sample every Nth frame (default: 1 = all frames)."
    )
    ap.add_argument(
        "--depth-min", type=int, default=500,
        help="Near-clip depth (mm, default: 500)."
    )
    ap.add_argument(
        "--depth-max", type=int, default=4000,
        help="Far-clip depth (mm, default: 4000)."
    )
    ap.add_argument(
        "--leaf-thickness", type=float, default=0.003,
        help="Display-only back-face/skirt thickness in metres (0 = disabled)."
    )
    ap.add_argument(
        "--max-hole-fill-px", type=int, default=24,
        help="Maximum inpaint distance from real depth in pixels."
    )
    ap.add_argument(
        "--max-triangle-jump", type=float, default=0.025,
        help="Maximum height jump for neighbouring mesh triangles in metres."
    )
    ap.add_argument(
        "--no-cleanup", action="store_true",
        help="Disable statistical outlier removal and density mesh trimming."
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset path does not exist: {dataset_root}")
        sys.exit(1)

    # Build sweep grid
    grid = FULL_GRID if args.full else QUICK_GRID
    if args.max_frames is not None:
        grid["max_frames"] = args.max_frames
    if args.smooth_sigma is not None:
        grid["smooth_sigma"] = args.smooth_sigma
    if args.coverage is not None:
        grid["coverage_threshold"] = args.coverage

    combos = list(product(
        grid["max_frames"],
        grid["smooth_sigma"],
        grid["coverage_threshold"],
    ))

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(args.output).resolve()
        if args.output
        else dataset_root / f"sweep_{ts_str}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_overrides = {
        "sample_stride":    args.stride,
        "depth_min":        args.depth_min,
        "depth_max":        args.depth_max,
        "mesh_cleanup":     not args.no_cleanup,
        "add_leaf_thickness": args.leaf_thickness > 0,
        "leaf_thickness_m": args.leaf_thickness if args.leaf_thickness > 0 else 0.003,
        "max_hole_fill_distance_px": args.max_hole_fill_px,
        "max_triangle_height_jump_m": args.max_triangle_jump,
    }

    print(f"\nCanopy parameter sweep on: {dataset_root.name}")
    print(f"  Sweep dir : {sweep_dir}")
    print(f"  Grid size : {len(combos)} combinations\n")

    runs = []
    for i, (mf, ss, cov) in enumerate(combos, start=1):
        run_id = f"run_{i:02d}_mf{mf}_ss{ss}_cov{cov}"
        params = {
            "max_frames":         mf,
            "smooth_sigma":       ss,
            "coverage_threshold": cov,
        }
        print(f"[{i}/{len(combos)}] {run_id}  params={params}")
        info = _run_one(dataset_root, run_id, sweep_dir, params, base_cfg_overrides)
        runs.append(info)

    # Save machine-readable summary
    summary_path = sweep_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    # HTML report
    report_path = _build_report(sweep_dir, runs, dataset_root.name, grid)

    successes = [r for r in runs if r["success"]]
    print(f"\n{'='*60}")
    print(f"Sweep complete.  {len(successes)}/{len(runs)} runs succeeded.")
    if successes:
        best = max(successes, key=lambda r: r.get("point_count", 0))
        print(f"Best run (most points): {best['run_id']}")
        print(f"  max_frames={best['params']['max_frames']}  "
              f"smooth_sigma={best['params']['smooth_sigma']}  "
              f"coverage={best['params']['coverage_threshold']}")
        print(f"  {best['point_count']:,} points, {best['frames_used']} frames used")
    print(f"\nReport: {report_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
