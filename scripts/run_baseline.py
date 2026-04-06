#!/usr/bin/env python3
"""
Baseline reconstruction runner for plant data.

Runs two full reconstructions:
  - fx=1108  (placeholder / prior literature guess)
  - fx=900   (approximate sensor-class guess)

Outputs (relative to project root):
  baseline/baseline_fx1108.ply
  baseline/baseline_fx900.ply
  baseline/metrics.json

Usage:
  cd PhenoFusion3D
  python scripts/run_baseline.py [--step N] [--frames N]
"""

import os
import sys
import json
import shutil
import time
import argparse
import numpy as np
import open3d as o3d

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_io.loader import load_image_pairs
from processing.reconstructor import Reconstructor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = 'data/main/test_plant_rs13_1'
RGB_DIR   = os.path.join(DATA_DIR, 'rgb')
DEPTH_DIR = os.path.join(DATA_DIR, 'depth')
OUT_DIR   = 'baseline'
TMP_DIR   = os.path.join(OUT_DIR, '_tmp')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_K(fx, fy=None, cx=640.0, cy=360.0):
    """Build a 3×3 pinhole intrinsics matrix."""
    if fy is None:
        fy = fx
    return np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def run_reconstruction(label, K, pairs, depth_scale=1000.0, depth_trunc=3.2, voxel_size=0.005):
    """
    Run one full reconstruction and return (summary_dict, dst_ply_path).
    """
    print(f'\n{"=" * 62}')
    print(f'  RUN: {label}   fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}')
    print(f'  frames={len(pairs)}  depth_scale={depth_scale}  depth_trunc={depth_trunc}m')
    print(f'{"=" * 62}')

    tmp = os.path.join(TMP_DIR, label)
    os.makedirs(tmp, exist_ok=True)

    t_start = time.time()

    recon = Reconstructor(
        pairs=pairs,
        K=K,
        dist=None,           # no distortion for placeholder runs
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        voxel_size=voxel_size,
        save_path=tmp
    )
    final_pcd, succeed, fail = recon.run()

    elapsed = time.time() - t_start

    # --- Aggregate metrics ---
    total   = len(pairs)
    n_ok    = len(succeed)
    n_fail  = len(fail)
    pct_ok  = round(100.0 * n_ok / total, 2) if total > 0 else 0.0
    n_pts   = len(np.asarray(final_pcd.points)) if final_pcd and not final_pcd.is_empty() else 0

    # Exclude frame-0 (fitness=1.0 by convention) from ICP statistics
    icp_frames = [s for s in succeed if s['frame'] > 0]
    fitnesses  = [s['fitness'] for s in icp_frames]
    rmses      = [s['rmse']    for s in icp_frames]

    # Build merged per-frame list
    per_frame = []
    for s in succeed:
        per_frame.append({
            'frame':   s['frame'],
            'status':  'OK',
            'fitness': round(s['fitness'], 6),
            'rmse':    round(s['rmse'],    6)
        })
    for f in fail:
        per_frame.append({
            'frame':   f['frame'],
            'status':  'FAILED',
            'fitness': 0.0,
            'rmse':    0.0,
            'reason':  f.get('reason', '')
        })
    per_frame.sort(key=lambda x: x['frame'])

    summary = {
        'label':            label,
        'intrinsics': {
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2])
        },
        'reconstruction': {
            'total_frames':     total,
            'frames_succeeded': n_ok,
            'frames_failed':    n_fail,
            'pct_succeeded':    pct_ok,
            'total_points':     n_pts,
            'elapsed_s':        round(elapsed, 1)
        },
        'icp_stats': {
            'icp_frames_evaluated': len(icp_frames),
            'fitness_mean': round(float(np.mean(fitnesses)), 6) if fitnesses else 0.0,
            'fitness_min':  round(float(np.min(fitnesses)),  6) if fitnesses else 0.0,
            'fitness_max':  round(float(np.max(fitnesses)),  6) if fitnesses else 0.0,
            'fitness_std':  round(float(np.std(fitnesses)),  6) if fitnesses else 0.0,
            'rmse_mean':    round(float(np.mean(rmses)),     6) if rmses else 0.0,
            'rmse_max':     round(float(np.max(rmses)),      6) if rmses else 0.0,
            'rmse_std':     round(float(np.std(rmses)),      6) if rmses else 0.0,
        },
        'per_frame': per_frame
    }

    # --- Copy final PLY ---
    src_ply = os.path.join(tmp, 'merge_pcd_live.ply')
    dst_ply = os.path.join(OUT_DIR, f'baseline_{label}.ply')
    if os.path.exists(src_ply):
        shutil.copy2(src_ply, dst_ply)
        size_mb = os.path.getsize(dst_ply) / (1024 * 1024)
        print(f'[baseline] PLY → {dst_ply}  ({size_mb:.1f} MB)')
    else:
        print(f'[baseline] WARNING: merge_pcd_live.ply not found at {src_ply}')

    print(f'[baseline] {label}: {n_ok}/{total} frames OK ({pct_ok:.1f}%), '
          f'{n_pts:,} pts, fitness_mean={summary["icp_stats"]["fitness_mean"]:.4f}, '
          f'elapsed={elapsed:.0f}s')

    return summary, dst_ply


# ---------------------------------------------------------------------------
# Geometry inspection (non-interactive statistics)
# ---------------------------------------------------------------------------

def inspect_geometry(ply_path, label):
    """
    Load PLY and compute geometry statistics as objective proxies for
    visual quality (tilt, folding, noise, density).
    Returns a dict suitable for embedding in metrics.json.
    """
    if not os.path.exists(ply_path):
        return {'error': f'file not found: {ply_path}'}

    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        return {'error': 'empty point cloud'}

    pts = np.asarray(pcd.points)
    n   = len(pts)

    # Bounding box
    bb_min   = pts.min(axis=0)
    bb_max   = pts.max(axis=0)
    bb_ext   = bb_max - bb_min          # [dx, dy, dz] in metres
    centroid = pts.mean(axis=0)

    # PCA → dominant axes (proxy for structural tilt)
    centered  = pts - centroid
    cov       = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order     = eigvals.argsort()[::-1]
    eigvals   = eigvals[order]
    eigvecs   = eigvecs[:, order]       # columns are principal axes

    # Tilt: angle between the primary axis and the world Y-up direction
    primary_axis = eigvecs[:, 0]
    cos_y = float(np.clip(abs(primary_axis[1]), 0.0, 1.0))
    tilt_from_Y_deg = float(np.degrees(np.arccos(cos_y)))

    # Explained variance ratio
    total_var = eigvals.sum()
    var_ratio = (eigvals / total_var).tolist() if total_var > 0 else [0, 0, 0]

    # Folding proxy: ratio of Z-spread to XY-spread
    xy_spread = float(np.sqrt(bb_ext[0] ** 2 + bb_ext[1] ** 2))
    z_spread  = float(bb_ext[2])
    z_xy_ratio = round(z_spread / xy_spread, 3) if xy_spread > 0 else 0.0

    # Noise proxy: residual distance std after removing first PC
    proj_on_pc1 = (centered @ primary_axis)[:, np.newaxis] * primary_axis
    residuals   = centered - proj_on_pc1
    noise_std   = float(np.linalg.norm(residuals, axis=1).std())

    # Statistical outlier count (approximate, using radius 2cm, min 5 neighbours)
    try:
        _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        n_outliers = n - len(inlier_idx)
        pct_outlier = round(100.0 * n_outliers / n, 2)
    except Exception:
        n_outliers  = -1
        pct_outlier = -1.0

    # --- Auto visual notes ---
    notes = []
    if tilt_from_Y_deg > 35:
        notes.append(f'significant tilt: primary axis {tilt_from_Y_deg:.1f}° from Y-up — '
                     'likely wrong fx causing perspective distortion')
    elif tilt_from_Y_deg > 15:
        notes.append(f'moderate tilt: {tilt_from_Y_deg:.1f}° from Y-up')

    if z_xy_ratio > 1.5:
        notes.append(f'depth spread large relative to footprint (z/xy={z_xy_ratio:.2f}) — '
                     'possible ICP drift / folding along camera axis')

    if pct_outlier > 10:
        notes.append(f'high outlier rate: {pct_outlier:.1f}% of points are statistical outliers')

    if var_ratio[0] > 0.85:
        notes.append('reconstruction collapsed to near-planar structure (PC1 explains '
                     f'{var_ratio[0]*100:.0f}% variance) — likely registration failure')

    if n < 50_000:
        notes.append(f'very low point count ({n:,}) — possible sparse / heavily failed reconstruction')

    if not notes:
        notes.append('geometry appears nominal — no obvious tilt, folding, or excess noise')

    return {
        'n_points': n,
        'centroid_xyz_m': [round(float(x), 4) for x in centroid],
        'bounding_box_extent_xyz_m': [round(float(x), 4) for x in bb_ext],
        'bounding_box_volume_m3': round(float(bb_ext[0] * bb_ext[1] * bb_ext[2]), 4),
        'z_to_xy_spread_ratio': z_xy_ratio,
        'primary_axis_tilt_deg_from_Y': round(tilt_from_Y_deg, 2),
        'pca_explained_variance_ratio': [round(float(v), 4) for v in var_ratio],
        'residual_noise_std_m': round(noise_std, 5),
        'statistical_outlier_count': n_outliers,
        'statistical_outlier_pct': pct_outlier,
        'visual_notes': notes
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run baseline reconstructions')
    parser.add_argument('--step',   type=int, default=1,
                        help='Frame step size for loader (default: 1 = every frame)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Limit to first N pairs (default: all)')
    parser.add_argument('--no-inspect', action='store_true',
                        help='Skip geometry inspection step')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # --- Load pairs once ---
    print(f'[baseline] Loading image pairs from {RGB_DIR} ...')
    pairs = load_image_pairs(RGB_DIR, DEPTH_DIR, step=args.step)
    if args.frames:
        pairs = pairs[:args.frames]
        print(f'[baseline] Capped to first {len(pairs)} pairs.')

    # --- Define runs ---
    runs = [
        ('fx1108', build_K(fx=1108.0, cx=640.0, cy=360.0)),
        ('fx900',  build_K(fx=900.0,  cx=640.0, cy=360.0)),
    ]

    all_metrics = {}
    ply_paths   = {}

    for label, K in runs:
        summary, ply_path = run_reconstruction(label, K, pairs)
        all_metrics[label] = summary
        ply_paths[label]   = ply_path

    # --- Geometry inspection ---
    if not args.no_inspect:
        print(f'\n[baseline] Computing geometry statistics for visual inspection...')
        for label, ply_path in ply_paths.items():
            print(f'  Inspecting {label} ...')
            stats = inspect_geometry(ply_path, label)
            all_metrics[label]['geometry_inspection'] = stats
            print(f'  → {label} notes: {stats.get("visual_notes", [])}')

    # --- Write metrics.json ---
    metrics_path = os.path.join(OUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\n[baseline] metrics.json written → {metrics_path}')

    # --- Summary table ---
    print(f'\n{"─" * 62}')
    print(f'  BASELINE SUMMARY')
    print(f'{"─" * 62}')
    for label, m in all_metrics.items():
        r  = m['reconstruction']
        ic = m['icp_stats']
        gi = m.get('geometry_inspection', {})
        print(f'  {label}:')
        print(f'    frames: {r["frames_succeeded"]}/{r["total_frames"]} OK '
              f'({r["pct_succeeded"]:.1f}%)')
        print(f'    points: {r["total_points"]:,}')
        print(f'    ICP fitness: mean={ic["fitness_mean"]:.4f}  '
              f'min={ic["fitness_min"]:.4f}  std={ic["fitness_std"]:.4f}')
        print(f'    ICP rmse:    mean={ic["rmse_mean"]:.4f}  max={ic["rmse_max"]:.4f}')
        if gi:
            print(f'    tilt from Y: {gi.get("primary_axis_tilt_deg_from_Y", "?")}°')
            print(f'    z/xy ratio:  {gi.get("z_to_xy_spread_ratio", "?")}')
            for note in gi.get('visual_notes', []):
                print(f'    NOTE: {note}')
    print(f'{"─" * 62}')
    print(f'\n[baseline] All done.')
    print(f'  PLY files: baseline/baseline_fx1108.ply  baseline/baseline_fx900.ply')
    print(f'  Metrics:   baseline/metrics.json')


if __name__ == '__main__':
    main()
