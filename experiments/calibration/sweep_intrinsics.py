#!/usr/bin/env python3
"""
Phase 1 – Manual focal-length sweep + full reconstruction with winner.

Per-fx metrics (anchor frame 50, neighbours ±4):
  - mean ICP fitness + RMSE  (source→anchor registration)
  - RANSAC plane inlier ratio (floor region: raw depth 3200–4000 mm)
      The floor is a known flat surface.  Correct fx projects floor pixels
      into a tight plane; wrong fx distorts the XY scale so the floor
      curves → fewer RANSAC inliers.

Combined score = plane_inlier_ratio * 0.6 + mean_fitness * 0.4
  (plane_inlier_ratio weighted higher — it is the physically grounded metric)

Steps executed:
  1. Sweep fx in FX_VALUES, compute metrics, print live table
  2. Write intrinsics_sweep.csv
  3. Pick best fx, write candidate_intrinsics.json
  4. Run full reconstruction with best fx, save candidate_reconstruction.ply

Outputs:
  experiments/calibration/intrinsics_sweep.csv
  experiments/calibration/candidate_intrinsics.json
  experiments/calibration/candidate_reconstruction.ply

Usage (from project root):
  python experiments/calibration/sweep_intrinsics.py
"""

import os, sys, json, csv, copy
import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from file_io.loader     import load_image_pairs
from file_io.exporter   import save_ply
from processing.rgbd    import rgbd2pcd
from processing.icp     import color_icp
from processing.utils   import clean_pcd
from processing.reconstructor import Reconstructor

# ── config ───────────────────────────────────────────────────────────────────
RGB_DIR   = 'data/main/test_plant_rs13_1/rgb'
DEPTH_DIR = 'data/main/test_plant_rs13_1/depth'
OUT_DIR   = 'experiments/calibration'

FX_VALUES   = [700, 800, 900, 1000, 1050, 1108, 1200]
ANCHOR      = 50       # anchor frame for sweep metrics
N_NEIGHBORS = 4        # register anchor ± N neighbours

DEPTH_SCALE   = 1000.0
DEPTH_TRUNC   = 3.2    # metres — plant region cutoff (used for ICP)
VOXEL_SIZE    = 0.005  # metres

# Floor region for plane fitting (from depth histogram)
FLOOR_MIN_MM  = 3200   # floor starts after plant ends
FLOOR_MAX_MM  = 4000   # floor pixels confirmed in histogram (130k)
PLANE_DIST_THR = 0.03  # RANSAC inlier threshold (metres)
PLANE_ITERS    = 1000

# Score weights
W_PLANE   = 0.6
W_FITNESS = 0.4
# ─────────────────────────────────────────────────────────────────────────────


def build_K(fx, width=1280, height=720):
    return np.array(
        [[fx, 0.0, width / 2.0],
         [0.0, fx, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64
    )


def load_raw(rgb_path, depth_path):
    color = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    return color, depth


def depth_to_pcd_raw(color, depth_raw, K, d_min_mm, d_max_mm):
    """
    Project pixels whose raw depth is in [d_min_mm, d_max_mm] to 3-D.
    No inpainting — zero pixels become genuine no-data.
    """
    depth = depth_raw.copy()
    depth[(depth < d_min_mm) | (depth > d_max_mm)] = 0

    h, w  = color.shape[:2]
    fx    = float(K[0, 0]);  fy = float(K[1, 1])
    cx    = float(K[0, 2]);  cy = float(K[1, 2])

    o3d_color = o3d.geometry.Image(color.astype(np.uint8))
    o3d_depth = o3d.geometry.Image(depth.astype(np.uint16))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=DEPTH_SCALE,
        depth_trunc=d_max_mm / DEPTH_SCALE + 0.1,
        convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


def make_plant_pcd(color, depth_raw, K):
    """Standard plant cloud (with inpainting) for ICP."""
    return clean_pcd(
        rgbd2pcd(color, depth_raw, K, dist=None,
                 depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC),
        voxel_size=VOXEL_SIZE
    )


# ── RANSAC plane metric ───────────────────────────────────────────────────────

def plane_inlier_ratio(color, depth_raw, K):
    """
    Project floor pixels (FLOOR_MIN_MM–FLOOR_MAX_MM) and fit a RANSAC plane.
    Returns inlier_ratio in [0, 1].  Returns 0.0 if too few floor points.
    """
    floor_pcd = depth_to_pcd_raw(color, depth_raw, K, FLOOR_MIN_MM, FLOOR_MAX_MM)

    n = len(floor_pcd.points)
    if n < 50:
        return 0.0   # not enough floor pixels in this frame

    try:
        _, inliers = floor_pcd.segment_plane(
            distance_threshold=PLANE_DIST_THR,
            ransac_n=3,
            num_iterations=PLANE_ITERS
        )
        return round(len(inliers) / n, 4)
    except Exception:
        return 0.0


# ── ICP metric ────────────────────────────────────────────────────────────────

def icp_fitness_rmse(anchor_pcd, pairs, anchor_idx, K):
    """Register anchor ± N_NEIGHBORS. Returns (mean_fitness, mean_rmse)."""
    nbr_indices = [i for i in range(
        max(0, anchor_idx - N_NEIGHBORS),
        min(len(pairs), anchor_idx + N_NEIGHBORS + 1)
    ) if i != anchor_idx]

    fitnesses, rmses = [], []
    for ni in nbr_indices:
        color_n, depth_n = load_raw(*pairs[ni])
        src = make_plant_pcd(color_n, depth_n, K)
        if src.is_empty():
            continue
        try:
            _, _, fit, rmse = color_icp(src, anchor_pcd, voxel_size=VOXEL_SIZE)
            fitnesses.append(fit)
            rmses.append(rmse)
        except Exception:
            pass

    mean_fit  = float(np.mean(fitnesses)) if fitnesses else 0.0
    mean_rmse = float(np.mean(rmses))     if rmses     else 0.0
    return round(mean_fit, 6), round(mean_rmse, 6)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(pairs):
    anchor_idx            = min(ANCHOR, len(pairs) - 1)
    color_a, depth_a      = load_raw(*pairs[anchor_idx])

    print(f'\n  {"fx":>6}  {"fitness":>8}  {"rmse":>8}  {"plane_inlier":>13}  {"score":>8}')
    print(f'  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*13}  {"─"*8}')

    rows = []
    for fx in FX_VALUES:
        K = build_K(fx)

        # ICP metric
        anchor_pcd         = make_plant_pcd(color_a, depth_a, K)
        mean_fit, mean_rmse = icp_fitness_rmse(anchor_pcd, pairs, anchor_idx, K)

        # RANSAC plane metric (floor pixels, no inpainting)
        p_ratio = plane_inlier_ratio(color_a, depth_a, K)

        score = round(W_PLANE * p_ratio + W_FITNESS * mean_fit, 6)

        rows.append({
            'fx':                fx,
            'mean_fitness':      mean_fit,
            'mean_rmse':         mean_rmse,
            'plane_inlier_ratio': p_ratio,
            'score':             score,
        })
        print(f'  {fx:>6}  {mean_fit:>8.4f}  {mean_rmse:>8.4f}  {p_ratio:>13.4f}  {score:>8.4f}')

    print(f'  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*13}  {"─"*8}')
    return rows


# ── Full reconstruction ───────────────────────────────────────────────────────

def run_full_reconstruction(best_fx, pairs):
    K        = build_K(best_fx)
    save_dir = os.path.join(OUT_DIR, f'recon_fx{best_fx}')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[sweep] Full reconstruction: fx={best_fx}, {len(pairs)} frames ...')
    print(f'[sweep] Intermediate PLY → {save_dir}/merge_pcd_live.ply')
    print(f'[sweep] (this may take 20–60 min depending on hardware)\n')

    recon = Reconstructor(
        pairs=pairs,
        K=K,
        dist=None,
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        voxel_size=VOXEL_SIZE,
        save_path=save_dir
    )
    final_pcd, succeed, fail = recon.run()

    # Copy final PLY to a fixed name
    out_ply = os.path.join(OUT_DIR, 'candidate_reconstruction.ply')
    save_ply(final_pcd, out_ply)

    n_pts  = len(np.asarray(final_pcd.points)) if not final_pcd.is_empty() else 0
    pct_ok = 100.0 * len(succeed) / len(pairs) if pairs else 0.0
    print(f'\n[sweep] Reconstruction done: {len(succeed)}/{len(pairs)} frames OK '
          f'({pct_ok:.1f}%), {n_pts:,} points')
    print(f'[sweep] PLY → {out_ply}')
    return final_pcd, len(succeed), len(fail), n_pts


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('[sweep] Loading image pairs ...')
    pairs = load_image_pairs(RGB_DIR, DEPTH_DIR, step=1)
    print(f'[sweep] {len(pairs)} pairs found.')
    print(f'[sweep] Anchor frame: {ANCHOR}  |  neighbours: ±{N_NEIGHBORS}')
    print(f'[sweep] Floor depth range: {FLOOR_MIN_MM}–{FLOOR_MAX_MM} mm')
    print(f'[sweep] FX candidates: {FX_VALUES}')

    # ── Step 1-2: sweep ───────────────────────────────────────────────────
    rows = run_sweep(pairs)

    # ── Step 3: write CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'intrinsics_sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['fx', 'mean_fitness', 'mean_rmse',
                           'plane_inlier_ratio', 'score']
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[sweep] CSV → {csv_path}')

    # ── Step 4: pick best fx ──────────────────────────────────────────────
    best = max(rows, key=lambda r: (r['score'], -r['mean_rmse']))

    candidate = {
        'best_fx':            best['fx'],
        'cx':                 1280 / 2.0,
        'cy':                 720  / 2.0,
        'mean_fitness':       best['mean_fitness'],
        'mean_rmse':          best['mean_rmse'],
        'plane_inlier_ratio': best['plane_inlier_ratio'],
        'score':              best['score'],
        'depth_min_m':        2.0,
        'depth_trunc_m':      DEPTH_TRUNC,
        'sweep_weights':      {'plane': W_PLANE, 'fitness': W_FITNESS},
        'all_results':        rows,
    }
    json_path = os.path.join(OUT_DIR, 'candidate_intrinsics.json')
    with open(json_path, 'w') as f:
        json.dump(candidate, f, indent=2)
    print(f'[sweep] JSON → {json_path}')
    print(f'\n[sweep] ── Best fx = {best["fx"]} ──')
    print(f'         fitness={best["mean_fitness"]:.4f}  '
          f'rmse={best["mean_rmse"]:.4f}  '
          f'plane_inlier={best["plane_inlier_ratio"]:.4f}  '
          f'score={best["score"]:.4f}')

    # ── Step 5: full reconstruction ───────────────────────────────────────
    run_full_reconstruction(best['fx'], pairs)

    print('\n[sweep] All done.')
    print(f'  intrinsics_sweep.csv         → {csv_path}')
    print(f'  candidate_intrinsics.json    → {json_path}')
    print(f'  candidate_reconstruction.ply → {OUT_DIR}/candidate_reconstruction.ply')


if __name__ == '__main__':
    main()
