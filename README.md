# PhenoFusion3D

This repository now exposes the useful part of the stakeholder notebook
`stakeholder_reference/3D_reconstruction.ipynb` as a local CPU-friendly pipeline.
The ROS and RealSense plant-scanning code remains only as reference and is no
longer required for reconstruction.

## Plant Canopy Reconstruction

`rs_data_5` is a rail/conveyor-style capture: each record contains the target
large-leaf plant plus other objects that pass through the fixed top-down camera.
For this data, the plant-focused canopy mode is the default. It automatically
segments the green leaf canopy when no `reconstruction/masks/` folder is present,
selects the large target-plant frames, compensates the plant's image-plane rail
motion, fuses the depth on an expanded canvas, and exports a coloured surface.

Run the bundled `rs_data_5` batch:

```bash
python main.py --input rs_data_5 --max-frames 11 --coverage-threshold 1
```

This writes one output folder per record under `rs_data_5/canopy_batch/`:

- `canopy_points.ply`: fused canopy point cloud
- `canopy_mesh.ply`: coloured surface mesh
- `canopy_viewer.html`: drag-to-rotate browser viewer
- `fused_rgb_masked.png`: aligned target-plant appearance
- `fused_mask.png` and `auto_masks/`: generated plant masks
- `canopy_topdown.png` and `canopy_oblique.png`: quick previews
- `canopy_summary.json`: selected frames, shifts, depth ranges, and counts

If you have hand-made masks, pass them with `--mask-dir`. To reconstruct only one
record, point `--input` at that subfolder:

```bash
python main.py --input rs_data_5/test_plant_20230809132457 --max-frames 11
```

## Local RGB-D Reconstruction

Input folders should contain:

- `rgb_*.png`
- `depth_*.png`
- `kdc_intrinsics.txt` or `kd_intrinsics.txt` (optional but recommended)

Run reconstruction on the bundled sample dataset:

```bash
python main.py --mode rgbd --input test_plant_rs13_1 --step-size 24 --max-frames 12
```

For sequences that contain setup frames before the plant appears, limit the frame
window and keep only green points:

```bash
python main.py --mode rgbd --input test_plant_rs13_1 --step-size 12 --start-index 220 --end-index 340 --green-filter
```

The command writes results under `test_plant_rs13_1/reconstruction_local/` by default:

- `merge_pcd_cam.ply`: merged point cloud
- `pose/`: estimated per-frame poses
- `reconstruction_summary.json`: run metadata and failure details

If you want denser output, decrease `--step-size`. If you want a quick smoke run,
keep `--max-frames` small.

## Mask-Guided Canopy Reconstruction

Older datasets with precomputed masks are still supported. This path uses
`reconstruction/masks/mask_*.png` plus the depth maps, with the same expanded
canvas fusion used by the automatic `rs_data_5` flow.

```bash
python main.py --mode canopy --input test_plant_rs13_1 --output-dir test_plant_rs13_1/canopy_local
```

Useful outputs:

- `canopy_points.ply`: fused orthographic canopy point cloud
- `canopy_mesh.ply`: leaf-like canopy surface mesh
- `canopy_viewer.html`: interactive 3D viewer you can drag in a browser
- `fused_rgb_masked.png`: the aligned plant appearance used for fusion
- `canopy_topdown.png` and `canopy_oblique.png`: quick previews
- `canopy_summary.json`: selected frames and fusion metadata
