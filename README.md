# PhenoFusion3D

Python tools for **RGB-D–based 3D reconstruction**: turn paired colour and depth images into coloured point clouds, align successive frames with ICP, and merge them into a single model. The project targets phenotyping / plant-imaging workflows (ANU COMP8715 Technical Team Project). A **PyQt desktop app** handles capture, quality checks, and interactive reconstruction; the same codebase also exposes a **headless CLI** for plant-canopy batch runs and local RGB-D fusion (`python main.py --help`).

## Prerequisites

- **Python 3.10+** (3.12 is used in development; match your team’s version).
- A C++ runtime compatible with **Open3D** wheels on your OS (on Windows, the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) is usually required).

## Getting started

For lab and dev installs (preferred), use the bundled installers — see [install/README.md](install/README.md):

- **Lab Linux + ROS:** `./install/install_linux.sh`
- **Windows (camera-only):** `.\install\install_windows.ps1`

For a manual install (any OS), from the repository root:

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux:    source venv/bin/activate
pip install -e ".[windows]"   # or ".[ros]" on the lab Linux machine
```

Launch the app:

```bash
python main.py
```

With arguments, `main.py` runs batch reconstruction instead of opening the GUI (`python main.py --help`). The notebook-oriented canopy and local-merge logic is also documented in `docs/canopy_reconstruction_notes.md` (English) and `docs/canopy_reconstruction_notes_zh.md` (Chinese).

Do not commit large datasets or generated point clouds; see `.gitignore` (`data/`, `*.ply`, `*.pcd`, etc.).

## CLI: plant canopy reconstruction

`rs_data_5` is a rail/conveyor-style capture: each record contains the target large-leaf plant plus other objects that pass through the fixed top-down camera. For this data, the plant-focused canopy mode is the default. It automatically segments the green leaf canopy when no `reconstruction/masks/` folder is present, selects the large target-plant frames, compensates the plant's image-plane rail motion, fuses the depth on an expanded canvas, and exports a coloured surface.

```bash
python main.py --input rs_data_5 --max-frames 11 --coverage-threshold 1
```

This writes one output folder per record under `rs_data_5/canopy_batch/`: `canopy_points.ply`, `canopy_mesh.ply`, `canopy_viewer.html`, `fused_rgb_masked.png`, `fused_mask.png` / `auto_masks/`, previews, and `canopy_summary.json`. Use `--mask-dir` for hand-made masks, or point `--input` at a single record subfolder.

## CLI: local RGB-D reconstruction

Input folders should contain `rgb_*.png`, `depth_*.png`, and ideally `kdc_intrinsics.txt` or `kd_intrinsics.txt`.

```bash
python main.py --mode rgbd --input test_plant_rs13_1 --step-size 24 --max-frames 12
```

Optional frame window and green-only merge:

```bash
python main.py --mode rgbd --input test_plant_rs13_1 --step-size 12 --start-index 220 --end-index 340 --green-filter
```

Default output layout: `<input>/reconstruction_local/` with `merge_pcd_cam*.ply`, `pose/`, and `reconstruction_summary.json`.

## CLI: mask-guided canopy

For datasets with `reconstruction/masks/mask_*.png`:

```bash
python main.py --mode canopy --input test_plant_rs13_1 --output-dir test_plant_rs13_1/canopy_local
```

Outputs include `canopy_points.ply`, `canopy_mesh.ply`, `canopy_viewer.html`, masked RGB, previews, and `canopy_summary.json`.

## Capture (in-app)

The **Data Capture** panel drives an RGB-D capture without leaving the app.

- **Backend = Auto** picks ROS+gantry on the lab machine and RealSense-only on Windows.
- **ROS + Gantry** (lab Linux) wraps `stakeholder_reference/rospy_thread_fin_1.py` with the same `Twist`-based velocity command, the same `align(rs.stream.color)` pipeline and the same intrinsics save logic. UI-tunable parameters: velocity (m/s), end position (m), FPS.
- **RealSense Only** (Windows / dev) captures from the camera directly for `Duration (s)` seconds. Useful for sanity tests.

Output layout (consumed directly by the loader):

```
data/captures/<YYYYMMDDhhmmss>/
    rgb/0.png, 1.png, ...
    depth/0.png, 1.png, ...
    kdc_intrinsics.txt        # color stream
    kd_intrinsics.txt         # depth stream
    session.json              # backend, velocity, frame_index -> gantry position
```

After a successful capture the **Data Loading** fields are auto-populated so you can immediately run the quality check or reconstruction.

## Quality Check (in-app)

The **Data Quality** panel runs depth + ICP diagnostics on the loaded sequence:

- **Quick Check** -- ~15 random consecutive pairs, ~10–30 s.
- **Full Report** -- every consecutive pair; writes `quality_report.csv` and `quality_report.txt` next to the dataset.

Per-pair metrics: depth validity %, median depth (m), point count, ICP fitness, ICP inlier RMSE (m), per-pair rotation magnitude (deg).

Verdict bands (default thresholds):

| Metric | PASS | WARN | FAIL |
|---|---|---|---|
| ICP fitness (mean) | ≥ 0.50 | 0.30–0.50 | < 0.30 |
| ICP inlier RMSE (mean) | ≤ 0.005 m | 0.005–0.015 m | > 0.015 m |
| Depth validity per frame | ≥ 30 % | 10–30 % | < 10 % |
| Per-frame rotation (gantry) | < 1° | 1–5° | > 5° |

The same `min_fitness` / `max_rmse` thresholds are now enforced inside the reconstructor: frames whose ICP result misses either bar are marked **REJECTED** and don't pollute the merged cloud.

## Organizing `data/main` (ICL-style layout)

Team RGB-D drops often use **`rgb_*.png`** and **`depth_*.png`** in a single folder per capture (e.g. `data/main/<sequence>/`). To mirror **`data/icl_nuim/`** (`rgb/0.png`, `depth/0.png`, and **`kdc_intrinsics.txt`** at the sequence root), activate the venv (see above) and run from the repo root:

```bash
python scripts/reorganize_data_main.py --dry-run
python scripts/reorganize_data_main.py
```

Use **`--move`** instead of copy if you want to remove the flat `rgb_*` / `depth_*` files after moving them into `rgb/` and `depth/`. For one sequence, separate RGB/depth folders, or **`camera_N`** layouts, see **`python scripts/reorganize_to_icl_layout.py --help`**.

## Project layout

| Path | Role |
|------|------|
| `file_io/loader.py` | **`load_image_pairs`** — pairs RGB/depth PNGs from two folders (stakeholder `rgb_*.png` / `depth_*.png`, or ICL-NUIM-style `0.png`, `1.png`, …); optional **`step`** subsamples pairs. **`load_intrinsics`** / **`get_default_intrinsics`** for camera JSON |
| `file_io/exporter.py` | **`save_ply`** — write a point cloud to PLY; **`save_metrics_csv`** — per-frame metrics (e.g. fitness / RMSE) to CSV |
| `processing/rgbd.py` | **`rgbd2pcd`** — RGB + depth → Open3D coloured point cloud |
| `processing/icp.py` | Colour ICP with point-to-plane fallback |
| `processing/utils.py` | Downsampling, outlier removal, normals, optional GPU/CuPy check |
| `processing/reconstructor.py` | **`Reconstructor`** — sequential merge via ICP; optional **`save_path`** writes **`merge_pcd_live.ply`** after each successful frame (live merge snapshot) |
| `tests/` | Unit tests (`test_loader`, `test_rgbd`, `test_icp`) |
| `tests/smoke_reconstructor.py` | End-to-end smoke script (synthetic frames → merged cloud) |
| `stakeholder_reference/` | Reference scripts from stakeholders (e.g. `3D_recons.py`); may expect extra deps such as PyTorch |
| `data/` | Local RGB-D sequences (gitignored; keep datasets here, e.g. `data/icl_nuim/`, `data/main/`) |
| `scripts/reorganize_to_icl_layout.py` | CLI: convert stakeholder flat `rgb_*`/`depth_*` layout → `rgb/N.png`, `depth/N.png` + `kdc_intrinsics.txt` |
| `scripts/reorganize_data_main.py` | Wrapper: batch that for each subfolder of `data/main` |
| `app/`, `main.py`, `visualiser/` | PyQt **MainWindow** + Open3D live viewer; `main.py` with CLI args runs headless reconstruction |
| `processing/canopy.py` | **Canopy** reconstruction config + `reconstruct_canopy` (CLI and batch) |

## Data conventions

- **Two filename layouts** (see `load_image_pairs` in `file_io/loader.py`):
  - **Stakeholder:** `rgb_*.png` and `depth_*.png` in separate directories.
  - **ICL-NUIM / numbered:** `0.png`, `1.png`, … in `rgb/` and `depth/` (same count; paired by natural sort order).
- **Subsampling:** Pass **`step=n`** to use every *n*-th pair (e.g. faster experiments).
- **Intrinsics:** JSON in the style of `kdc_intrinsics.txt` with keys such as `K` (3×3), `dist`, `width`, `height`. If the file is missing or invalid, use **`get_default_intrinsics()`** (optionally pass image size to match your frames).
- **Depth units:** Defaults in **`Reconstructor`** assume depth in **millimetres** and **`depth_scale=1000.0`**. For **ICL-NUIM** Freiburg PNG releases, **`depth_scale=1.0`** is typical (depth in metres); tune **`depth_scale`**, **`depth_trunc`**, and **`voxel_size`** if colours or alignment look wrong (e.g. when slicing a subset of frames).

## Sample dataset (ICL-NUIM)

A convenient public RGB-D sequence in PNG form is **ICL-NUIM — living room trajectory 1 (Freiburg PNG)**:

- Download: [http://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz](http://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz)

After extracting under `data/` (e.g. `data/icl_nuim/`), you typically get **`rgb/`** and **`depth/`** folders of matching numbered PNGs, plus metadata such as a ground-truth trajectory (**`livingRoom1.gt.freiburg`**) for evaluation—the loader does not read that file; it only needs paired RGB/depth paths and intrinsics.

Point `load_image_pairs` at your **`rgb`** and **`depth`** directories, supply **`kdc_intrinsics.txt`** (or project-specific intrinsics JSON), and set **`depth_scale`** (often **`1.0`** for this dataset) when constructing **`Reconstructor`**.

## Running tests

From the repository root (with the venv activated):

```bash
python -m pytest tests -q
```

Tests prepend the project root to `sys.path` so imports like `from processing.rgbd import ...` resolve without installing the repo as a package.

## Trying the reconstruction pipeline

The **`Reconstructor`** class in `processing/reconstructor.py` takes a list of `(rgb_path, depth_path)` tuples, intrinsics **`K`**, optional distortion **`dist`**, and runs the sequential ICP merge.

- If **`save_path`** is set, **`merge_pcd_live.ply`** is updated in that folder after each successful frame (final file = full merged cloud at end of run). Use **`file_io.exporter.save_ply`** for one-off or custom export paths; **`save_metrics_csv`** if you record per-frame metrics in a list of dicts.

For a quick check without real data:

```bash
python tests/smoke_reconstructor.py
```

Typical real-data usage: **`pairs = load_image_pairs(rgb_dir, depth_dir, step=1)`**, **`load_intrinsics(path)`** or defaults, then **`Reconstructor(pairs=..., K=..., dist=..., depth_scale=..., save_path=...).run()`**. You can slice **`pairs`** (e.g. Python list slicing) to run on a subset of frames.

## Dependencies (summary)

Declared in `requirements.txt`: Open3D, OpenCV, NumPy, SciPy, natsort, tqdm, PyQt5, pyqtgraph, matplotlib, pyrealsense2 (camera capture). Developer extras (`pytest`, `ruff`) are in `pyproject.toml` under `[project.optional-dependencies] dev`. Optional acceleration paths (e.g. CuPy) are referenced in `processing/utils.py` but are not required for the core tests.

## Contributing tips

- Keep new logic alongside existing modules (`file_io`, `processing`) so tests stay easy to run from the repo root.
- When adding scripts, assume the working directory is the project root or insert the root onto `sys.path` like the tests do.
- Large assets stay out of git per `.gitignore`; use **`data/`** locally and the sample URL above for a standard benchmark sequence.
