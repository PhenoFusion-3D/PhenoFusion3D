# Capture Protocol for 3D Plant Reconstruction

This document describes how to capture data that will produce the best results
with PhenoFusion3D's canopy reconstruction pipeline, and what additional captures
are required to go beyond the current 2.5D top-surface result towards a true
full-plant 3D model.

---

## 1. What the current pipeline can and cannot reconstruct

The **Canopy** reconstruction mode (the default recommended mode) performs a
top-down depth-fusion:

| Feature | Capability |
|---|---|
| Top leaf surface | Excellent — dense, accurate, smooth |
| Leaf colour/texture | Very good (median-fused RGB) |
| Leaf edges from above | Good, with slight blur at tips |
| Leaf underside | **Not captured** — physically invisible from above |
| Stem and main branches | Partial — visible portions only |
| Back faces of vertical/curled leaves | **Not captured** — self-occluded |
| True leaf thickness | Simulated (software only, see §5) |
| True 360-degree geometry | **Requires additional capture angles** |

If your goal is only the **top-surface phenotype** (e.g., leaf area index, leaf
spread, canopy height-map), a single top-down pass already gives excellent results.

If your goal is a **full-plant 3D model** visible from any angle, follow the
multi-pass protocol in §3.

---

## 2. Optimal single top-down pass (current workflow)

### Camera settings
- Depth camera: Intel RealSense D400-series (D405 at 2–3 m range is ideal).
- RGB: 1280 × 720 or 1920 × 1080 at 30 fps.
- Depth: 1280 × 720 at 30 fps (aligned to RGB).
- Enable spatial and temporal filters in the RealSense SDK to reduce flying pixels.

### Lighting
- Use diffuse, even overhead lighting.  Avoid strong directional sunlight, which
  creates specular highlights on leaves and saturates depth readings.
- If using a grow-light canopy, mount the camera below the light plane so the
  leaves are lit from above while the camera looks down.

### Gantry setup
- Travel speed: **3–5 cm/s** (slow enough that sequential frames overlap ≥ 60%).
- Frame rate: 30 fps.
- This gives an inter-frame step of approximately 1–1.7 mm — well within the
  canopy pipeline's phase-correlation alignment tolerance.
- Capture height above canopy: **1.5–2.5 m** for an RSD405.  Closer gives finer
  detail; further increases field of view.
- Traverse the full canopy **twice** (one forward pass, one return pass) to double
  the number of candidate frames available for fusion.

### Dataset structure produced
The gantry software saves:

```
<session_id>/
    rgb_<timestamp>.png    (one per frame)
    depth_<timestamp>.png
    kdc_intrinsics.txt
    gantry_config.json
```

This flat layout is supported directly by the canopy pipeline — no reorganisation
needed.

### Recommended canopy parameters (starting point)
| Parameter | Recommended value | Notes |
|---|---|---|
| Canopy Stride | 10 | Sample every 10th frame |
| Max Frames | 9–15 | Increase for sparse canopy |
| Depth Sigma | 3.5 | Increase to 5–7 if frames are sparse/noisy |
| Coverage Min | 1 | Increase to 2 if edge speckle is visible |
| Mask Sensitivity | Default | Use Loose for pale/yellow-green plants |
| Leaf Thickness | Off | Enable only when side-view is needed |

Run `sweep_canopy.py` (see below) to find the best parameters for a new plant
species or lighting condition before committing to a single configuration.

---

## 3. Multi-pass protocol for full-plant 3D geometry

To capture the hidden sides, undersides, and self-occluded surfaces of a plant,
you need **multiple camera viewpoints** that collectively see every surface.

### Option A — Rotate the plant (recommended)

1. After the standard top-down pass, rotate the **plant pot** by **90°**
   and capture another top-down pass.
2. Repeat at 180° and 270°.
3. This gives four datasets, each seeing a different side of every leaf.

Reconstruction workflow (future work):
- Run canopy fusion independently on all four datasets.
- Register the four point clouds with ICP using the canopy result as a common
  reference surface.
- Merge into a single cleaned mesh.

### Option B — Tilt the camera

Mount the camera on a tiltable bracket:

- Standard pass at 0° (straight down).
- Two additional passes at **±30–45° tilt** (one from each side).

This captures the leaf undersides and vertical stem faces without moving the
plant.  Requires re-calibrating the camera intrinsics at each tilt angle.

### Option C — Angled gantry passes

Run three parallel gantry passes at different lateral offsets, with the camera
tilted inward at each pass:

```
  ← offset →
   \    |   /
    \   |  /       camera beams looking inward
     \  | /
      [ plant ]
```

This is the most mechanically complex option but requires no manual plant
handling.

### Option D — Turntable under fixed camera

Place the plant on a motorised turntable.  Rotate 360° in small steps (e.g., 5°)
while capturing depth frames from a fixed oblique angle.  This is ideal for pot
plants and produces the most complete geometry.

---

## 4. Using `sweep_canopy.py` to find optimal parameters

Before running a full session, use the sweep tool to find the best parameter
combination for your current plant and lighting conditions:

```bash
# Quick sweep (3 combinations, ~3–5 min):
python sweep_canopy.py --dataset data/main/<your_dataset>

# Full sweep (24 combinations, ~20–40 min):
python sweep_canopy.py --dataset data/main/<your_dataset> --full

# Custom sweep:
python sweep_canopy.py --dataset data/main/<your_dataset> \
    --max-frames 5 9 15 \
    --smooth-sigma 2.0 3.5 6.0 \
    --coverage 1 2
```

Open `sweep_<timestamp>/sweep_report.html` in a browser.  The table is sortable
— sort by **Points** descending to see the parameter combination that produced
the densest reconstruction.  The preview cards show the fused RGB image, depth
map, oblique 3D view, and coverage mask side-by-side for each run.

Record the winning `max_frames`, `smooth_sigma`, and `coverage` values in your
session notes and enter them in the PhenoFusion3D UI before the final run.

---

## 5. Leaf thickness simulation (software-only improvement)

Enable the **Leaf Thickness** checkbox in the Canopy section of the UI (or pass
`--leaf-thickness 0.003` to `reconstruct_canopy.py`).

This duplicates the top-surface point cloud with a small downward Z offset
(default 3 mm), creating the appearance of solid leaves when viewed from the
side.  It does not add real geometric information but removes the "paper-thin"
artefact that is otherwise visible in side views.

Typical setting: **2–5 mm** depending on plant species.

---

## 6. Expected output quality by capture method

| Capture method | Top view | Side view | Underside | Use case |
|---|---|---|---|---|
| Single top-down pass | ★★★★★ | ★★☆☆☆ | ✗ | Canopy metrics, leaf area |
| Top-down + leaf thickness (software) | ★★★★★ | ★★★☆☆ | ✗ | Presentation, phenotype |
| Plant rotation (4× 90°) | ★★★★★ | ★★★★☆ | ★★★☆☆ | Full plant analysis |
| Tilt passes (0°, ±35°) | ★★★★☆ | ★★★★☆ | ★★★☆☆ | Fixed gantry upgrade |
| Turntable 360° | ★★★★★ | ★★★★★ | ★★★★★ | High-accuracy 3D model |

---

## 7. Intrinsics and calibration checklist

- Always capture `kdc_intrinsics.txt` with the camera at the exact height and
  orientation used for reconstruction.  Changing height changes the distortion
  profile.
- Re-run `calibrate_gantry.py` (or the Calibrate Gantry button) whenever the
  gantry speed, frame rate, or camera mounting position changes.
- Keep the `gantry_config.json` alongside each dataset — the canopy pipeline
  reads it automatically.

---

## 8. Troubleshooting common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Only a few frames selected (< 5) | Low mask area — poor plant/background contrast | Use **Loose** mask sensitivity |
| Speckled edges in result | Too many noisy frames included | Increase Coverage Min to 2; use Strict mask |
| Blurry leaf boundaries | Depth Sigma too high | Reduce to 2.0–3.0 |
| Holes in the centre of leaves | Glossy/reflective leaf surface | Enable NaN inpainting in depth pre-processing |
| Large missing region | Plant moved during scan | Increase Max Frames and re-capture |
| Side views look flat | Only top-down data | Enable Leaf Thickness or capture additional angles |
