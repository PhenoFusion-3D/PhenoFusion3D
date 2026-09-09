# Offline analysis integration evidence — 9 September 2026

## Scope and baseline

Integrated on the fetched lab baseline `597c95b`. The only existing application
UI change is an additive Analysis menu and optional child-process lifecycle.
`main.py`, `capture/`, `app/controller.py`, capture workers, the existing
`processing/reconstructor.py`, launcher/setup scripts and camera dependency pins
have no integration changes. This is source comparison, not physical hardware
acceptance.

Earlier local research edits, including capture experiments, were preserved in
Git stash `5f80f45d2ec7f833ce09751ae1c7e23b56944d80`, named
`Research before offline analysis integration 2026-09-09`. The durable backup branch is `codex/research-preserved-20260909`. Its third parent
contains the untracked research files. Do not apply the entire stash to lab main:
that would reintroduce the earlier capture experiments. Raw recordings and
existing generated results were retained in place.

## Automated checks

The final integration suite passed 145 tests, with one skipped and five deselected.
It includes the invalid-calibration guard and real offscreen MainWindow
startup, Analysis dialog wiring, separate process failure, capture-priority
cancellation, shutdown, sensor ICP execution, depth units, frame identity,
calibration, occlusion, visibility batching, shared support-plane coordinates,
manual matching and camera/ROS/gantry regression checks.

The five excluded cases were reproduced on clean `597c95b`: three TSDF tests
expect AgentConfig fields absent from that baseline, and two Linux setup tests
cannot run as written under Windows. `tests/test_e2e.py` also cannot collect on
that clean baseline because `processing.canopy` is absent. These existing issues
were not repaired by changing unrelated lab code.

## Recorded-data checks

- Coleus recording ending `109`: the app's processing entrypoint selected 44
  metric camera poses and 22 dense views without supplied poses or dataset-specific
  frame selection. All 22 ICP refinements passed their gates. Camera reprojection
  median was 0.219 pixels and P90 0.644 pixels. Camera optimization reached its
  iteration limit; those residuals do not establish numerical convergence or
  physical accuracy.
- The complete pipeline was run, then fusion was rerun from its unchanged camera
  and stereo outputs to verify bounded visibility batches and the shared support
  plane. The reviewed result contains 1,266,476 points, minimum three supporting
  views, median six, and extents 0.519 × 0.429 × 0.369 m. Front, side and top
  previews were inspected. Gaps, pot geometry and some edge artifacts remain.
  Output: `generated/integration_109_final/reviewed_result/` in the working
  repository. This is an automatic candidate, not identical to the earlier
  capture-specific research recipe or certified physical measurements.
- The previous nine manually marked leaves were processed into 18 separate
  length/width comparisons with recorded frame tracking and depth search radii.
  Output: `generated/integration_leaf_validation_v4/`. Some estimates require
  depth sampling up to 25 pixels away and deserve review.
- RGB-D reference extraction completed on the three-plant best-lighting recording.
  Three older input layouts passed preflight and selected the sensor-depth route
  when encoder metadata was absent. These input checks are not full reconstruction
  or physical-scale validation across those datasets.

Generated datasets, models and reports are local evidence and are not in the
source commit. The software recreates its reports offline; no AI service is used.

## Lab acceptance still required

Follow the existing lab procedure for startup, capture, jog/release-stop, home,
buffered save and shutdown. Then open Analysis and process a saved recording.
Check geometry against photographs; exclude the pot and confirm the physical
height base before validating dimensions. No hardware was connected for these
checks, and no claim of perfect reconstruction for arbitrary plants is made.

See [ANALYSIS_WORKFLOW.md](ANALYSIS_WORKFLOW.md) for inputs, supported capture
geometry, manual specimen matching and recording improvements.

The model-trait extraction and explicitly matched RGB-D comparison also completed
for the Coleus candidate. Those reports are pipeline checks; pot-inclusive
descriptors are marked unvalidated and must not be presented as plant-only traits.
