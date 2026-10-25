# Reproducing results

A result is reproducible only when its inputs, code, environment, parameters and
manual decisions are available. A rendered report or exported cloud alone does
not supply that record. Obtain datasets and intermediate/final outputs through
the [research archive](RESEARCH_ARCHIVE.md), then choose the workflow below.

## Preserve the run identity

Before a reference run, record:

- Software commit and any release tag; Python environment or container digest.
- Dataset/camera identity, source paths and delivered checksum manifest.
- Camera intrinsics, depth units, pose source and coordinate frame, as applicable.
- Actual command or application selections, parameters and manual annotations.
- Fresh output location, operator, date, completion status and logs.
- Expected checks and tolerances agreed before running, followed by actual results.

Check byte identity for archived files. For processing outputs that can differ
numerically, compare the pre-agreed counts, definitions and numerical tolerances.
A plausible image or high ICP overlap is not independent physical accuracy.

## Workflow map

| Result | Reproduction entry point | Required material and limitation |
|---|---|---|
| Standard RGB-D reconstruction | Main reconstruction panel or offline Reconstruct tab | Matched RGB/depth, intrinsics, depth units and applicable motion data; inspected geometry is only observed coverage. |
| Cleanup and model traits | Offline model-traits path; documented cleanup/segmentation scripts | Reviewed plant-only cloud, axis and saved masks/settings. Cloud hull area is not total leaf area. |
| Manual trait comparison | Validate traits or Matched leaves | Same plant/leaf identity, physical manual record, reviewed endpoints and units. Keep exclusions and provisional estimates. |
| August spectral analysis and 3D overlay | `processing.hyperspectral` | The reviewed 28 August inputs/profile and matching RGB-D products. The automatic historical recipe does not accept arbitrary recordings. |
| September reviewed spectral patches | `processing.research_workspace.spectral_extract` | Compatible ENVI files and explicit reviewed region/reference configuration. Patch extraction is not a full-scene image. |
| Spectral point review | `processing.research_workspace.spectral_viewer` | Complete extraction `result` directories including arrays, metadata and warnings. |
| Sparse spectral associations | `processing.research_workspace.spectral_fusion` | Existing geometry and explicit correspondence configuration. Unassigned surfaces have no inferred spectrum. |
| Whole-scene spectral explorer and final website | Archived generation code/configuration and website sources | Preserve the large viewer data, full band files, source identity and build recipe; these are not all stored in Git. |
| Research evidence workspace | `processing.research_workspace` | Saved clouds, setup JSON and optional reviewed endpoint JSON; this assembles evidence rather than reconstructing the clouds. |

## Safe command discovery

Run from the repository root in the analysis environment. These help commands do
not initiate processing or contact hardware:

```text
python -m processing.analysis_workflow --help
python -m processing.hyperspectral --help
python -m processing.research_workspace --help
python -m processing.research_workspace.spectral_extract --help
python -m processing.research_workspace.spectral_viewer --help
python -m processing.research_workspace.spectral_fusion --help
```

All six commands were checked against this checkout on 26 October 2026. Help
success validates the interface, not scientific output or lab compatibility.

## An independent reference run

Use the receiving operator's environment and an agreed copy of one reference
dataset. Avoid live hardware for this recorded-data rehearsal.

For cloud evidence review, first create a template in an unused location:

```text
python -m processing.research_workspace template --output generated/handover_setup_01
```

Edit the resulting `workspace_template.json` with actual cloud paths, units and
known evidence. Do not leave illustrative paths or assert verified scale unless
supported. Build into a new output location:

```text
python -m processing.research_workspace build --manifest generated/handover_setup_01/workspace_template.json --output generated/handover_review_01
```

Open the generated `result/index.html`, inspect the cloud and metadata, and
record a pass only if the chosen input/result checks succeed. This demonstrates
workspace reproducibility, not a fresh RGB-D reconstruction. For reconstruction,
follow the input check and Reconstruct tab in [ANALYSIS_WORKFLOW.md](../../ANALYSIS_WORKFLOW.md).

For a reviewed spectral extraction configuration with paths adjusted to the
restored archive:

```text
python -m processing.research_workspace.spectral_extract --config path/to/reviewed_extraction.json --output generated/handover_spectral_01
```

The placeholder configuration must be replaced with a real reviewed file. See
[ENVI format and reference requirements](../RESEARCH_SPECTRAL_EXTRACTION.md).
Raw DN is not calibrated reflectance; retain missing/invalid band flags and
reference assumptions in every export.

## Historical August replay

Use the complete matching archive folders, including reviewed RGB-D specimen
products and historical spectral inputs. From a clean output location, start
with input inspection using the paths in the delivered manifest:

```text
python -m processing.hyperspectral all --hsi "PATH/20260828" --rgbd "PATH/test_plant_20260828120800_best_lighting" --output "PATH/new-inspection" --inspect-only
```

This writes inspection output; it does not prove the optional dependencies or
physical fusion are valid. Run the selected stage only after reviewing the
inspection. Full commands and required files are in
[HYPERSPECTRAL_WORKFLOW.md](../../HYPERSPECTRAL_WORKFLOW.md).

## Result status and superseded outputs

Preserve historical outputs with dates and status. Corrected manual endpoints
and subsequent reviews may supersede earlier comparison tables. A historical
report can contain an old cloud count or calibration assumption: identify which
run a claim refers to. The final poster is a communication artifact, not the
source of processing parameters.

Retain these distinctions in future work:

- Observed 3D geometry does not establish unseen stems or complete leaf surfaces.
- RGB-D image references share sensor data and are not independent ground truth.
- Reviewed sparse associations do not establish dense cross-sensor registration.
- The August overlay demonstration and September 2D exploration are separate
  datasets and products.
- Spectral descriptors remain exploratory unless their radiometric and biological
  interpretation has independent supporting validation.
