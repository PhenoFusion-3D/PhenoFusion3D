# Troubleshooting

Use the exact delivered environment and preserve the error log before changing
anything. Resolve input and setup problems independently from scientific
interpretation. A successful launch does not prove a result is calibrated.

| Symptom | Check and action |
|---|---|
| The repository has no recordings or PLY models | Expected for large ignored assets. Retrieve the [research archive](RESEARCH_ARCHIVE.md), verify the manifest and update local paths. |
| RGB and depth files have equal counts but loading fails | Normalised frame IDs must match as well as counts. Find the missing/wrongly named original pair; do not pair adjacent unrelated files or rename them to force success. |
| Missing calibration or depth scale | Use the matching capture intrinsics and recorded sensor units. Approximate legacy intrinsics and guessed scale cannot support metric claims. See [offline analysis](../../ANALYSIS_WORKFLOW.md). |
| Plant scale looks wrong | Verify units, image resolution and camera intrinsics. Do not tune scale to match the expected manual trait. |
| Existing output folder is rejected | Select a fresh output directory outside the inputs. Completed and failed previous runs must remain distinguishable. |
| Saved configuration points to the old laptop | Restore the required archive and update the configuration's declared paths. Relative paths may resolve against the configuration location; inspect the workflow guide before moving it. |
| Historical hyperspectral run rejects a new recording | The automatic August recipe is restricted by reviewed input provenance. Use explicit new-data extraction/review where supported; do not defeat fingerprint checks or rename files. |
| Spectral extraction rejects ENVI input | Check header, byte length, unsigned-16-bit BIL layout, byte order, offsets and nanometre wavelength vector against the [input contract](../RESEARCH_SPECTRAL_EXTRACTION.md). |
| Spectral value or index is blank | Missing coverage, invalid bands, clipping/reference quality or unavailable wavelength may be responsible. Keep the quality flags; never replace missing data with another leaf or zero. |
| Clicking a grey 3D point produces no spectrum | Grey geometry may only provide context. Only explicitly associated measured points have spectra. Marker size is a display aid, not a measured footprint. |
| 3D report is static or WebGL unavailable | Use the report window's browser option and a WebGL-capable local browser. Check asset paths and optional Qt WebEngine installation. Do not change capture SDKs to fix a report viewer. |
| Qt platform plugin error on Linux | Launch through `main.py`/the established launcher, which creates Qt before importing OpenCV. Use a separate analysis environment to investigate; do not replace the working lab Qt environment blindly. |
| No RealSense camera or several connected devices | Check existing USB/system setup and the model/serial report from `./setup.sh --check`; choose the intended `PHENOFUSION_CAMERA_SERIAL` for multiple cameras. |
| No gantry position / movement unavailable | Verify the existing host ROS runtime/workspace and driver. `PHENOFUSION_ROS_WS` can identify a nonstandard workspace. Do not install ROS into the application venv. Follow the lab operator's procedure. |
| Analysis does not start while another task is active | Capture/processing guards prevent conflicts. Complete or cancel the current job; cancelled outputs remain incomplete. |
| Cloud has missing stems, floating fragments or holes | Inspect source coverage, masks, alignment and rejected observations. Do not fill missing surfaces and call them measured. Review cleanup exclusions and the supported capture geometry. |
| Measured leaf width differs from the ruler | Verify the same leaf, base/tip/width landmarks, units, curvature and depth support. Preserve provisional/excluded status; a good visual outline is insufficient. |
| Website is unavailable but local reports exist | Check the separate website deployment record and administrator. Local files and a successful build do not establish live publication. Use the documented offline copy where available. |

## Collect a support record

Include the commit/release, machine and OS, Python environment, input identity,
workflow action, complete error message, output path and the steps already tried.
State whether the problem is an import/setup failure, invalid input, processing
failure, viewer problem, hardware issue or a scientific limitation. This helps
the maintainer reproduce the fault without requiring the original development
laptop.

## Recover safely

Keep originals read-only, keep a second verified archive copy, and rerun into a
new output folder. For a processing interruption, preserve `run_status.json` or
other completion markers and do not label partial output complete. For physical
capture or gantry faults, use the lab's established stop/recovery procedure and
supervision; this document does not replace hardware operating instructions.
