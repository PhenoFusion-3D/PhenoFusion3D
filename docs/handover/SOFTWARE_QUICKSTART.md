# Software quick start

Use this guide to open the handed-over application and process an agreed saved
recording. The team reports successful operation on the existing lab machine;
record its release, machine and demonstrated capabilities in the acceptance
record. This guide does not claim that every camera or a container was tested.

## Choose the correct environment

| Environment | Purpose and boundary |
|---|---|
| Existing Linux lab installation | Capture, the established ROS gantry integration, and supported processing. Retain the working host drivers and ROS workspace. |
| Separate analysis environment | Offline processing and inspection of copied recordings. No physical camera or gantry is needed for these actions. |
| Windows installation | Offline analysis and supported RealSense camera-only capture. The provided Windows installer does not install ROS gantry support. |
| Container | Follow the delivered container instructions and evidence if supplied. A native lab pass does not establish USB, display or ROS operation inside a container. |

## Open the existing lab installation

Open **PhenoFusion3D** from Ubuntu Activities. For a terminal launch from the
existing checkout:

```bash
source .venv-linux/bin/activate
python main.py
```

Do not reinstall a working environment simply to demonstrate handover. Identify
the checkout and interpreter already used by the lab. The following existing
check is read-only and reports dependencies, ROS runtime and connected cameras:

```bash
./setup.sh --check
```

It verifies availability, not a safe physical movement or capture test. The lab
operator retains responsibility for the established rig procedure.

## Install a separate copy when needed

On a Linux lab machine that already has Python 3.10-3.12, its venv module,
RealSense system configuration and the working ROS/gantry workspace, run:

```bash
./setup.sh
source .venv-linux/bin/activate
python main.py
```

The script creates the project environment and a per-user launcher. It does not
install system packages, ROS, drivers or a gantry workspace. See the exact
[installation boundary](../../install/README.md).

For a separate offline analysis checkout, without changing the working lab
installation:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -e ".[analysis,hyperspectral,hyperspectral-viewer]"
python main.py
```

On Windows, use `py -3.11 -m venv .venv-analysis`, then
`.venv-analysis\Scripts\python.exe` in place of `python` for the last two
commands. The optional viewer requires a compatible Qt/WebEngine environment;
a local browser is available when embedded 3D viewing is unavailable.
Dependency installation normally requires network access. An offline installer
or container must supply its dependencies separately; this command is not an
offline-installation promise.

The camera-only Windows installer is
[install/install_windows.ps1](../../install/install_windows.ps1). It installs
the pinned RealSense Python binding and checks imports/devices. Do not use it to
replace a lab's Linux capture environment.

## Retrieve inputs before the first run

Follow the [research archive catalogue](RESEARCH_ARCHIVE.md). A Git clone does
not include `data/`, generated reports or most point clouds. Copy an agreed
reference dataset, preserve original files and verify its delivered manifest.
Keep the results destination outside the raw recordings. Choose a fresh output
folder for each run.

For RGB-D, require matching RGB/depth frame IDs, matching image dimensions,
recorded camera intrinsics and known depth units. `session.json` may supply
frame positions and units; historical recordings may need explicit documented
parameters. Never infer metric scale by making a plant look the right size.

## First operator walkthrough

1. Launch the software and open **Analysis > Offline reconstruction, traits and
   hyperspectral fusion...**. Choose the tab for the requested task below.
2. Select the agreed reference inputs and a fresh results folder. Use the
   workflow's input check before processing; review missing metadata or warnings.
3. Run a small reference workflow. Inspect its result and save the log, settings,
   source identity and output location. A saved report replay is useful for
   inspection, but it is not a fresh reconstruction.
4. Reopen the exported result after closing the application. Verify that it is
   in the handed-over persistent storage rather than a temporary directory.
5. Have the receiving operator repeat the instructions and record the outcome.

| Task | Tab or action | Full instructions |
|---|---|---|
| Reconstruct saved RGB-D | Reconstruct | [Offline analysis](../../ANALYSIS_WORKFLOW.md) |
| Model descriptors and comparisons | Model traits / RGB-D references; Validate traits; Matched leaves | [Offline analysis](../../ANALYSIS_WORKFLOW.md) |
| Historical August spectral processing/fusion | Hyperspectral / fusion - experimental | [Historical workflow](../../HYPERSPECTRAL_WORKFLOW.md) |
| Review specimen clouds and endpoint evidence | Research workspace | [Workspace guide](../RESEARCH_WORKSPACE.md) |
| Extract spectra from reviewed ENVI regions | Research workspace | [Spectral extraction](../RESEARCH_SPECTRAL_EXTRACTION.md) |
| Inspect extracted spectra and sparse measured associations | Spectral review / 3D fusion | [Spectral review guide](../../SPECTRAL_FUSION_WORKFLOW.md) |

The historical August recipe is dataset restricted. The September whole-scene
pixel explorer and its generated report are separate archived research outputs;
do not assume that a website screenshot means every step exists as an app button.
Use [Reproducing results](REPRODUCING_RESULTS.md) to find each executable path.

## Live capture is a separate walkthrough

Use the established lab procedure and [operator guide](../final-sprint-20260928/USER_GUIDE.md).
`RealSense Only` captures without gantry motion. `ROS + Gantry` uses the existing
ROS driver and camera. If multiple cameras are connected, the operator must
select the intended serial through `PHENOFUSION_CAMERA_SERIAL`. Do not change
movement limits, endpoints or drivers during a handover demonstration.

On a capture failure, preserve the log and follow the operator's stop/recovery
procedure. Offline processing can continue on recorded data; that does not count
as passing a failed live capture acceptance check.
