# Software maintenance

This guide is for the next maintainer of PhenoFusion3D. Start from the exact
handed-over commit and preserve a known working lab installation before changing
dependencies, capture or motion behaviour. The team reports that the current
software works on the lab machine and that stakeholder access exists; retain
that evidence with the final release identity.

## Ownership and release identity

Record the receiving maintainer, repository administrator, data custodian and
website administrator in the handover record. Repository access and ongoing
maintenance responsibility are different agreements. Agree a support end date
and contact route; no indefinite student support is implied.

The local `origin` may be a personal fork. Before pushing, verify the intended
stakeholder destination with `git remote -v`. Record the delivered commit with
`git rev-parse HEAD`; a package version such as `0.2.0` is not sufficient to
identify a scientific run. Do not describe an unpushed local commit as received
by the stakeholder.

## Components and boundaries

| Component | Main locations | Responsibility |
|---|---|---|
| Desktop application | `main.py`, `app/` | Qt startup, panels, job coordination and report windows. Create Qt before importing OpenCV-dependent modules. |
| Capture and motion | `capture/` | RealSense capture and bridge to the existing host ROS interpreter/workspace. |
| Input and output | `file_io/` | Paired frame identifiers, intrinsics/session loading and exports. |
| Reconstruction | `processing/rgbd.py`, `icp.py`, `reconstructor.py`, `rgb_recovery/` | RGB-D conversion, pose/alignment logic and offline recovery routes. |
| Traits | `processing/analysis_workflow.py`, `pointcloud_post.py`, `trait_validation.py`, `leaf_tracking.py` | Model definitions, same-organ comparison and observation recovery. |
| Spectral workflows | `processing/hyperspectral/`, `processing/research_workspace/` | Historical recipe, measured extraction, sparse associations and evidence reports. |
| Setup | `setup.sh`, `install/`, `pyproject.toml` | Isolated application environment and declared dependencies. |
| Research records | `docs/results/`, `docs/final-sprint-20260928/` and the external archive | Dated evidence and methods; historical records are not overwritten by later findings. |

The website is a separate repository and build. Keep its viewers and large
assets consistent with the [research archive](RESEARCH_ARCHIVE.md). A website
build does not establish application or camera operation.

## Dependency and installation discipline

The Linux setup supports the existing Python 3.10-3.12 lab arrangement and keeps
ROS on the host. `pyrealsense2==2.54.2.5684` is intentionally pinned in the lab
and Windows extras. Do not casually upgrade it, install `rospy` into the GUI
venv, replace drivers or change the lab's catkin workspace.

Optional `analysis`, `hyperspectral` and `hyperspectral-viewer` extras extend the
application's recorded-data paths. Test changes in a separate environment first.
The core dependency declarations use lower bounds, not a complete frozen lock;
retain the accepted environment's package inventory with the release. Rebuilding
later can otherwise resolve different dependency versions.

Container packaging must declare its image tag/digest, mounts, display support,
host/device boundary and tested workflows. Persist inputs and exports outside
the disposable container filesystem. Offline processing, hardware capture and
full lab integration require separate checks.

## Making a change

1. Preserve the known working commit, environment inventory and reference output.
2. Create a branch from the agreed baseline. State which behaviour changes and
   which existing data/assumptions it affects.
3. Add focused tests for changed scientific or input-handling behaviour. Preserve
   original recordings and use fresh output directories.
4. Run the relevant tests, then the agreed reference workflow. Review numerical
   and visual outputs, warnings and failure paths, not just process exit codes.
5. If capture, timing, motion, SDKs or host integration changed, arrange supervised
   lab acceptance before replacing the working release.
6. Review the change, update guides and known issues, and record the commit and
   verification in the release manifest. Keep a rollback copy.

For a development environment, install the project's `dev` extra before testing.
From the repository root:

```text
python -m pytest tests -q
```

Some tests use optional data or platform dependencies and may skip. Record the
actual pass/fail/skip summary and environment. Synthetic tests and CLI-help
checks do not prove physical accuracy or working camera/gantry hardware.

## Rollback and incident record

Stop processing before changing environments. Keep raw inputs unchanged and
save the failed run's log, settings, version and partial outputs with a failed
status. Restore the last accepted source/environment and rerun a known small
reference. Do not delete evidence or overwrite archived results to hide a failed
run. Follow the lab's established stop procedure for physical hardware.

Capture a useful issue with the application version/commit, OS, Python and key
package versions, input identity, operation, expected/actual behaviour and log.
Exclude credentials and restricted datasets from public issues; share an approved
minimal reproducer or refer to the institutional archive.

## Licensing and third party material

The repository `LICENSE` and README declare MPL-2.0. Retain bundled notices,
including `processing/hyperspectral/legacy/LICENSE` and its source manifest.
Check that release metadata matches the approved project licence before
publishing a package. Dataset, photography, model and dependency redistribution
permissions require their own review; a code licence does not automatically
cover all research materials.

## Before declaring closure

Confirm that the receiving person can locate and access both repositories,
retrieve the data/results archive, follow a reference workflow and find the
known limitations. Record future maintenance responsibility and any exceptions
with owner and date. Repository permission alone is not a complete transfer of
the ignored research files or environment.
