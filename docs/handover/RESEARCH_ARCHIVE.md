# Research archive and transfer

The code repositories do not contain the complete research record. `data/`,
`generated/`, PLY and PCD outputs are intentionally excluded from the software
repository. The website's publication data is also a subset of the original
recordings and intermediate experiments.

## The handover set

The 26 October 2026 upload set is prepared under
`output/PhenoFusion3D_Handover_20261026/` on the delivery machine. Preserve the
folder's `START_HERE` instructions, archive index, per-file manifests and ZIP
checksums alongside the ZIPs when uploading to stakeholder-controlled OneDrive
or SharePoint. This local path is not a shared download address.

| Material | Delivered form | Purpose |
| --- | --- | --- |
| Software and website | Source snapshots and Git history bundles under `code/` | Continue development from the recorded revision. |
| Original and derived recordings | `archives/data__*.zip` | Complete locally inventoried `data/` tree, including earlier datasets and manual reference material. |
| Research outputs | `archives/research__*.zip` | Geometry, calibration, cleanup, traits, spectra, fusion, historical and rejected experiments, reports and associated scripts. |
| Extra reproduction scripts | Supplementary source archive | Research generation scripts and configuration outside tracked code. |
| Final presentation | Presentation archive | Final poster, video, narration and presentation source material. |
| Operational documents | Repository handover guides and meeting pack | Installation, reproduction, troubleshooting, release and acceptance records. |
| Container | Image export and image receipt when built and checked | Recorded-data environment; see [container instructions](../../install/container/README.md). |

The index generated with the finished packages is authoritative for counts,
sizes and hashes. ZIPs are independent archives, not numbered fragments of one
split ZIP. An existing historical ZIP inside a dataset is retained unless its
expanded equivalent is explicitly recorded as included. See packaging exclusions
for dependency caches, virtual environments and disposable rendering builds.

## Restore on another machine

1. Download the index, manifests, verification helper and every ZIP listed for
   the required scope. For full preservation, download the complete set.
2. Verify each downloaded ZIP against `SHA256SUMS.txt` before extracting. On
   PowerShell, `Get-FileHash -Algorithm SHA256 <zip>` prints its digest. On Linux,
   use `sha256sum -c SHA256SUMS.txt` from the package directory where provided.
3. Extract source and research ZIPs into a new staging area. Research ZIPs retain
   project-relative paths such as `data/main/...` and `generated/research_...`.
   Follow the package index's source-root instructions; do not flatten folders.
4. Keep the uniquely named external manifest for each ZIP. The embedded
   `HANDOVER_MANIFEST.json` has a common filename and may be replaced when merging
   extractions; that does not replace the external manifests.
5. Use `python tools/handover/archive_catalogue.py --help` for supported manifest
   creation and verification options. Do not treat a metadata-only inventory
   with null hashes as checksum evidence. The package index also records CRC
   verification of the ZIP contents.
6. Open one RGB image, depth frame, hyperspectral header/data pair, cloud, trait
   table and report from the downloaded copy. Run an agreed reference workflow
   using [the reproduction guide](REPRODUCING_RESULTS.md).

## Recipient custody checklist

Record the shared storage URL, institutional owner, access group, available
capacity and retention policy in [the release record](RELEASE_RECORD.md). Ask the
receiving operator to open the link using their own account, download a package,
verify its checksum and restore a selected result. Save that dated confirmation.
OneDrive synchronisation on the sender's laptop is not proof of recipient access.

At preparation time, Adithya confirmed stakeholder access to both repositories
and working lab software. Dataset/results storage and recipient receipt have not
yet been confirmed. ZIP creation and local checks do not change that status.

## Preserve scientific meaning

Keep capture IDs, camera identities, original frame names, units, calibration
assumptions, commands and manual masks/endpoints. Distinguish original, derived,
current, provisional, superseded and rejected outputs. An intermediate result is
part of the record even when a later result replaces its interpretation.

The August spectral overlay, September whole-scene spectral inspection and
September partial 3D associations are separate results. A public viewer or an
old validation report is not evidence of dense spectral fusion for all plants.
Use the final corrected leaf review rather than superseded accuracy summaries.

## Completeness boundary

The package covers the enumerated local research tree and separately exported
website repository. Before closing the project, each team member must check for
additional camera recordings, notebooks or results held only on lab disks,
personal drives or other computers. Add any missing source location to the index
and verify its transfer. This cannot be certified from one laptop alone.
