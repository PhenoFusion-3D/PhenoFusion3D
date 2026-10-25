ARCHIVE CATALOGUE AND DELIVERY VERIFICATION

This script is read-only with respect to selected data. It writes only the specified manifest or verification report. It does not copy, delete or upload research files.

CREATE METADATA INVENTORY
python tools/handover/archive_catalogue.py create --root . --selection generated/delivery_review_20261026/archive_selection.json --output generated/delivery_review_20261026/new_inventory.json

CREATE FULL HASH MANIFEST
Freeze the selected result folders first. This reads every selected byte and can take a long time.
python tools/handover/archive_catalogue.py create --root . --selection generated/delivery_review_20261026/archive_selection.json --output generated/delivery_review_20261026/source_hash_manifest.json --sha256 --allow-large-hash

VERIFY THE RECEIVED COPY
Use one of the delivered per-archive *.manifest.json files. Set --root to the folder into which the ZIPs were extracted; paths under data/ and generated/ are relative to that folder.
python tools/handover/archive_catalogue.py verify --root "D:/PhenoFusion3D_Delivered" --manifest "D:/Delivery/archives/data__main__test_plant_10-7.zip.manifest.json" --report "D:/Delivery/verification/september_data.json"

Exit codes: 0 = successful selected-file check; 1 = file errors; 2 = invalid arguments, incomplete/metadata-only manifest or unsafe path.
A PASS report verifies manifest-listed bytes. It does not prove that every project source was included, that a remote account can access the archive, or that a stakeholder accepted delivery. Record those separately.

The selection JSON has a roots list. Each entry is a relative directory string or an object containing path. Rejects absolute paths, parent traversal and source symlinks. Typical cache and environment directories are excluded; review the exclusion list against your actual setup.
