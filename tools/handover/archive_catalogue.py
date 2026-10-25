"""Inventory selected research roots and verify their delivered copy.

Metadata inventory is the default. SHA-256 generation is explicit and can take
hours on large archives. No files are copied, changed, deleted, or uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

SCHEMA = "phenofusion3d.archive-manifest.v1"
EXCLUDE_DIRS = {".git", ".venv", "venv", ".venv-linux", "isolated_linux_env", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def inside(root: Path, relative: str) -> Path:
    """Reject traversal, absolute paths and symlink escapes in manifest input."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not relative:
        raise ValueError(f"Expected safe relative path: {relative!r}")
    path = root.joinpath(rel)
    path.resolve().relative_to(root.resolve())
    return path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def create(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    selected = selection.get("roots", [])
    if not selected:
        raise ValueError("Selection must contain a non-empty roots array.")
    output = args.output.resolve()
    records, errors, seen = [], [], set()
    for entry in selected:
        relroot = entry if isinstance(entry, str) else entry["path"]
        directory = inside(root, relroot)
        if not directory.exists() or directory.is_symlink():
            errors.append({"path": relroot, "error": "Selected path missing or a symlink"})
            continue
        if directory.is_file():
            if output == directory.resolve():
                raise ValueError("Manifest must not overwrite a selected source file.")
            relative = directory.relative_to(root).as_posix()
            if relative not in seen:
                seen.add(relative)
                stat = directory.stat()
                records.append({"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": None})
            continue
        if output.is_relative_to(directory.resolve()):
            raise ValueError("Write the manifest outside the selected archive roots to protect source files.")
        def walk_error(error):
            errors.append({"path": str(error.filename), "error": str(error)})
        for base, dirs, files in os.walk(directory, followlinks=False, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not Path(base, d).is_symlink())
            for name in sorted(files):
                path = Path(base, name)
                if path.is_symlink() or path.resolve() == output:
                    continue
                relative = path.relative_to(root).as_posix()
                if relative in seen:
                    continue
                seen.add(relative)
                try:
                    stat = path.stat()
                    records.append({"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": None})
                except OSError as error:
                    errors.append({"path": relative, "error": str(error)})
    total = sum(item["bytes"] for item in records)
    print(f"Selected {len(records):,} files; {total / 1024**3:.2f} GiB", flush=True)
    if args.sha256 and total > 10 * 1024**3 and not args.allow_large_hash:
        raise ValueError("Hashing exceeds 10 GiB. Review selection and rerun with --allow-large-hash when ready.")
    if args.sha256:
        for index, item in enumerate(records, 1):
            path = inside(root, item["path"])
            try:
                before = path.stat()
                if (before.st_size, before.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                    raise OSError("Source changed after inventory; freeze results and rerun.")
                item["sha256"] = digest(path)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    item["sha256"] = None
                    raise OSError("Source changed while hashing; freeze results and rerun.")
            except OSError as error:
                errors.append({"path": item["path"], "error": str(error)})
            if index % 100 == 0 or index == len(records):
                print(f"Hashed {index:,}/{len(records):,}", flush=True)
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "selection": selected,
        "excluded_directory_names": sorted(EXCLUDE_DIRS),
        "scope_note": "Selected roots only; review project sources and historical datasets for completeness before acceptance.",
        "mode": "sha256" if args.sha256 else "metadata_only",
        "files": records,
        "file_count": len(records),
        "total_bytes": total,
        "errors": errors,
        "ready_for_checksum_verification": bool(args.sha256 and records and not errors and all(x["sha256"] for x in records)),
        "stakeholder_receipt_confirmed": False,
    }
    write_json(output, result)
    print(f"Saved {output}")
    return 1 if errors else 0


def verify(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != SCHEMA or not manifest.get("ready_for_checksum_verification"):
        raise ValueError("Use a complete SHA-256 manifest, not a metadata-only or incomplete catalogue.")
    files = manifest.get("files", [])
    if not files or len(files) != manifest.get("file_count") or any(not x.get("sha256") for x in files):
        raise ValueError("Manifest file count or SHA-256 coverage is incomplete.")
    root = args.root.resolve()
    report = args.report.resolve()
    if report == args.manifest.resolve() or any(report == inside(root, x["path"]).resolve() for x in files):
        raise ValueError("Verification report must not overwrite the manifest or a delivered file.")
    failures, passed = [], 0
    for index, item in enumerate(files, 1):
        try:
            path = inside(root, item["path"])
            if path.is_symlink():
                raise ValueError("Symlink is not an independently delivered regular file.")
            before = path.stat()
            if before.st_size != item["bytes"]:
                raise ValueError("File size differs.")
            if digest(path) != item["sha256"]:
                raise ValueError("SHA-256 differs.")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("File changed while verifying.")
            passed += 1
        except (OSError, ValueError) as error:
            failures.append({"path": item["path"], "error": str(error)})
        if index % 100 == 0 or index == len(files):
            print(f"Verified {index:,}/{len(files):,}", flush=True)
    result = {"schema": "phenofusion3d.archive-verification.v1", "verified_utc": datetime.now(timezone.utc).isoformat(), "manifest": str(args.manifest.resolve()), "manifest_sha256": digest(args.manifest), "destination_root": str(root), "files_passed": passed, "files_failed": len(failures), "failures": failures, "status": "PASS" if not failures else "FAIL", "scope_note": "All manifest-listed files checked; does not certify unlisted files, project completeness, remote account access or stakeholder acceptance."}
    write_json(args.report.resolve(), result)
    print(result["status"])
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("create", help="Inventory selected roots; optionally hash files")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--sha256", action="store_true")
    build.add_argument("--allow-large-hash", action="store_true")
    check = sub.add_parser("verify", help="Verify delivered files against a complete SHA-256 manifest")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        return create(args) if args.command == "create" else verify(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
