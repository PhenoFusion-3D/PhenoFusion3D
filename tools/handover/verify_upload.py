"""Verify the downloaded handover folder against its SHA256SUMS.txt (Python 3)."""
import argparse
import hashlib
from pathlib import Path
import re
import sys


def verify(root: Path, sums: Path) -> int:
    root = root.resolve(strict=True)
    seen = set()
    failures = 0
    count = 0
    for line in sums.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{64})  (.+)", line)
        if not match:
            raise ValueError("Malformed checksum line")
        expected, relative = match.groups()
        if "\\" in relative or ":" in relative:
            raise ValueError("Unsafe checksum path: " + relative)
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts or relative in seen:
            raise ValueError("Unsafe or repeated checksum path: " + relative)
        seen.add(relative)
        target = (root / raw).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Checksum target escapes package folder: " + relative)
        count += 1
        try:
            print("Checking " + relative, flush=True)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().lower() != expected.lower():
                raise ValueError("SHA-256 mismatch")
        except (OSError, ValueError) as exc:
            failures += 1
            print("FAIL " + relative + ": " + str(exc), flush=True)
    if not count:
        raise ValueError("Empty checksum list")
    print(f"Checked {count} files; failures: {failures}.")
    print("PASS: listed bytes match." if not failures else "FAIL: restore the missing or changed files.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checksums", type=Path, default=Path("SHA256SUMS.txt"))
    args = parser.parse_args()
    try:
        return verify(args.root, args.checksums)
    except (ValueError, OSError) as exc:
        print("Invalid verification request: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
