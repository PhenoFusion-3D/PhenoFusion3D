#!/usr/bin/env python3
"""
Convenience entrypoint: batch-reorganize every immediate subfolder under data/main
that contains rgb_*.png and depth_*.png into ICL-style rgb/ and depth/ trees.

Example:
  python scripts/reorganize_data_main.py --dry-run
  python scripts/reorganize_data_main.py --move

Extra CLI flags are forwarded to reorganize_to_icl_layout (e.g. --camera 0, --step 2).
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = ["--batch-root", str(_ROOT / "data" / "main")]

_spec = importlib.util.spec_from_file_location(
    "reorganize_to_icl_layout",
    Path(__file__).resolve().parent / "reorganize_to_icl_layout.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    sys.exit(_mod.main(_DEFAULT + sys.argv[1:]))
