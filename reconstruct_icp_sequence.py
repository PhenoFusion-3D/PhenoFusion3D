#!/usr/bin/env python3
"""
Sequential RGB-D reconstruction with colored ICP (stakeholder 3D_recons.py logic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_registration_agent import RegistrationAgent
from pipeline_reconstructor import Reconstructor, ReconstructorConfig


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Sequential colored ICP merge of RGB-D frames.")
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Subsampling step across sorted frame pairs (stakeholder step_size).",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max sampled frames to process (default: all usable with integer division).",
    )
    ap.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort RGB and depth before projection.",
    )
    ap.add_argument(
        "--no-erode",
        action="store_true",
        help="Disable depth-edge erosion (flying-pixel suppression is on by default).",
    )
    ap.add_argument(
        "--allow-rotation",
        action="store_true",
        help=(
            "Allow ICP to accumulate rotation in the pose chain. "
            "Default (off): strip the rotation component from each per-step ICP result "
            "and keep only translation, which is correct for a linear-translation gantry "
            "and prevents systematic rotation drift from smearing the plant cloud."
        ),
    )
    ap.add_argument(
        "--icp-voxel",
        type=float,
        default=0.01,
        help="Voxel size for colored ICP downsampling (metres).",
    )
    ap.add_argument(
        "--depth-trunc",
        type=float,
        default=2.5,
        help="Max depth metres passed to RGBD fusion (dataset depth in mm).",
    )
    ap.add_argument(
        "--plant-icp",
        action="store_true",
        help=(
            "Mask non-plant RGB-D pixels before ICP so registration is driven by "
            "leaf/stem geometry instead of gantry metal."
        ),
    )
    args = ap.parse_args()
    if args.step < 1:
        raise SystemExit("--step must be >= 1")

    config = ReconstructorConfig(
        data=args.data,
        step=args.step,
        max_frames=args.max_frames,
        undistort=args.undistort,
        erode_depth_edges=not args.no_erode,
        allow_rotation=args.allow_rotation,
        icp_voxel=args.icp_voxel,
        depth_trunc=args.depth_trunc,
        plant_icp=args.plant_icp,
    )
    Reconstructor(config=config, agent=RegistrationAgent()).run()


if __name__ == "__main__":
    main()
