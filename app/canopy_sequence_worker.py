"""
QThread wrapper for multi-plant canopy sequence reconstruction.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import open3d as o3d
from PyQt5.QtCore import QThread, pyqtSignal


class CanopySequenceWorker(QThread):
    """Run ``reconstruct_canopy_sequence.py`` in a background thread."""

    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        dataset_root: str,
        depth_min: int = 500,
        depth_max: int = 4_000,
        detection_stride: int = 3,
        fusion_stride: int = 1,
        max_frames: int = 15,
        coverage_threshold: int = 1,
        smooth_sigma: float = 0.8,
        component_min_area: int = 8_000,
        leaf_thickness: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_root = dataset_root
        self.depth_min = int(depth_min)
        self.depth_max = int(depth_max)
        self.detection_stride = max(1, int(detection_stride))
        self.fusion_stride = max(1, int(fusion_stride))
        self.max_frames = max(3, int(max_frames))
        self.coverage_threshold = max(1, int(coverage_threshold))
        self.smooth_sigma = float(smooth_sigma)
        self.component_min_area = max(1, int(component_min_area))
        self.leaf_thickness = max(0.0, float(leaf_thickness))

    def run(self):
        try:
            root = Path(self.dataset_root).resolve()
            script = Path(__file__).resolve().parents[1] / "reconstruct_canopy_sequence.py"
            out_dir = root / "canopy_sequence_ui"
            cmd = [
                sys.executable,
                str(script),
                "--input", str(root),
                "--output", str(out_dir),
                "--component-instances",
                "--stride", str(self.detection_stride),
                "--fusion-stride", str(self.fusion_stride),
                "--max-frames", str(self.max_frames),
                "--reference-spacing-m", "0.08",
                "--min-score-ratio", "0.25",
                "--component-min-area", str(self.component_min_area),
                "--component-edge-penalty", "0.20",
                "--track-max-step-px", "55",
                "--track-max-area-ratio", "4.0",
                "--track-overlap-ratio", "0.55",
                "--coverage", str(self.coverage_threshold),
                "--smooth-sigma", str(self.smooth_sigma),
                "--depth-min", str(self.depth_min),
                "--depth-max", str(self.depth_max),
                "--leaf-thickness", str(self.leaf_thickness),
                "--max-hole-fill-px", "8",
                "--max-triangle-jump", "0.025",
                "--canopy-sheet",
                "--sheet-pixel-step", "2",
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                details = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(details[-2000:] or "Canopy sequence reconstruction failed.")

            pcd_path = out_dir / "sequence_points.ply"
            if pcd_path.exists():
                pcd = o3d.io.read_point_cloud(str(pcd_path))
            else:
                pcd = o3d.geometry.PointCloud()
            self.finished.emit(pcd, str(out_dir / "sequence_summary.json"))
        except Exception as exc:
            self.error.emit(str(exc))
