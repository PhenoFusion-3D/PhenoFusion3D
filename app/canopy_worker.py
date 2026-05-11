"""
QThread wrapper for the canopy (top-down plant fusion) reconstruction pipeline.
"""
from PyQt5.QtCore import QThread, pyqtSignal

from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy

# Mask sensitivity presets — map UI label → config field overrides
_MASK_PRESETS = {
    "loose":   {"mask_s_min": 25, "mask_v_min": 20, "mask_exg_min": 10, "min_mask_area": 100_000},
    "default": {},
    "strict":  {"mask_s_min": 65, "mask_v_min": 55, "mask_exg_min": 35, "min_mask_area": 200_000},
}


class CanopyWorker(QThread):
    """Run :func:`reconstruct_canopy` in a background thread.

    Signals
    -------
    finished(pcd, summary_str):
        Emitted when the pipeline completes.  *pcd* is the final
        Open3D PointCloud; *summary_str* is the path to the JSON summary.
    error(str):
        Emitted on exception.
    """

    finished = pyqtSignal(object, str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        dataset_root: str,
        intrinsics_path: str = '',
        depth_min: int = 500,
        depth_max: int = 4_000,
        stride: int = 1,
        max_frames: int = 15,
        max_candidates: int = 0,
        coverage_threshold: int = 1,
        smooth_sigma: float = 2.0,
        mask_sensitivity: str = 'default',
        add_leaf_thickness: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_root       = dataset_root
        self.intrinsics_path    = intrinsics_path
        self.depth_min          = depth_min
        self.depth_max          = depth_max
        self.stride             = stride
        self.max_frames         = max_frames
        self.max_candidates     = max_candidates
        self.coverage_threshold = coverage_threshold
        self.smooth_sigma       = smooth_sigma
        self.mask_sensitivity   = mask_sensitivity
        self.add_leaf_thickness = add_leaf_thickness

    def run(self):
        try:
            mask_overrides = _MASK_PRESETS.get(self.mask_sensitivity, {})
            cfg = CanopyReconstructionConfig(
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                sample_stride=self.stride,
                max_frames=self.max_frames,
                max_candidates=self.max_candidates,
                coverage_threshold=self.coverage_threshold,
                smooth_sigma=self.smooth_sigma,
                add_leaf_thickness=self.add_leaf_thickness,
                auto_mask=True,
                **mask_overrides,
            )
            result = reconstruct_canopy(self.dataset_root, config=cfg)

            import open3d as o3d
            pcd = o3d.io.read_point_cloud(result.point_cloud_path)
            self.finished.emit(pcd, result.summary_path)
        except Exception as exc:
            self.error.emit(str(exc))
