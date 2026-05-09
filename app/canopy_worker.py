"""
QThread wrapper for the canopy (top-down plant fusion) reconstruction pipeline.
"""
from PyQt5.QtCore import QThread, pyqtSignal

from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy


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
        stride: int = 10,
        max_frames: int = 9,
        max_candidates: int = 40,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_root    = dataset_root
        self.intrinsics_path = intrinsics_path
        self.depth_min       = depth_min
        self.depth_max       = depth_max
        self.stride          = stride
        self.max_frames      = max_frames
        self.max_candidates  = max_candidates

    def run(self):
        try:
            cfg = CanopyReconstructionConfig(
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                sample_stride=self.stride,
                max_frames=self.max_frames,
                max_candidates=self.max_candidates,
                auto_mask=True,
            )
            result = reconstruct_canopy(self.dataset_root, config=cfg)

            import open3d as o3d
            pcd = o3d.io.read_point_cloud(result.point_cloud_path)
            self.finished.emit(pcd, result.summary_path)
        except Exception as exc:
            self.error.emit(str(exc))
