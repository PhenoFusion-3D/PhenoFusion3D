import os
import math
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from file_io.loader   import load_image_pairs, load_intrinsics, get_default_intrinsics
from file_io.exporter import save_ply, save_metrics_csv
from app.worker       import ProcessingWorker
from app.canopy_worker import CanopyWorker
from app.canopy_sequence_worker import CanopySequenceWorker
from app.capture_worker import CaptureWorker
from app.calibration_worker import CalibrationWorker
from app.quality_worker import QualityWorker
from capture          import CaptureParams
from capture.gantry   import GantryController
from processing.quality import QualityParams, QualityThresholds
from processing.registration_agent import AgentConfig
from visualiser.viewer import PointCloudViewer


# Strict reconstruction acceptance thresholds (match QualityThresholds defaults)
DEFAULT_MIN_FITNESS = 0.3
DEFAULT_MAX_RMSE    = 0.015


class Controller(QObject):

    status_changed         = pyqtSignal(str)
    frame_processed        = pyqtSignal(int, int, object, float, float, str)
    reconstruction_complete = pyqtSignal(object, list, list)
    error_occurred         = pyqtSignal(str)

    # Capture pipeline signals
    capture_progress = pyqtSignal(int, int)
    capture_complete = pyqtSignal(str, int)
    capture_error    = pyqtSignal(str)

    # Quality pipeline signals
    quality_progress = pyqtSignal(int, int)
    quality_ready    = pyqtSignal(object)
    quality_error    = pyqtSignal(str)
    calibration_done = pyqtSignal(float, int)

    # Capture lifecycle (panel needs this to disable jog during capture).
    capture_started  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker          = None
        self.canopy_worker   = None
        self.canopy_sequence_worker = None
        self.capture_worker  = None
        self.quality_worker  = None
        self.calibration_worker = None
        self.viewer          = PointCloudViewer()
        self.final_pcd       = None
        self.all_metrics     = []
        self.n_success       = 0
        self.n_fail          = 0

        # Last known DataPanel paths (refreshed before quality / reconstruction)
        self._last_rgb_dir   = None
        self._last_depth_dir = None
        self._last_intr_path = None
        self._last_step_size = 1
        self._last_gantry_step_m_per_frame = 0.00127
        self._last_gantry_axis = 0
        self._last_depth_min_mm = 0
        self._last_depth_trunc = 3.5
        self._last_bbox = None
        self._last_enable_feature_init = False

        # Gantry controller -- ROS init is deferred to first call so this
        # is cheap on Windows / non-ROS hosts.
        self.gantry = GantryController()

    # ---------------------------------------------------------------- run
    @pyqtSlot(str, str, str, int, float, int, int, float, object, bool, bool, bool, float, bool, int, object)
    def on_run_clicked(self, rgb_dir, depth_dir, intrinsics_path, step_size,
                       gantry_step_m_per_frame, gantry_axis,
                       depth_min_mm, depth_trunc, bbox, enable_feature_init,
                       use_tsdf=False, mask_background=True, tsdf_voxel_m_ui=0.003,
                       use_canopy=False, canopy_stride=1, canopy_extras=None):
        self.n_success   = 0
        self.n_fail      = 0
        self.all_metrics = []
        self.final_pcd   = None

        # --- Canopy mode: runs a completely separate worker ---
        if use_canopy:
            extras = canopy_extras or {}
            if extras.get('sequence_mode'):
                self._start_canopy_sequence(
                    rgb_dir, depth_dir, intrinsics_path,
                    depth_min_mm, depth_trunc, canopy_stride, extras
                )
            else:
                self._start_canopy(rgb_dir, depth_dir, intrinsics_path,
                                   depth_min_mm, depth_trunc, canopy_stride,
                                   extras)
            return

        try:
            pairs = load_image_pairs(rgb_dir, depth_dir, step=step_size)
        except Exception as e:
            self.error_occurred.emit(f'Failed to load images:\n{e}')
            return

        intr = load_intrinsics(intrinsics_path) if intrinsics_path else None
        if intr:
            K, dist, _, _ = intr
        else:
            K, dist = get_default_intrinsics()

        self._last_rgb_dir   = rgb_dir
        self._last_depth_dir = depth_dir
        self._last_intr_path = intrinsics_path
        self._last_step_size = step_size
        self._last_gantry_step_m_per_frame = gantry_step_m_per_frame
        self._last_gantry_axis = gantry_axis
        self._last_depth_min_mm = depth_min_mm
        self._last_depth_trunc = depth_trunc
        self._last_bbox = bbox
        self._last_enable_feature_init = enable_feature_init

        self.status_changed.emit(f'Starting reconstruction: {len(pairs)} frames...')

        is_icl = 'icl' in rgb_dir.lower()

        depth_scale = 5000.0 if is_icl else 1000.0
        max_iter     = 30    if is_icl else 80
        erode        = True  if is_icl else False
        inpaint      = True  if is_icl else False

        # Optional plant-focused ICP. Keep disabled for full-scene sequence
        # reconstruction so the tray, box, rails, and background remain visible.
        bg_mask  = mask_background and not is_icl
        p_icp    = bg_mask   # plant_icp follows mask_background for non-ICL data

        # ICP mode: frame-to-frame colour ICP (stakeholder approach, no pose needed).
        #   voxel_size=0.01 gives ICP correspondence radius ~20mm which is large
        #   enough to bridge 0.45 px/frame sub-pixel motion at 2.5m depth.
        #   bbox is intentionally ignored so the full frame (including tray/floor
        #   background) provides stable ICP anchors.
        # TSDF mode: kinematic poses from gantry step+axis; needs accurate calibration.
        use_known_poses = is_icl or use_tsdf
        gantry_step_m   = gantry_step_m_per_frame * step_size
        # tsdf_voxel_m: use UI-supplied value when non-ICL, else fixed fine voxel for ICL.
        tsdf_voxel_m = 0.005 if is_icl else tsdf_voxel_m_ui

        if use_known_poses:
            voxel_size = 0.02 if is_icl else 0.005
            icp_bbox   = bbox
        else:
            # ICP: wider correspondence radius; full-frame background aids registration
            voxel_size = 0.02 if is_icl else 0.01
            icp_bbox   = None

        agent_config = AgentConfig(
            floor_min_fitness=DEFAULT_MIN_FITNESS,
            floor_max_rmse=DEFAULT_MAX_RMSE,
            enable_feature_init=enable_feature_init,
        )

        self.worker = ProcessingWorker(
            pairs=pairs, K=K, dist=dist,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            voxel_size=voxel_size,
            max_iter=max_iter,
            bbox=icp_bbox,
            gantry_step_m=gantry_step_m,
            gantry_axis=gantry_axis,
            depth_min_mm=depth_min_mm,
            erode=erode,
            inpaint=inpaint,
            use_known_poses=use_known_poses,
            tsdf_voxel_m=tsdf_voxel_m,
            min_fitness=DEFAULT_MIN_FITNESS,
            max_rmse=DEFAULT_MAX_RMSE,
            save_path=os.path.join(os.path.dirname(rgb_dir), 'output'),
            agent_config=agent_config,
            mask_background=bg_mask,
            plant_icp=p_icp,
        )
        self.worker.frame_done.connect(self._on_frame)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self.error_occurred)
        self.worker.start()
        self.viewer.start()

    def _start_canopy(self, rgb_dir, depth_dir, intrinsics_path,
                      depth_min_mm, depth_trunc, stride, canopy_extras=None):
        """Launch a CanopyWorker for the top-down plant fusion pipeline."""
        from pathlib import Path
        dataset_root = str(self._dataset_root_from_dirs(rgb_dir, depth_dir))
        self.status_changed.emit(f'Starting canopy reconstruction on {dataset_root}...')

        extras = canopy_extras or {}
        self.canopy_worker = CanopyWorker(
            dataset_root=dataset_root,
            intrinsics_path=intrinsics_path,
            depth_min=int(depth_min_mm) if depth_min_mm else 500,
            depth_max=int(depth_trunc * 1000) if depth_trunc else 4000,
            stride=int(stride),
            max_frames=int(extras.get('max_frames', 15)),
            max_candidates=int(extras.get('max_candidates', 0)),
            coverage_threshold=int(extras.get('coverage', 1)),
            smooth_sigma=float(extras.get('smooth_sigma', 2.0)),
            mask_sensitivity=str(extras.get('mask_sensitivity', 'default')),
            add_leaf_thickness=bool(extras.get('add_leaf_thickness', True)),
        )
        self.canopy_worker.finished.connect(self._on_canopy_finished)
        self.canopy_worker.error.connect(self.error_occurred)
        self.canopy_worker.start()
        self.viewer.start()

    def _start_canopy_sequence(self, rgb_dir, depth_dir, intrinsics_path,
                               depth_min_mm, depth_trunc, stride, canopy_extras=None):
        """Launch multi-plant canopy sequence reconstruction."""
        dataset_root = str(self._dataset_root_from_dirs(rgb_dir, depth_dir))
        extras = canopy_extras or {}
        self.status_changed.emit(f'Starting multi-plant canopy sequence on {dataset_root}...')
        self.canopy_sequence_worker = CanopySequenceWorker(
            dataset_root=dataset_root,
            depth_min=int(depth_min_mm) if depth_min_mm else 500,
            depth_max=int(depth_trunc * 1000) if depth_trunc else 4000,
            detection_stride=max(1, int(stride)),
            fusion_stride=1,
            max_frames=int(extras.get('max_frames', 15)),
            coverage_threshold=int(extras.get('coverage', 1)),
            smooth_sigma=float(extras.get('smooth_sigma', 0.8)),
            component_min_area=int(extras.get('component_min_area', 8000)),
            leaf_thickness=0.003 if extras.get('add_leaf_thickness', False) else 0.0,
        )
        self.canopy_sequence_worker.finished.connect(self._on_canopy_sequence_finished)
        self.canopy_sequence_worker.error.connect(self.error_occurred)
        self.canopy_sequence_worker.start()
        self.viewer.start()

    def _dataset_root_from_dirs(self, rgb_dir, depth_dir):
        from pathlib import Path
        rgb_path = Path(rgb_dir)
        depth_path = Path(depth_dir)
        if rgb_path.name.lower() == 'rgb' and depth_path.name.lower() == 'depth':
            return rgb_path.parent
        if any(rgb_path.glob('rgb_*.png')):
            return rgb_path
        return rgb_path.parent

    @pyqtSlot(object, str)
    def _on_canopy_finished(self, pcd, summary_str):
        self.final_pcd = pcd
        if pcd is not None and not pcd.is_empty():
            self.viewer.update(pcd)
        pt_count = len(pcd.points) if (pcd and not pcd.is_empty()) else 0
        self.status_changed.emit(
            f'Canopy done. {pt_count:,} points. Use File menu to export.'
        )
        self.reconstruction_complete.emit(pcd, [], [])

    @pyqtSlot(object, str)
    def _on_canopy_sequence_finished(self, pcd, summary_str):
        self.final_pcd = pcd
        if pcd is not None and not pcd.is_empty():
            self.viewer.update(pcd)
        pt_count = len(pcd.points) if (pcd and not pcd.is_empty()) else 0
        self.status_changed.emit(
            f'Canopy sequence done. {pt_count:,} points. Summary: {summary_str}'
        )
        self.reconstruction_complete.emit(pcd, [], [])

    @pyqtSlot(str, str, str, float, int)
    def on_calibrate_requested(self, rgb_dir, depth_dir, intrinsics_path, velocity_mps, fps):
        self.status_changed.emit('Calibrating gantry motion...')
        self.calibration_worker = CalibrationWorker(
            rgb_dir,
            depth_dir,
            intrinsics_path,
            velocity_mps=velocity_mps,
            fps=fps,
        )
        self.calibration_worker.done.connect(self._on_calibration_done)
        self.calibration_worker.error.connect(self._on_calibration_error)
        self.calibration_worker.start()

    @pyqtSlot(float, int)
    def _on_calibration_done(self, step_m, axis):
        self.status_changed.emit(
            f'Gantry calibration: step={step_m * 1000:.3f} mm/frame, axis={axis}'
        )
        self.calibration_done.emit(step_m, axis)
        self.calibration_worker = None

    @pyqtSlot(str)
    def _on_calibration_error(self, msg):
        self.status_changed.emit('Gantry calibration failed.')
        self.error_occurred.emit(f'Gantry calibration failed:\n{msg}')
        self.calibration_worker = None

    @pyqtSlot()
    def on_stop_clicked(self):
        if self.worker:
            self.worker.stop()
        self.status_changed.emit('Stopping...')

    @pyqtSlot(int, int, object, float, float, str)
    def _on_frame(self, idx, total, pcd, fitness, rmse, status):
        if status in ('OK', 'RECOVERED', 'INTEGRATED', 'WARN'):
            self.n_success += 1
        else:
            self.n_fail += 1
        self.all_metrics.append({'frame': idx, 'status': status, 'fitness': fitness, 'rmse': rmse})
        self.viewer.update(pcd)
        self.frame_processed.emit(idx, total, pcd, fitness, rmse, status)
        if status == 'INTEGRATED':
            metric_msg = f'depth={fitness * 100:.1f}%'
        elif isinstance(fitness, float) and math.isnan(fitness):
            metric_msg = status
        else:
            metric_msg = f'fitness={fitness:.4f}'
        self.status_changed.emit(f'Frame {idx + 1}/{total} | {metric_msg}')

    @pyqtSlot(object, list, list)
    def _on_finished(self, final_pcd, succeed, fail):
        self.final_pcd = final_pcd
        if final_pcd is not None and not final_pcd.is_empty():
            self.viewer.update(final_pcd)
        self.status_changed.emit(
            f'Done. {len(succeed)} frames succeeded, {len(fail)} failed. '
            f'Use File menu to export.'
        )
        self.reconstruction_complete.emit(final_pcd, succeed, fail)

    # ---------------------------------------------------------------- capture
    @pyqtSlot(str, str, float, float, int, float, bool, bool)
    def on_capture_clicked(self, backend_pref, out_root, velocity_mps,
                           end_position_m, fps, duration_s,
                           enable_depth_filters=True, preserve_raw_depth=False):
        params = CaptureParams(
            out_root=out_root or 'data/captures',
            fps=fps,
            velocity_mps=velocity_mps,
            end_position_m=end_position_m,
            duration_s=duration_s,
            enable_depth_filters=enable_depth_filters,
            preserve_raw_depth=preserve_raw_depth,
        )
        self.status_changed.emit(f'Capture starting (backend={backend_pref})...')
        self.capture_started.emit()
        self.capture_worker = CaptureWorker(backend_pref, params)
        self.capture_worker.frame_captured.connect(self.capture_progress)
        self.capture_worker.finished.connect(self._on_capture_finished)
        self.capture_worker.error.connect(self._on_capture_error)
        self.capture_worker.start()

    @pyqtSlot()
    def on_capture_stop(self):
        if self.capture_worker:
            self.capture_worker.stop()
        self.status_changed.emit('Capture stop requested...')

    @pyqtSlot(str, int)
    def _on_capture_finished(self, out_dir, n_frames):
        rgb_dir   = os.path.join(out_dir, 'rgb')
        depth_dir = os.path.join(out_dir, 'depth')
        intr_path = os.path.join(out_dir, 'kdc_intrinsics.txt')
        if not os.path.exists(intr_path):
            intr_path = ''
        self._last_rgb_dir   = rgb_dir
        self._last_depth_dir = depth_dir
        self._last_intr_path = intr_path
        self.status_changed.emit(f'Capture done. {n_frames} frames -> {out_dir}')
        self.capture_complete.emit(out_dir, n_frames)

    @pyqtSlot(str)
    def _on_capture_error(self, msg):
        self.status_changed.emit(f'Capture error: {msg}')
        self.capture_error.emit(msg)

    # ---------------------------------------------------------------- quality
    def _build_quality_params(self, rgb_dir: str) -> QualityParams:
        is_icl = 'icl' in rgb_dir.lower()
        return QualityParams(
            depth_scale=5000.0 if is_icl else 1000.0,
            depth_trunc=self._last_depth_trunc,
            voxel_size=0.02 if is_icl else 0.005,
            max_iter=30 if is_icl else 80,
            bbox=self._last_bbox,
            depth_min_mm=self._last_depth_min_mm,
            erode=is_icl,
            inpaint=is_icl,
            gantry_step_m=self._last_gantry_step_m_per_frame,
            gantry_axis=self._last_gantry_axis,
            thresholds=QualityThresholds(),
        )

    def on_quality_paths(
        self, rgb_dir, depth_dir, intrinsics_path, step_size,
        gantry_step_m_per_frame=None, gantry_axis=None,
        depth_min_mm=None, depth_trunc=None, bbox=None,
        enable_feature_init=None,
    ):
        """Capture the most recent DataPanel state for use by Quick/Full check."""
        self._last_rgb_dir   = rgb_dir
        self._last_depth_dir = depth_dir
        self._last_intr_path = intrinsics_path
        self._last_step_size = step_size
        if gantry_step_m_per_frame is not None:
            self._last_gantry_step_m_per_frame = gantry_step_m_per_frame
        if gantry_axis is not None:
            self._last_gantry_axis = gantry_axis
        if depth_min_mm is not None:
            self._last_depth_min_mm = depth_min_mm
        if depth_trunc is not None:
            self._last_depth_trunc = depth_trunc
        self._last_bbox = bbox
        if enable_feature_init is not None:
            self._last_enable_feature_init = enable_feature_init

    def _ensure_paths(self) -> tuple | None:
        if not self._last_rgb_dir or not self._last_depth_dir:
            self.quality_error.emit('Set RGB and depth folders first.')
            return None
        try:
            pairs = load_image_pairs(self._last_rgb_dir, self._last_depth_dir, step=1)
        except Exception as e:
            self.quality_error.emit(f'Failed to load images: {e}')
            return None
        intr = load_intrinsics(self._last_intr_path) if self._last_intr_path else None
        if intr:
            K, dist, _, _ = intr
        else:
            K, dist = get_default_intrinsics()
        return pairs, K, dist

    @pyqtSlot()
    def on_quick_check_clicked(self):
        loaded = self._ensure_paths()
        if loaded is None:
            return
        pairs, K, dist = loaded
        params = self._build_quality_params(self._last_rgb_dir)
        self.status_changed.emit(f'Quick quality check on {len(pairs)} frames...')
        self.quality_worker = QualityWorker(pairs, K, dist, params, mode='quick', n_samples=15)
        self.quality_worker.progress.connect(self.quality_progress)
        self.quality_worker.report_ready.connect(self._on_quality_ready)
        self.quality_worker.error.connect(self._on_quality_error)
        self.quality_worker.start()

    @pyqtSlot()
    def on_full_report_clicked(self):
        loaded = self._ensure_paths()
        if loaded is None:
            return
        pairs, K, dist = loaded
        params = self._build_quality_params(self._last_rgb_dir)
        # Save report next to the dataset
        out_dir = os.path.dirname(self._last_rgb_dir)
        self.status_changed.emit(f'Full quality report on {len(pairs)} frames...')
        self.quality_worker = QualityWorker(
            pairs, K, dist, params, mode='full', out_dir=out_dir,
        )
        self.quality_worker.progress.connect(self.quality_progress)
        self.quality_worker.report_ready.connect(self._on_quality_ready)
        self.quality_worker.error.connect(self._on_quality_error)
        self.quality_worker.start()

    @pyqtSlot(object)
    def _on_quality_ready(self, report):
        self.status_changed.emit(
            f'Quality: {report.verdict} ({report.n_pairs_evaluated} pairs)'
        )
        self.quality_ready.emit(report)

    @pyqtSlot(str)
    def _on_quality_error(self, msg):
        self.status_changed.emit(f'Quality error: {msg}')
        self.quality_error.emit(msg)

    # ---------------------------------------------------------------- gantry
    @pyqtSlot(float)
    def on_gantry_jog(self, velocity_mps: float):
        if velocity_mps == 0.0:
            self.gantry.stop()
        else:
            self.gantry.start_jog(velocity_mps)

    @pyqtSlot()
    def on_gantry_stop(self):
        self.gantry.stop()

    @pyqtSlot(float)
    def on_gantry_goto(self, position_m: float):
        self.gantry.go_to(position_m)

    @pyqtSlot()
    def on_gantry_home(self):
        self.gantry.go_home()

    def shutdown(self):
        """Called from MainWindow.closeEvent. Final safety stop +
        unregister subscribers."""
        try:
            self.gantry.shutdown()
        except Exception:
            pass

    # ---------------------------------------------------------------- export
    def export_ply(self, path):
        if self.final_pcd:
            ok = save_ply(self.final_pcd, path)
            self.status_changed.emit(f'PLY saved: {path}' if ok else 'PLY export failed.')

    def export_csv(self, path):
        ok = save_metrics_csv(self.all_metrics, path)
        self.status_changed.emit(f'CSV saved: {path}' if ok else 'CSV export failed.')
