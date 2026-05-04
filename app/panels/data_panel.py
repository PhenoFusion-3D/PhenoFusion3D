from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QSpinBox, QFileDialog, QMessageBox,
    QDoubleSpinBox, QComboBox, QGroupBox, QFormLayout, QCheckBox
)
from PyQt5.QtCore import pyqtSignal
import os

from file_io.loader import load_gantry_config, load_session_json


class DataPanel(QWidget):

    # rgb_dir, depth_dir, intrinsics, step, gantry_step_m, gantry_axis,
    # depth_min_mm, depth_trunc_m, bbox, enable_feature_init, use_tsdf
    run_requested       = pyqtSignal(str, str, str, int, float, int, int, float, object, bool, bool)
    calibrate_requested = pyqtSignal(str, str, str, float, int)
    stop_requested      = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel('Data Loading')
        title.setStyleSheet('font-weight:bold; font-size:14px;')
        layout.addWidget(title)

        self.rgb_edit   = self._add_folder_row(layout, 'RGB Images:')
        self.depth_edit = self._add_folder_row(layout, 'Depth Images:')
        self.intr_edit  = self._add_file_row(layout,   'Intrinsics JSON:', optional=True)

        # Step size
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel('Step Size:'))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 20)
        self.step_spin.setValue(2)
        self.step_spin.setToolTip('Use every Nth frame (2 = every other frame)')
        step_row.addWidget(self.step_spin)
        step_row.addStretch()
        layout.addLayout(step_row)

        # Advanced gantry / depth parameters
        advanced = QGroupBox('Advanced / Gantry')
        advanced.setCheckable(True)
        advanced.setChecked(True)
        advanced_layout = QFormLayout(advanced)
        advanced_layout.setContentsMargins(8, 8, 8, 8)
        advanced_layout.setSpacing(6)

        self.recon_mode_combo = QComboBox()
        self.recon_mode_combo.addItem('ICP  (frame-to-frame, recommended)', userData=False)
        self.recon_mode_combo.addItem('TSDF (known poses, requires calibration)', userData=True)
        self.recon_mode_combo.setCurrentIndex(0)
        self.recon_mode_combo.setToolTip(
            'ICP: registers each frame against the previous using colour ICP — '
            'no gantry step or axis knowledge needed.\n'
            'TSDF: integrates frames using exact kinematic camera poses — '
            'requires accurate Gantry Step and Axis calibration.'
        )
        advanced_layout.addRow('Recon Mode:', self.recon_mode_combo)

        self.gantry_step_spin = QDoubleSpinBox()
        self.gantry_step_spin.setRange(0.01, 50.0)
        self.gantry_step_spin.setDecimals(3)
        self.gantry_step_spin.setSingleStep(0.01)
        self.gantry_step_spin.setValue(1.27)
        self.gantry_step_spin.setSuffix(' mm/frame')
        self.gantry_step_spin.setToolTip('Gantry travel per original captured frame.')
        advanced_layout.addRow('Gantry Step:', self.gantry_step_spin)

        self.gantry_axis_combo = QComboBox()
        self.gantry_axis_combo.addItems([
            '0 - X (horizontal)',
            '1 - Y (vertical)',
        ])
        self.gantry_axis_combo.setCurrentIndex(0)
        advanced_layout.addRow('Gantry Axis:', self.gantry_axis_combo)

        self.gantry_velocity_spin = QDoubleSpinBox()
        self.gantry_velocity_spin.setRange(0.001, 2.0)
        self.gantry_velocity_spin.setDecimals(3)
        self.gantry_velocity_spin.setSingleStep(0.005)
        self.gantry_velocity_spin.setValue(0.038)
        self.gantry_velocity_spin.setSuffix(' m/s')
        self.gantry_velocity_spin.setToolTip(
            'Capture gantry velocity. Used to compute step = velocity / fps.'
        )
        advanced_layout.addRow('Velocity:', self.gantry_velocity_spin)

        self.gantry_fps_spin = QSpinBox()
        self.gantry_fps_spin.setRange(1, 120)
        self.gantry_fps_spin.setValue(30)
        self.gantry_fps_spin.setSuffix(' fps')
        self.gantry_fps_spin.setToolTip(
            'Capture frame rate. Used to compute step = velocity / fps.'
        )
        advanced_layout.addRow('FPS:', self.gantry_fps_spin)

        self.depth_min_spin = QSpinBox()
        self.depth_min_spin.setRange(0, 5000)
        self.depth_min_spin.setValue(0)
        self.depth_min_spin.setSuffix(' mm')
        self.depth_min_spin.setToolTip('Discard depth closer than this value. 0 disables near clipping.')
        advanced_layout.addRow('Depth Min:', self.depth_min_spin)

        self.depth_trunc_spin = QDoubleSpinBox()
        self.depth_trunc_spin.setRange(0.5, 10.0)
        self.depth_trunc_spin.setDecimals(2)
        self.depth_trunc_spin.setSingleStep(0.1)
        self.depth_trunc_spin.setValue(3.5)
        self.depth_trunc_spin.setSuffix(' m')
        self.depth_trunc_spin.setToolTip('Discard depth farther than this value.')
        advanced_layout.addRow('Depth Trunc:', self.depth_trunc_spin)

        bbox_row = QHBoxLayout()
        self.bbox_x1_spin = self._bbox_spin()
        self.bbox_y1_spin = self._bbox_spin()
        self.bbox_x2_spin = self._bbox_spin()
        self.bbox_y2_spin = self._bbox_spin()
        for label, spin in (
            ('x1', self.bbox_x1_spin),
            ('y1', self.bbox_y1_spin),
            ('x2', self.bbox_x2_spin),
            ('y2', self.bbox_y2_spin),
        ):
            bbox_row.addWidget(QLabel(label))
            bbox_row.addWidget(spin)
        self.detect_roi_btn = QPushButton('Detect ROI')
        self.detect_roi_btn.setEnabled(False)
        self.detect_roi_btn.setToolTip('Automatic plant ROI detection is planned; enter bbox manually for now.')
        bbox_row.addWidget(self.detect_roi_btn)
        advanced_layout.addRow('BBox:', bbox_row)

        self.feature_init_check = QCheckBox('Enable FPFH init (slow)')
        self.feature_init_check.setToolTip(
            'Use feature-based global initialization as a final ICP recovery strategy.'
        )
        advanced_layout.addRow('ICP Recovery:', self.feature_init_check)

        self.calibrate_btn = QPushButton('Calibrate Gantry')
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setToolTip('Estimate gantry axis and step from RGB/depth frames.')
        self.calibrate_btn.clicked.connect(self._on_calibrate)
        advanced_layout.addRow(self.calibrate_btn)

        layout.addWidget(advanced)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton('Run Reconstruction')
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            'QPushButton { background:#2563eb; color:white; border-radius:4px; padding:6px; font-weight:bold; }'
            'QPushButton:disabled { background:#94a3b8; }'
        )
        self.run_btn.clicked.connect(self._on_run)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            'QPushButton { background:#dc2626; color:white; border-radius:4px; padding:6px; font-weight:bold; }'
            'QPushButton:disabled { background:#94a3b8; }'
        )
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _add_folder_row(self, parent_layout, label):
        parent_layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText('Select folder...')
        browse = QPushButton('Browse')
        browse.setFixedWidth(60)
        browse.clicked.connect(lambda: self._browse_folder(edit))
        row.addWidget(edit)
        row.addWidget(browse)
        parent_layout.addLayout(row)
        return edit

    def _bbox_spin(self):
        spin = QSpinBox()
        spin.setRange(0, 10000)
        spin.setValue(0)
        spin.setFixedWidth(70)
        return spin

    def _add_file_row(self, parent_layout, label, optional=False):
        lbl_row = QHBoxLayout()
        lbl_row.addWidget(QLabel(label))
        if optional:
            opt = QLabel('(optional)')
            opt.setStyleSheet('color:#94a3b8; font-size:11px;')
            lbl_row.addWidget(opt)
        lbl_row.addStretch()
        parent_layout.addLayout(lbl_row)

        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText('Select file... (default intrinsics used if blank)')
        browse = QPushButton('Browse')
        browse.setFixedWidth(60)
        browse.clicked.connect(lambda: self._browse_file(edit))
        row.addWidget(edit)
        row.addWidget(browse)
        parent_layout.addLayout(row)
        return edit

    def _browse_folder(self, edit):
        path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if path:
            edit.setText(path)
            self._load_saved_gantry_config()
            self._load_session_json_into_ui()
            self._validate()

    def _browse_file(self, edit):
        path, _ = QFileDialog.getOpenFileName(self, 'Select File', '', 'JSON/Text (*.txt *.json)')
        if path:
            edit.setText(path)

    def _validate(self):
        rgb_ok   = bool(self.rgb_edit.text())
        depth_ok = bool(self.depth_edit.text())
        self.run_btn.setEnabled(rgb_ok and depth_ok)
        self.calibrate_btn.setEnabled(rgb_ok and depth_ok)

    def _on_run(self):
        rgb_dir   = self.rgb_edit.text()
        depth_dir = self.depth_edit.text()
        intr_path = self.intr_edit.text()
        step      = self.step_spin.value()
        gantry_step_m = self.gantry_step_spin.value() / 1000.0
        gantry_axis   = self.gantry_axis_combo.currentIndex()
        depth_min_mm  = self.depth_min_spin.value()
        depth_trunc   = self.depth_trunc_spin.value()
        bbox          = self._bbox_from_controls()
        enable_feature_init = self.feature_init_check.isChecked()
        use_tsdf = bool(self.recon_mode_combo.currentData())

        # Quick count check
        import glob
        rgb_count = len(glob.glob(os.path.join(rgb_dir, '*.png')))
        if rgb_count == 0:
            QMessageBox.warning(self, 'No Images', f'No PNG files found in:\n{rgb_dir}')
            return

        self.set_running(True)
        self.run_requested.emit(
            rgb_dir, depth_dir, intr_path, step,
            gantry_step_m, gantry_axis, depth_min_mm, depth_trunc,
            bbox, enable_feature_init, use_tsdf
        )

    def _on_calibrate(self):
        rgb_dir   = self.rgb_edit.text()
        depth_dir = self.depth_edit.text()
        intr_path = self.intr_edit.text()
        if not rgb_dir or not depth_dir:
            QMessageBox.warning(self, 'Missing Data', 'Select RGB and depth folders first.')
            return
        self.calibrate_requested.emit(
            rgb_dir,
            depth_dir,
            intr_path,
            self.gantry_velocity_spin.value(),
            self.gantry_fps_spin.value(),
        )

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.calibrate_btn.setEnabled(not running)

    def set_paths(self, rgb_dir: str, depth_dir: str, intrinsics: str = ''):
        """Programmatically populate paths (called after a successful capture)."""
        self.rgb_edit.setText(rgb_dir or '')
        self.depth_edit.setText(depth_dir or '')
        self.intr_edit.setText(intrinsics or '')
        self._load_saved_gantry_config()
        self._load_session_json_into_ui()
        self._validate()

    def set_gantry_params(self, step_m: float, axis: int):
        """Populate gantry controls from calibration results."""
        self.gantry_step_spin.setValue(step_m * 1000.0)
        self.gantry_axis_combo.setCurrentIndex(max(0, min(1, int(axis))))

    def _bbox_from_controls(self):
        x1 = self.bbox_x1_spin.value()
        y1 = self.bbox_y1_spin.value()
        x2 = self.bbox_x2_spin.value()
        y2 = self.bbox_y2_spin.value()
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
        return None

    def _load_saved_gantry_config(self):
        for path in (self.rgb_edit.text(), self.depth_edit.text()):
            cfg = load_gantry_config(path)
            if cfg:
                step_m, axis = cfg
                self.set_gantry_params(step_m, axis)
                return

    def _load_session_json_into_ui(self):
        for path in (self.rgb_edit.text(), self.depth_edit.text()):
            session = load_session_json(path)
            if not session:
                continue
            velocity_mps = session.get('velocity_mps')
            fps = session.get('fps')
            if velocity_mps:
                self.gantry_velocity_spin.setValue(float(velocity_mps))
            if fps:
                self.gantry_fps_spin.setValue(int(fps))
            return
