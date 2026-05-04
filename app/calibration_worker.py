import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from file_io.loader import (
    load_image_pairs, load_intrinsics, get_default_intrinsics,
    save_gantry_config,
)


class CalibrationWorker(QThread):
    """Estimate gantry axis and per-original-frame step from capture settings."""

    done  = pyqtSignal(float, int)  # step_m_per_frame, axis
    error = pyqtSignal(str)

    def __init__(self, rgb_dir, depth_dir, intr_path, velocity_mps=0.0, fps=0, frame_gap=50):
        super().__init__()
        self.rgb_dir      = rgb_dir
        self.depth_dir    = depth_dir
        self.intr_path    = intr_path
        self.velocity_mps = float(velocity_mps or 0.0)
        self.fps          = int(fps or 0)
        self.frame_gap    = frame_gap

    def run(self):
        try:
            physics_step_m = None
            if self.velocity_mps > 0.0 and self.fps > 0:
                physics_step_m = self.velocity_mps / float(self.fps)

            pairs = load_image_pairs(self.rgb_dir, self.depth_dir, step=1)
            if len(pairs) < 2:
                raise RuntimeError('Need at least 2 RGB/depth frames for calibration.')

            gap = min(self.frame_gap, len(pairs) - 1)
            rgb_a, depth_a = pairs[0]
            rgb_b, _ = pairs[gap]

            img_a = cv2.imread(rgb_a, cv2.IMREAD_GRAYSCALE)
            img_b = cv2.imread(rgb_b, cv2.IMREAD_GRAYSCALE)
            if img_a is None or img_b is None:
                raise RuntimeError('Failed to read RGB frames for calibration.')
            if img_a.shape != img_b.shape:
                raise RuntimeError(
                    f'Calibration frames have different sizes: {img_a.shape} vs {img_b.shape}'
                )

            intr = load_intrinsics(self.intr_path) if self.intr_path else None
            if intr:
                K, _, _, _ = intr
            else:
                h, w = img_a.shape[:2]
                K, _ = get_default_intrinsics(w, h)

            fx, fy = float(K[0, 0]), float(K[1, 1])

            phase_step_m = None
            phase_axis = 0
            try:
                shift, response = cv2.phaseCorrelate(
                    img_a.astype(np.float32),
                    img_b.astype(np.float32),
                )
                if response <= 0:
                    raise RuntimeError(f'response={response:.4f}')

                shift_x, shift_y = float(shift[0]), float(shift[1])
                if abs(shift_x) >= abs(shift_y):
                    phase_axis = 0
                    shift_ppf = shift_x / float(gap)
                    focal = fx
                else:
                    phase_axis = 1
                    shift_ppf = shift_y / float(gap)
                    focal = fy

                depth = cv2.imread(depth_a, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise RuntimeError('failed to read depth frame')

                valid_depth = depth[(depth > 100) & (depth < 5000)]
                if valid_depth.size == 0:
                    raise RuntimeError('no valid depth pixels')

                median_depth_m = float(np.median(valid_depth)) / 1000.0
                phase_step_m = abs(shift_ppf * median_depth_m / focal)
            except Exception as e:
                print(f'[calib] WARNING: phase-corr sanity check failed: {e}')

            if physics_step_m is not None:
                step_m = physics_step_m
                # Trust phase-corr for axis direction (detects which image axis
                # has larger motion even for sub-pixel per-frame shifts); fall
                # back to axis=0 only when phase-corr itself failed.
                axis = phase_axis if phase_step_m is not None else 0
                if phase_step_m and phase_step_m > 0:
                    ratio = max(phase_step_m, physics_step_m) / min(phase_step_m, physics_step_m)
                    if ratio > 3.0:
                        print(
                            '[calib] WARNING: phase-corr magnitude '
                            f'({phase_step_m:.6f} m) differs >3x from '
                            f'physics ({physics_step_m:.6f} m); using physics magnitude, '
                            f'phase-corr axis={axis}.'
                        )
                    else:
                        print(
                            f'[calib] step={step_m:.6f} m (physics), axis={axis} (phase-corr).'
                        )
            elif phase_step_m and phase_step_m > 0:
                step_m = phase_step_m
                axis = phase_axis
            else:
                raise RuntimeError('Estimated gantry step is zero.')

            save_gantry_config(self.rgb_dir, step_m, axis)
            self.done.emit(float(step_m), int(axis))
        except Exception as e:
            self.error.emit(str(e))
