import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from file_io.loader import load_image_pairs, load_intrinsics, get_default_intrinsics


class CalibrationWorker(QThread):
    """Estimate gantry axis and per-original-frame step from RGB-D frames."""

    done  = pyqtSignal(float, int)  # step_m_per_frame, axis
    error = pyqtSignal(str)

    def __init__(self, rgb_dir, depth_dir, intr_path, frame_gap=50):
        super().__init__()
        self.rgb_dir   = rgb_dir
        self.depth_dir = depth_dir
        self.intr_path = intr_path
        self.frame_gap = frame_gap

    def run(self):
        try:
            import cv2

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

            shift, response = cv2.phaseCorrelate(
                img_a.astype(np.float32),
                img_b.astype(np.float32),
            )
            shift_x, shift_y = float(shift[0]), float(shift[1])
            if response <= 0:
                raise RuntimeError(
                    f'Phase correlation failed: response={response:.4f}'
                )

            if abs(shift_x) >= abs(shift_y):
                axis = 0
                shift_ppf = shift_x / float(gap)
                focal = fx
            else:
                axis = 1
                shift_ppf = shift_y / float(gap)
                focal = fy

            depth = cv2.imread(depth_a, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError('Failed to read depth frame for calibration.')

            valid_depth = depth[(depth > 100) & (depth < 5000)]
            if valid_depth.size == 0:
                raise RuntimeError('No valid depth pixels in first depth frame.')

            median_depth_m = float(np.median(valid_depth)) / 1000.0
            step_m = abs(shift_ppf * median_depth_m / focal)
            if step_m <= 0:
                raise RuntimeError('Estimated gantry step is zero.')

            self.done.emit(float(step_m), int(axis))
        except Exception as e:
            self.error.emit(str(e))
