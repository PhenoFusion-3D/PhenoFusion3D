"""Offline dependency and geometry check; never opens a camera or moves a gantry."""
import json
import os
from importlib.metadata import version

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
app = QApplication([])
import cv2
import numpy as np
import open3d as o3d
import pandas
import spectral
from PyQt5 import QtWebEngineWidgets

points = np.array([[0., 0., 0.], [0.01, 0., 0.], [0., 0.01, 0.]])
cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
shift = np.eye(4)
shift[:3, 3] = [0.1, 0.2, 0.3]
cloud.transform(shift)
assert np.allclose(np.asarray(cloud.points), points + [0.1, 0.2, 0.3])
assert cv2.cvtColor(np.zeros((2, 2, 3), np.uint8), cv2.COLOR_BGR2GRAY).shape == (2, 2)
print(json.dumps({"status": "pass", "scope": "offline dependencies and synthetic geometry only", "hardware_accessed": False,
    "versions": {p: version(p) for p in ["numpy", "open3d", "opencv-python", "PyQt5", "spectral", "pandas", "PyQtWebEngine"]}}, indent=2))
app.quit()
