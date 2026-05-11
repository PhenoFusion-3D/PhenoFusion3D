from pathlib import Path

import numpy as np
import open3d as o3d

from processing.canopy import (
    CanopyReconstructionConfig,
    _fill_depth_inside_mask,
    _frame_positions_m,
    _nearest_fill_in_mask,
    _select_frames,
)
from visualiser.viewer import write_canopy_mesh_viewer


def test_select_frames_prefers_high_quality_reference_and_nearby_positions():
    candidates = [
        {"token": i, "position_m": float(i), "score": 1.0}
        for i in range(5)
    ]
    candidates[2]["score"] = 10.0

    selected, reference = _select_frames(
        candidates,
        CanopyReconstructionConfig(max_frames=3, max_candidates=0),
    )

    assert reference == 2
    assert [item["token"] for item in selected] == [1, 2, 3]


def test_frame_positions_use_flat_filename_position_tokens(tmp_path):
    pairs = [
        (100000, tmp_path / "rgb_100000.png", tmp_path / "depth_100000.png"),
        (103000, tmp_path / "rgb_103000.png", tmp_path / "depth_103000.png"),
        (106000, tmp_path / "rgb_106000.png", tmp_path / "depth_106000.png"),
    ]

    positions, info = _frame_positions_m(tmp_path, pairs)

    assert info["source"] == "filename_position_token"
    assert abs(positions[100000] - 0.1) < 1e-12
    assert abs(positions[106000] - 0.106) < 1e-12
    assert abs(info["median_step_m"] - 0.003) < 1e-12
    assert "best below" in info["warning"]


def test_foreground_depth_gating_rejects_far_leakage_inside_leaf_mask():
    depth = np.full((5, 5), 1000, dtype=np.uint16)
    depth[0, 0] = 3000
    mask = np.full((5, 5), 255, dtype=np.uint8)
    cfg = CanopyReconstructionConfig(
        min_valid_depth_points=1,
        foreground_percentile=90.0,
        foreground_margin_mm=80.0,
    )

    foreground, stats = _fill_depth_inside_mask(depth, mask, cfg)

    assert foreground[0, 0] == 0
    assert foreground[2, 2] == 1000
    assert stats["valid_depth_points"] == 24


def test_nearest_fill_only_fills_inside_mask():
    values = np.zeros((3, 3), dtype=np.float32)
    values[1, 1] = 1234.0
    valid = values > 0
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False

    filled = _nearest_fill_in_mask(values, valid, mask)

    assert filled[1, 1] == 1234.0
    assert filled[0, 1] == 1234.0
    assert filled[0, 0] == 0.0


def test_canopy_mesh_viewer_exports_mesh_first_html(tmp_path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.6, 0.1], [0.0, 0.7, 0.1], [0.0, 0.5, 0.1]])
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors

    out = write_canopy_mesh_viewer(
        mesh,
        tmp_path / "viewer.html",
        point_cloud=pcd,
        metadata={"Model": "display mesh"},
    )

    html = Path(out).read_text(encoding="utf-8")
    assert "Point overlay" in html
    assert '"triangleCount":1' in html
    assert "display mesh" in html
