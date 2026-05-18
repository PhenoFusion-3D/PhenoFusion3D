from pathlib import Path

import numpy as np
import open3d as o3d

from processing.canopy import (
    CanopyReconstructionConfig,
    _build_canopy_sheet_display_mesh,
    _fill_depth_inside_mask,
    _frame_positions_m,
    _nearest_fill_in_mask,
    _nearest_fill_rgb_in_mask,
    _select_frames,
)
from visualiser.viewer import write_canopy_mesh_viewer
from reconstruct_canopy_sequence import (
    _non_overlapping_y_offsets,
    _track_identity_keys,
    _track_overlap_fraction,
)


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


def test_spread_reference_frames_samples_across_track():
    candidates = [
        {
            "token": i,
            "position_m": float(i),
            "score": 10.0 if i == 5 else 1.0,
        }
        for i in range(10)
    ]

    selected, reference = _select_frames(
        candidates,
        CanopyReconstructionConfig(
            max_frames=4,
            max_candidates=0,
            reference_token=5,
            spread_reference_frames=True,
        ),
    )

    assert reference == 5
    tokens = [item["token"] for item in selected]
    assert 0 in tokens
    assert 5 in tokens
    assert 9 in tokens


def test_track_overlap_uses_component_identity_not_just_frame_token():
    track_a = {
        10: {"label": 1},
        11: {"label": 1},
    }
    track_b = {
        10: {"label": 2},
        11: {"label": 2},
    }

    assert _track_overlap_fraction(
        _track_identity_keys(track_a),
        _track_identity_keys(track_b),
    ) == 0.0


def test_sequence_layout_offsets_separate_overlapping_bounds():
    bounds = [
        (-0.2, 0.2, -0.2, 0.2),
        (-0.2, 0.2, -0.2, 0.2),
    ]
    targets = [(0.0, 0.0), (0.0, 0.05)]

    offsets = _non_overlapping_y_offsets(bounds, targets, margin_m=0.1)

    first_max_y = bounds[0][3] + offsets[0][1]
    second_min_y = bounds[1][2] + offsets[1][1]
    assert second_min_y >= first_max_y + 0.1 - 1e-12


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


def test_nearest_rgb_fill_repairs_only_inside_mask():
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[1, 1] = [10, 80, 20]
    valid = np.zeros((3, 3), dtype=bool)
    valid[1, 1] = True
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False

    filled = _nearest_fill_rgb_in_mask(rgb, valid, mask)

    assert filled[0, 1].tolist() == [10, 80, 20]
    assert filled[0, 0].tolist() == [0, 0, 0]


def test_canopy_sheet_display_mesh_bridges_smoothed_relief():
    mask = np.ones((4, 4), dtype=bool)
    depth = np.full((4, 4), 1800, dtype=np.float32)
    depth[1:3, 1:3] = 1500
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :] = [20, 120, 30]
    K = np.array([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]])

    mesh = _build_canopy_sheet_display_mesh(
        depth,
        mask,
        rgb,
        K,
        relief_m=0.02,
        smooth_sigma=1.0,
        thickness_m=0.0,
        pixel_step=1,
    )

    assert len(mesh.vertices) == 16
    assert len(mesh.triangles) == 18
    assert np.asarray(mesh.vertices)[:, 2].max() <= 0.0200001


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
