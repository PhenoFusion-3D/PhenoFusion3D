from pathlib import Path

import cv2
import numpy as np
import pytest
import json

from processing.reference_traits import (
    FrameCandidate,
    FramePair,
    _hull_area,
    _oriented_spans,
    _pixel_area_sum,
    plant_mask,
    select_plant_candidates,
    valid_foreground_mask,
)
from processing.trait_validation import compare_manual_to_3d, compare_reference_to_3d


def test_reference_rejects_invalid_calibration(tmp_path):
    from processing.reference_traits import load_intrinsics
    (tmp_path/'kdc_intrinsics.txt').write_text(json.dumps(dict(K=[[0,0,80],[0,120,60],[0,0,1]],width=160,height=120)))
    with pytest.raises(ValueError,match='camera matrix'):load_intrinsics(tmp_path)


def test_plant_mask_selects_green_object() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.ellipse(image, (80, 60), (35, 22), 0, 0, 360, (30, 150, 35), -1)
    mask = plant_mask(image, min_component_area=50)
    assert np.count_nonzero(mask) > 1_500
    assert mask[60, 80] == 255
    assert mask[5, 5] == 0


def test_metric_projected_area_uses_depth_and_intrinsics() -> None:
    mask = np.ones((10, 20), dtype=np.uint8) * 255
    depth = np.ones((10, 20), dtype=np.uint16) * 1_000
    matrix = np.array([[100.0, 0, 10.0], [0, 100.0, 5.0], [0, 0, 1.0]])
    assert np.isclose(_pixel_area_sum(mask, depth, matrix), 0.02)


def test_hull_area_and_oriented_spans() -> None:
    points = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float64)
    assert np.isclose(_hull_area(points), 2.0)
    major, minor = _oriented_spans(points)
    assert np.isclose(major, 2.0)
    assert np.isclose(minor, 1.0)


def test_candidate_selection_uses_separated_peaks() -> None:
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    candidates = []
    for index, score in [(10, 100.0), (12, 90.0), (80, 70.0), (150, 60.0)]:
        pair = FramePair(Path(f"rgb_{index}.png"), Path(f"depth_{index}.png"), index, index)
        candidates.append(FrameCandidate(pair, mask, score, 16, 1.0))
    selected = select_plant_candidates(candidates, expected_plants=3, min_separation_frames=30)
    assert [item.pair.frame_index for item in selected] == [10, 80, 150]


def test_foreground_depth_gating_removes_far_leakage() -> None:
    mask = np.ones((20, 20), dtype=np.uint8) * 255
    depth = np.ones((20, 20), dtype=np.uint16) * 1_500
    depth[:, 15:] = 8_000
    valid = valid_foreground_mask(mask, depth)
    assert np.count_nonzero(valid[:, :15]) == 300
    assert np.count_nonzero(valid[:, 15:]) == 0


def test_comparison_uses_explicit_mapping_and_projected_hull(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    traits_dir = tmp_path / "traits"
    reference_dir.mkdir()
    (reference_dir / "reference_traits.json").write_text(
        '{"plants":[{"plant_id":1,"canopy_major_span_m":2.0,'
        '"canopy_minor_span_m":1.0,"visible_depth_relief_m":0.5,'
        '"projected_canopy_area_m2":1.2,"projected_convex_hull_area_m2":1.5}]}',
        encoding="utf-8",
    )
    model_dir = traits_dir / "plant_4"
    model_dir.mkdir(parents=True)
    (model_dir / "traits.json").write_text(
        '{"canopy_major_span_m":2.2,"canopy_minor_span_m":1.1,'
        '"height_robust_5_95_m":0.6,"projected_canopy_area_m2":1.3,'
        '"projected_convex_hull_area_m2":1.8}',
        encoding="utf-8",
    )
    rows = compare_reference_to_3d(
        reference_dir / "reference_traits.json",
        traits_dir,
        tmp_path / "comparison",
        plant_mapping={1: 4},
    )
    hull = next(row for row in rows if row["trait"] == "projected_convex_hull_area_m2")
    assert hull["model_plant_id"] == 4
    assert np.isclose(hull["percentage_difference"], 20.0)
    assert (tmp_path / "comparison" / "validation_report.html").exists()

    manual_csv = tmp_path / "manual.csv"
    manual_csv.write_text(
        "plant_id,physical_plant_height_m,physical_canopy_major_span_m\n"
        "1,0.5,2.0\n",
        encoding="utf-8",
    )
    manual_rows = compare_manual_to_3d(
        manual_csv,
        traits_dir,
        tmp_path / "comparison",
        plant_mapping={1: 4},
    )
    major = next(row for row in manual_rows if row["trait"] == "canopy_major_span_m")
    assert np.isclose(major["percentage_error"], 10.0)
    assert (tmp_path / "comparison" / "manual_comparison.csv").exists()
