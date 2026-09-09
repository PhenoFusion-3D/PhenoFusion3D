from __future__ import annotations

import csv
import html
import json
import os
import re
import math
from pathlib import Path
from typing import Any


def _read_reference(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("plants", []))


def _read_3d_traits(path: Path) -> dict[int, dict[str, Any]]:
    files = sorted(path.glob("plant_*/traits.json"))
    result: dict[int, dict[str, Any]] = {}
    for file in files:
        match = re.search(r"plant_(\d+)", file.parent.name)
        if match:
            result[int(match.group(1))] = json.loads(file.read_text(encoding="utf-8"))
    return result


def _difference(reference: float | None, measured: float | None) -> tuple[float | None, float | None]:
    if reference is None or measured is None:
        return None, None
    absolute = float(measured) - float(reference)
    percentage = absolute / float(reference) * 100.0 if float(reference) != 0 else None
    return absolute, percentage


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Measurements must be finite non-negative numbers; leave unknown values blank.')
    return result


def compare_manual_to_3d(
    manual_csv: str | Path,
    traits_dir: str | Path,
    output_dir: str | Path,
    *,
    plant_mapping: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    with Path(manual_csv).open(newline="", encoding="utf-8-sig") as handle:
        manual_rows = list(csv.DictReader(handle))
    model_traits = _read_3d_traits(Path(traits_dir))
    trait_pairs = [
        ("plant_height_m", "physical_plant_height_m", "height_top_1_pct_m"),
        ("canopy_major_span_m", "physical_canopy_major_span_m", "canopy_major_span_m"),
        ("canopy_minor_span_m", "physical_canopy_minor_span_m", "canopy_minor_span_m"),
        ("projected_canopy_area_m2", "physical_projected_canopy_area_m2", "projected_canopy_area_m2"),
        (
            "projected_convex_hull_area_m2",
            "physical_projected_convex_hull_area_m2",
            "projected_convex_hull_area_m2",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for manual in manual_rows:
        reference_id = int(manual["plant_id"])
        model_id = (plant_mapping or {}).get(reference_id, reference_id)
        model = model_traits.get(model_id, {})
        for trait, manual_field, model_field in trait_pairs:
            manual_value = _optional_float(manual.get(manual_field))
            model_value = _optional_float(model.get(model_field))
            if manual_value is None:
                continue
            absolute, percentage = _difference(manual_value, model_value)
            rows.append(
                {
                    "physical_plant_id": reference_id,
                    "model_plant_id": model_id,
                    "trait": trait,
                    "physical_reference": manual_value,
                    "model_3d_value": model_value,
                    "signed_error": absolute,
                    "percentage_error": percentage,
                    "note": (
                        "3D height uses the top 1 percent mean to reduce isolated-point sensitivity."
                        if trait == "plant_height_m"
                        else "Matched physical and 3D projected trait."
                    ),
                }
            )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manual_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if rows:
        with (destination / "manual_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _write_report(
    rows: list[dict[str, Any]], reference_json: Path, destination: Path
) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        percentage = row.get("percentage_difference")
        if percentage is not None:
            grouped.setdefault(str(row["trait"]), []).append(abs(float(percentage)))
    summary = {
        trait: {
            "sample_count": len(values),
            "mean_absolute_percentage_difference": sum(values) / len(values),
            "max_absolute_percentage_difference": max(values),
        }
        for trait, values in grouped.items()
    }
    (destination / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    preview = reference_json.parent / "selected_frames.jpg"
    preview_href = os.path.relpath(preview, destination).replace("\\", "/")
    table_rows = []
    for row in rows:
        percentage = row.get("percentage_difference")
        percentage_text = "n/a" if percentage is None else f"{float(percentage):+.2f}%"
        css_class = "good" if percentage is not None and abs(float(percentage)) <= 10 else "warn"
        reference_value = row.get("raw_rgbd_reference")
        model_value = row.get("model_3d_value")
        reference_text = "n/a" if reference_value is None else f"{float(reference_value):.6f}"
        model_text = "n/a" if model_value is None else f"{float(model_value):.6f}"
        table_rows.append(
            "<tr>"
            f"<td>{int(row['reference_plant_id'])}</td>"
            f"<td>{int(row['model_plant_id'])}</td>"
            f"<td>{html.escape(str(row['trait']))}</td>"
            f"<td>{reference_text}</td>"
            f"<td>{model_text}</td>"
            f"<td class='{css_class}'>{percentage_text}</td>"
            "</tr>"
        )
    summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(trait)}</td>"
        f"<td>{values['mean_absolute_percentage_difference']:.2f}%</td>"
        f"<td>{values['max_absolute_percentage_difference']:.2f}%</td>"
        "</tr>"
        for trait, values in summary.items()
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhenoFusion3D Trait Validation</title>
<style>
body{{font:15px system-ui,sans-serif;margin:0;background:#f4f6f5;color:#17201b}}main{{max-width:1100px;margin:auto;padding:28px}}
h1,h2{{letter-spacing:0}}.notice{{border-left:4px solid #b26a00;background:#fff7e6;padding:12px 16px}}
img{{max-width:100%;border:1px solid #ccd4cf;background:#fff}}table{{width:100%;border-collapse:collapse;background:#fff;margin:14px 0 28px}}
th,td{{padding:9px 10px;border:1px solid #d7ddd9;text-align:right}}th:nth-child(3),td:nth-child(3){{text-align:left}}
.good{{color:#176b3a;font-weight:700}}.warn{{color:#9a3e16;font-weight:700}}code{{background:#e9eeeb;padding:2px 4px}}
</style></head><body><main><h1>Trait validation</h1>
<p class="notice"><strong>Reference type:</strong> image-derived RGB-D, not physical manual ground truth. Visible depth relief is diagnostic and is not equivalent to ruler-measured plant height.</p>
<h2>Selected specimens</h2><img src="{html.escape(preview_href)}" alt="Selected reference frames and masks">
<h2>Error summary</h2><table><thead><tr><th>Trait</th><th>Mean absolute difference</th><th>Maximum absolute difference</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Detailed comparison</h2><table><thead><tr><th>Reference plant</th><th>3D plant</th><th>Trait</th><th>Raw RGB-D</th><th>3D model</th><th>Difference</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</main></body></html>"""
    (destination / "validation_report.html").write_text(report, encoding="utf-8")


def compare_reference_to_3d(
    reference_json: str | Path,
    traits_dir: str | Path,
    output_dir: str | Path,
    *,
    plant_mapping: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    reference = _read_reference(Path(reference_json))
    model_traits = _read_3d_traits(Path(traits_dir))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, raw in enumerate(reference):
        reference_plant_id = int(raw.get("plant_id", index + 1))
        model_plant_id = (plant_mapping or {}).get(reference_plant_id, reference_plant_id)
        model = model_traits.get(model_plant_id, {})
        comparisons = [
            (
                "canopy_major_span_m",
                raw.get("canopy_major_span_m"),
                model.get("canopy_major_span_m", max(model.get("bbox_width_m", 0.0), model.get("bbox_depth_m", 0.0))) if model else None,
            ),
            (
                "canopy_minor_span_m",
                raw.get("canopy_minor_span_m"),
                model.get("canopy_minor_span_m", min(model.get("bbox_width_m", 0.0), model.get("bbox_depth_m", 0.0))) if model else None,
            ),
            (
                "visible_depth_relief_m",
                raw.get("visible_depth_relief_m"),
                model.get("height_robust_5_95_m", model.get("bbox_height_m")) if model else None,
            ),
            (
                "projected_canopy_area_m2",
                raw.get("projected_canopy_area_m2"),
                model.get("projected_canopy_area_m2") if model else None,
            ),
            (
                "projected_convex_hull_area_m2",
                raw.get("projected_convex_hull_area_m2"),
                model.get("projected_convex_hull_area_m2") if model else None,
            ),
        ]
        for trait, reference_value, model_value in comparisons:
            absolute, percentage = _difference(reference_value, model_value)
            rows.append(
                {
                    "reference_plant_id": reference_plant_id,
                    "model_plant_id": model_plant_id,
                    "trait": trait,
                    "raw_rgbd_reference": reference_value,
                    "model_3d_value": model_value,
                    "signed_difference": absolute,
                    "percentage_difference": percentage,
                    "directly_comparable": True,
                    "note": (
                        "Explicit specimen mapping supplied."
                        if plant_mapping
                        else "Same-number mapping; confirm specimen identity before reporting."
                    ),
                }
            )

    json_path = destination / "comparison.json"
    csv_path = destination / "comparison.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    _write_report(rows, Path(reference_json), destination)
    return rows
