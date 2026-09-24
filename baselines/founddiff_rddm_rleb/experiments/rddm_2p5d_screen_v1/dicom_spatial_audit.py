#!/usr/bin/env python3
"""Read-only spatial metadata audit for three allowed Mayo patients."""
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pydicom
from natsort import natsorted

PATIENTS = ("L067", "L143", "L333")
DOSES = (("full_1mm", "target"), ("quarter_1mm", "25"))
DICOM_ROOT = Path("/home/xujian/data/Mayo2016")
NPY_ROOT = Path("/home/xujian/projects/CoreDiff/data_preprocess/gen_data/mayo_2016_npy")
OUT = Path(__file__).resolve().parent
TAGS = [
    "SeriesInstanceUID", "ImageOrientationPatient", "ImagePositionPatient",
    "InstanceNumber", "SliceThickness", "SpacingBetweenSlices",
]
EXPECTED = {"L067": 560, "L143": 585, "L333": 610}
EPS_ORIENT = 1e-4
EPS_PAIR_MM = 0.1
EPS_SPACING_MM = 0.05


def scalar_float(ds, name):
    value = ds.get(name)
    return None if value is None else float(value)


def read_dose(patient, dose, view):
    folder = DICOM_ROOT / patient / dose
    if not folder.is_dir():
        return {"patient": patient, "dose": dose, "view": view, "pass": False,
                "errors": ["DICOM directory missing"]}, None
    paths = natsorted(folder.glob("*.IMA"))
    errors = []
    if len(paths) != EXPECTED[patient]:
        errors.append("DICOM count differs from frozen count")
    records = []
    for path in paths:
        ds = pydicom.dcmread(
            path, stop_before_pixels=True, specific_tags=TAGS, force=False
        )
        missing = [name for name in TAGS[:4] if ds.get(name) is None]
        if missing:
            errors.append("required DICOM spatial/series/instance tag missing")
            break
        orient = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        pos = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        if orient.shape != (6,) or pos.shape != (3,) or not np.isfinite(orient).all() or not np.isfinite(pos).all():
            errors.append("invalid orientation or position")
            break
        normal = np.cross(orient[:3], orient[3:])
        length = float(np.linalg.norm(normal))
        if not math.isfinite(length) or abs(length - 1.0) > EPS_ORIENT:
            errors.append("invalid orientation normal")
            break
        records.append({
            "uid_hash": hashlib.sha256(str(ds.SeriesInstanceUID).encode()).hexdigest(),
            "orientation": orient,
            "position": pos,
            "normal": normal / length,
            "instance": int(ds.InstanceNumber),
            "thickness": scalar_float(ds, "SliceThickness"),
            "between": scalar_float(ds, "SpacingBetweenSlices"),
        })
    if len(records) != len(paths) or not records:
        return {"patient": patient, "dose": dose, "view": view, "dicom_files": len(paths),
                "metadata_records": len(records), "pass": False,
                "errors": sorted(set(errors))}, None

    ref_orientation = records[0]["orientation"]
    ref_normal = records[0]["normal"]
    orientation_deviation = max(float(np.max(np.abs(r["orientation"] - ref_orientation))) for r in records)
    normal_deviation = max(float(np.max(np.abs(r["normal"] - ref_normal))) for r in records)
    if orientation_deviation > EPS_ORIENT or normal_deviation > EPS_ORIENT:
        errors.append("orientation changes within dose")
    coords = [float(np.dot(r["position"], ref_normal)) for r in records]
    gaps = np.diff(np.asarray(coords))
    increasing = bool(np.all(gaps > 0))
    decreasing = bool(np.all(gaps < 0))
    if not (increasing or decreasing):
        errors.append("natural filename order is not strict spatial order")
    abs_gaps = np.abs(gaps)
    median_spacing = float(np.median(abs_gaps)) if len(abs_gaps) else None
    max_gap_deviation = float(np.max(np.abs(abs_gaps - median_spacing))) if len(abs_gaps) else None
    if median_spacing is None or median_spacing <= 0 or max_gap_deviation > EPS_SPACING_MM:
        errors.append("adjacent spatial spacing is not stable")
    uids = {r["uid_hash"] for r in records}
    if len(uids) != 1:
        errors.append("multiple SeriesInstanceUID values in one dose")
    instances = [r["instance"] for r in records]
    inst_steps = np.diff(np.asarray(instances))
    instance_monotonic = bool(np.all(inst_steps > 0) or np.all(inst_steps < 0))
    if not instance_monotonic:
        errors.append("InstanceNumber is not strictly monotonic in filename order")
    thick_values = [r["thickness"] for r in records]
    between_values = [r["between"] for r in records]
    def tag_summary(values):
        present = [v for v in values if v is not None]
        return {"present_count": len(present), "min": min(present) if present else None,
                "max": max(present) if present else None}
    thick = tag_summary(thick_values)
    between = tag_summary(between_values)
    if thick["present_count"] != len(records):
        errors.append("SliceThickness missing on at least one slice")
    elif thick["max"] - thick["min"] > EPS_SPACING_MM:
        errors.append("SliceThickness varies within dose")
    if between["present_count"] and (
        between["present_count"] != len(records)
        or between["max"] - between["min"] > EPS_SPACING_MM
        or abs(abs(between["min"]) - median_spacing) > EPS_SPACING_MM
    ):
        errors.append("SpacingBetweenSlices inconsistent with measured spacing")
    npy_missing = [
        index for index in range(len(paths))
        if not (NPY_ROOT / f"{patient}_{view}_{index:03d}_img.npy").is_file()
    ]
    if npy_missing:
        errors.append("expected indexed NPY path missing")
    summary = {
        "patient": patient, "dose": dose, "view": view,
        "dicom_files": len(paths), "expected_files": EXPECTED[patient],
        "series_uid_sha256": sorted(uids), "series_count": len(uids),
        "orientation_first": [float(v) for v in ref_orientation],
        "normal_first": [float(v) for v in ref_normal],
        "max_orientation_component_deviation": orientation_deviation,
        "filename_order_vs_spatial": "increasing" if increasing else "decreasing" if decreasing else "nonmonotonic",
        "instance_number_first_last": [instances[0], instances[-1]],
        "instance_number_monotonic": instance_monotonic,
        "coordinate_first_last_mm": [coords[0], coords[-1]],
        "median_adjacent_spacing_mm": median_spacing,
        "max_adjacent_spacing_deviation_mm": max_gap_deviation,
        "slice_thickness_mm": thick, "spacing_between_slices_mm": between,
        "expected_npy_indices_present": len(paths) - len(npy_missing),
        "missing_npy_indices": npy_missing[:20],
        "pass": not errors, "errors": sorted(set(errors)),
    }
    return summary, {"coords": coords, "orientation": ref_orientation, "normal": ref_normal,
                     "series": sorted(uids), "paths": paths}


def main():
    rows = []
    pair_rows = []
    for patient in PATIENTS:
        dose_data = {}
        for dose, view in DOSES:
            row, data = read_dose(patient, dose, view)
            rows.append(row)
            dose_data[dose] = data
        a, b = dose_data["full_1mm"], dose_data["quarter_1mm"]
        errors = []
        if a is None or b is None:
            errors.append("cannot compare doses: incomplete metadata")
            deltas = []
        else:
            if len(a["coords"]) != len(b["coords"]):
                errors.append("dose slice counts differ")
            if float(np.max(np.abs(a["orientation"] - b["orientation"]))) > EPS_ORIENT:
                errors.append("dose orientations differ")
            if float(np.max(np.abs(a["normal"] - b["normal"]))) > EPS_ORIENT:
                errors.append("dose normals differ")
            # Project BOTH doses onto the full-dose normal.
            full_coords = [float(np.dot(p, a["normal"])) for p in [
                pydicom.dcmread(f, stop_before_pixels=True,
                                specific_tags=["ImagePositionPatient"]).ImagePositionPatient
                for f in a["paths"]
            ]]
            quarter_coords = [float(np.dot(p, a["normal"])) for p in [
                pydicom.dcmread(f, stop_before_pixels=True,
                                specific_tags=["ImagePositionPatient"]).ImagePositionPatient
                for f in b["paths"]
            ]]
            deltas = [abs(x - y) for x, y in zip(full_coords, quarter_coords)]
            if len(deltas) != len(full_coords) or (deltas and max(deltas) > EPS_PAIR_MM):
                errors.append("full/quarter index positions do not match within tolerance")
        pair_rows.append({
            "patient": patient, "full_and_quarter_index_pairing_pass": not errors,
            "matched_count": len(deltas), "max_position_delta_mm": max(deltas) if deltas else None,
            "median_position_delta_mm": statistics.median(deltas) if deltas else None,
            "errors": errors,
        })
    all_pass = all(r["pass"] for r in rows) and all(r["full_and_quarter_index_pairing_pass"] for r in pair_rows)
    out = {
        "result": "PASS" if all_pass else "FAIL",
        "scope": {"patients": list(PATIENTS), "doses": [d for d, _ in DOSES],
                  "pixels_read": False, "npy_pixels_read": False,
                  "test_patient_files_accessed": 0, "external_memory_files_accessed": 0},
        "method": "pydicom.dcmread(stop_before_pixels=True,specific_tags=...) on explicitly scoped patient/dose directories; natsort identical to prep_mayo2016.py; normal=cross(IOP row,IOP column); coordinate=dot(IPP,normal)",
        "thresholds": {"orientation_component": EPS_ORIENT, "pair_position_mm": EPS_PAIR_MM,
                       "spacing_deviation_mm": EPS_SPACING_MM},
        "npy_index_basis": "prep_mayo2016.py independently natsorts each dose's *.IMA and enumerate() assigns patient_view_index_img.npy; audit verifies all expected exact NPY paths exist without opening their pixels. Binary NPY provenance cannot be independently proven without comparing pixel content, which this task prohibits.",
        "doses": rows, "dose_pairs": pair_rows,
    }
    (OUT / "DICOM_SPATIAL_AUDIT.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = [
        "# Mayo DICOM spatial metadata audit", "",
        f"Result: **{out['result']}**. Pixel data were not read or modified.",
        "",
        "| Patient | Dose | Slices | Series | Filename order | Spacing (mm) | Max gap deviation (mm) | NPY paths | Status |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['patient']} | {r['dose']} | {r.get('dicom_files', 0)} | {r.get('series_count', 0)} | "
            f"{r.get('filename_order_vs_spatial', 'unverified')} | "
            f"{r.get('median_adjacent_spacing_mm', float('nan')):.4f} | "
            f"{r.get('max_adjacent_spacing_deviation_mm', float('nan')):.4f} | "
            f"{r.get('expected_npy_indices_present', 0)} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += ["", "| Patient | Matched positions | Max full/quarter difference (mm) | Pairing |",
              "| --- | ---: | ---: | --- |"]
    for r in pair_rows:
        lines.append(f"| {r['patient']} | {r['matched_count']} | "
                     f"{r['max_position_delta_mm'] if r['max_position_delta_mm'] is not None else 'N/A'} | "
                     f"{'PASS' if r['full_and_quarter_index_pairing_pass'] else 'FAIL'} |")
    failures = [(r["patient"], r["dose"], r["errors"]) for r in rows if r["errors"]]
    failures += [(r["patient"], "cross-dose", r["errors"]) for r in pair_rows if r["errors"]]
    if failures:
        lines += ["", "Failures:"]
        for patient, dose, errors in failures:
            lines.append(f"- {patient} {dose}: {', '.join(errors)}")
    lines += ["", "NPY index correspondence is established by the deterministic preprocessing source and existence of the expected indexed files; NPY binary provenance cannot be checked without pixel comparison.", ""]
    (OUT / "DICOM_SPATIAL_AUDIT_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"result": out["result"], "doses": [(r["patient"], r["dose"], r["pass"]) for r in rows],
                      "pairs": [(r["patient"], r["full_and_quarter_index_pairing_pass"]) for r in pair_rows]}))


if __name__ == "__main__":
    main()
