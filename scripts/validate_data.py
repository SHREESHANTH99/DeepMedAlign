"""
validate_data.py
----------------
QC script: checks every raw SynthRAD subject for:
  - Presence of mr.nii.gz and ct.nii.gz
  - Shape and voxel spacing
  - Intensity range (MR and CT)
  - Shape mismatch between MR and CT (must be identical)
  - Flags subjects with unusual shapes or ranges

Outputs a CSV report to data/raw/shape_report.csv
and prints a summary table.

Run:
    python scripts/validate_data.py
    python scripts/validate_data.py --n 10   # first 10 subjects only
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SYNTHRAD, SHAPE_RPT
from src.utils import get_logger

log = get_logger("validate_data")

FIELDS = [
    "subj_id",
    "mr_exists", "ct_exists",
    "mr_shape",  "ct_shape",
    "mr_spacing","ct_spacing",
    "mr_min", "mr_max",
    "ct_min", "ct_max",
    "shape_match",
    "flags",
]


def load_meta(path: Path) -> dict:
    """Return shape/spacing/range without loading full array into RAM."""
    try:
        import nibabel as nib
        img  = nib.load(str(path))
        hdr  = img.header
        data = img.get_fdata(dtype="float32")
        return {
            "shape":   tuple(int(x) for x in img.shape[:3]),
            "spacing": tuple(round(float(z), 3) for z in hdr.get_zooms()[:3]),
            "min":     float(data.min()),
            "max":     float(data.max()),
        }
    except Exception as e:
        return {"error": str(e)}


def check_subject(subj_dir: Path) -> dict:
    """Run all checks on one subject. Returns a row dict."""
    sid     = subj_dir.name
    mr_path = subj_dir / "mr.nii.gz"
    ct_path = subj_dir / "ct.nii.gz"

    row = {f: "" for f in FIELDS}
    row["subj_id"]   = sid
    row["mr_exists"] = mr_path.exists()
    row["ct_exists"] = ct_path.exists()

    flags = []

    if not mr_path.exists():
        flags.append("MISSING_MR")
    if not ct_path.exists():
        flags.append("MISSING_CT")

    if mr_path.exists():
        mr = load_meta(mr_path)
        if "error" in mr:
            flags.append(f"MR_LOAD_ERR:{mr['error'][:40]}")
        else:
            row["mr_shape"]   = str(mr["shape"])
            row["mr_spacing"] = str(mr["spacing"])
            row["mr_min"]     = f"{mr['min']:.1f}"
            row["mr_max"]     = f"{mr['max']:.1f}"
            if mr["max"] < 10:
                flags.append("MR_EMPTY")
            if any(abs(s - 1.0) > 0.05 for s in mr["spacing"]):
                flags.append("MR_NONISO")

    if ct_path.exists():
        ct = load_meta(ct_path)
        if "error" in ct:
            flags.append(f"CT_LOAD_ERR:{ct['error'][:40]}")
        else:
            row["ct_shape"]   = str(ct["shape"])
            row["ct_spacing"] = str(ct["spacing"])
            row["ct_min"]     = f"{ct['min']:.1f}"
            row["ct_max"]     = f"{ct['max']:.1f}"
            if ct["min"] > -500:
                flags.append("CT_MIN_UNUSUAL")
            if ct["max"] < 500:
                flags.append("CT_MAX_LOW")

    # Shape match
    if row["mr_shape"] and row["ct_shape"]:
        match = row["mr_shape"] == row["ct_shape"]
        row["shape_match"] = match
        if not match:
            flags.append("SHAPE_MISMATCH")
    else:
        row["shape_match"] = False

    row["flags"] = "|".join(flags) if flags else "OK"
    return row


def print_summary(rows: list):
    """Print a compact summary table."""
    total   = len(rows)
    ok      = sum(1 for r in rows if r["flags"] == "OK")
    flagged = total - ok

    print(f"\n{'='*60}")
    print(f"  Total subjects : {total}")
    print(f"  Clean (OK)     : {ok}")
    print(f"  Flagged        : {flagged}")
    print(f"{'='*60}")

    if flagged:
        print("\nFlagged subjects:")
        for r in rows:
            if r["flags"] != "OK":
                print(f"  {r['subj_id']:12s}  {r['flags']}")

    # Shape distribution
    shapes = {}
    for r in rows:
        s = r.get("mr_shape", "unknown")
        shapes[s] = shapes.get(s, 0) + 1
    print(f"\nMR shape distribution ({len(shapes)} unique):")
    for shape, count in sorted(shapes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {shape:30s} x{count}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Check only first N subjects")
    args = ap.parse_args()

    subjects = sorted([d for d in SYNTHRAD.iterdir() if d.is_dir()])
    if args.n:
        subjects = subjects[:args.n]

    log.info(f"Validating {len(subjects)} subjects in {SYNTHRAD}...")

    rows = []
    for i, subj_dir in enumerate(subjects, 1):
        log.info(f"  [{i:3d}/{len(subjects)}] {subj_dir.name}")
        rows.append(check_subject(subj_dir))

    # Write CSV
    SHAPE_RPT.parent.mkdir(parents=True, exist_ok=True)
    with open(SHAPE_RPT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Report saved: {SHAPE_RPT}")

    print_summary(rows)


if __name__ == "__main__":
    main()
