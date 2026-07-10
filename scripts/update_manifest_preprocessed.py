"""
update_manifest_preprocessed.py
---------------------------------
Scans data/processed/ for all preprocessed outputs and updates
the manifest CSV with exact file paths for every subject.

This is what feeds Week 3's PyTorch Dataset.
Every column in the output manifest is consumed by src/dataset.py.

Run:
    python scripts/update_manifest_preprocessed.py
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC
from src.utils import get_logger

log = get_logger("update_manifest_preprocessed")


def find_preprocessed_files(sid: str) -> dict:
    """Scan the processed directory for one subject.

    Returns a dict with file paths (empty string if file not found).
    Does NOT raise — always returns something useful.
    """
    out = DATA_PROC / sid
    row: dict = {}

    # ── Preprocessed volumes ────────────────────────────────────────────────
    for key, fname in [
        ("mr_preprocessed",  f"{sid}_mr_norm.nii.gz"),
        ("mr_mask",          f"{sid}_mr_brain_mask.nii.gz"),
        ("ct_preprocessed",  f"{sid}_ct_norm.nii.gz"),
        ("ct_mask",          f"{sid}_ct_mask.nii.gz"),
    ]:
        p = out / fname
        row[key] = str(p) if p.exists() else ""

    # ── Classical registration outputs ──────────────────────────────────────
    for key, fname in [
        ("ct_rigid",    f"{sid}_ct_rigid.nii.gz"),
        ("ct_affine",   f"{sid}_ct_affine.nii.gz"),
        ("ct_bspline",  f"{sid}_ct_bspline.nii.gz"),
        ("rigid_tx",    f"{sid}_rigid.tfm"),
        ("affine_tx",   f"{sid}_affine.tfm"),
        ("bspline_tx",  f"{sid}_bspline.tfm"),
    ]:
        p = out / fname
        row[key] = str(p) if p.exists() else ""

    # ── NPY cache files (created by build_npy_cache.py, Day 8) ──────────────
    for key, fname in [
        ("mr_npy",   f"{sid}_mr.npy"),
        ("ct_npy",   f"{sid}_ct.npy"),
        ("mask_npy", f"{sid}_mask.npy"),
    ]:
        p = out / fname
        row[key] = str(p) if p.exists() else ""

    # ── Overall status ───────────────────────────────────────────────────────
    mr_ok  = bool(row["mr_preprocessed"])
    ct_ok  = bool(row["ct_preprocessed"])
    aff_ok = bool(row["ct_affine"])

    if mr_ok and ct_ok and aff_ok:
        row["preprocess_status"] = "complete"
    elif mr_ok and ct_ok:
        row["preprocess_status"] = "preprocessed_no_reg"
    elif mr_ok or ct_ok:
        row["preprocess_status"] = "partial"
    else:
        row["preprocess_status"] = "missing"

    return row


def main() -> None:
    # Load the best available base manifest
    base = None
    for name in ["manifest_final.csv", "manifest_v2.csv", "manifest.csv"]:
        p = DATA_RAW / name
        if p.exists():
            base = pd.read_csv(p)
            log.info(f"Loaded base manifest: {p.name}  ({len(base)} subjects)")
            break

    if base is None:
        log.error("No manifest found. Run build_manifest.py first.")
        raise FileNotFoundError("manifest not found in data/raw/")

    log.info(f"Scanning processed directory: {DATA_PROC}")

    rows = []
    for _, row in tqdm(base.iterrows(), total=len(base), desc="Scanning"):
        sid   = row["subject_id"]
        found = find_preprocessed_files(sid)
        rows.append({**row.to_dict(), **found})

    df = pd.DataFrame(rows)

    out_path = DATA_RAW / "manifest_processed.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    status_counts = df["preprocess_status"].value_counts()
    print("\n" + "=" * 55)
    print("MANIFEST UPDATE SUMMARY")
    print("=" * 55)
    print(f"Total subjects : {len(df)}")
    print("\nPreprocessing status:")
    for status, count in status_counts.items():
        print(f"  {status:35s}: {count}")

    complete = df[df["preprocess_status"] == "complete"]
    print(f"\nReady for Week 3 training : {len(complete)}/{len(df)}")

    if len(complete) < len(df):
        missing = df[df["preprocess_status"] == "missing"]
        if len(missing) > 0:
            print(f"\nMissing subjects (ask R3 to run preprocessing):")
            for sid in missing["subject_id"].tolist()[:5]:
                print(f"  {sid}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
    print("=" * 55)


if __name__ == "__main__":
    main()
