"""
generate_data_report.py
------------------------
Generates a full pipeline status report showing how many subjects
have each processing stage completed, with a per-split breakdown
and a Week 3 readiness verdict.

Run:
    python scripts/generate_data_report.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC
from src.utils import get_logger

log = get_logger("data_report")


def check_subject_status(sid: str) -> dict:
    """Check all pipeline file statuses for one subject.

    Returns a flat dict of bool flags — never raises.
    """
    out = DATA_PROC / sid

    def exists(fname: str) -> bool:
        return (out / fname).exists()

    return {
        "subject_id":     sid,
        # Preprocessing
        "mr_preprocessed": exists(f"{sid}_mr_norm.nii.gz"),
        "ct_preprocessed": exists(f"{sid}_ct_norm.nii.gz"),
        "mr_mask":         exists(f"{sid}_mr_brain_mask.nii.gz"),
        "ct_mask":         exists(f"{sid}_ct_mask.nii.gz"),
        # Registration
        "ct_rigid":        exists(f"{sid}_ct_rigid.nii.gz"),
        "ct_affine":       exists(f"{sid}_ct_affine.nii.gz"),
        "ct_bspline":      exists(f"{sid}_ct_bspline.nii.gz"),
        # NPY cache
        "mr_npy":          exists(f"{sid}_mr.npy"),
        "ct_npy":          exists(f"{sid}_ct.npy"),
        "mask_npy":        exists(f"{sid}_mask.npy"),
    }


def _progress_bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    # Load manifest
    manifest = None
    for name in ["manifest_processed.csv", "manifest.csv"]:
        p = DATA_RAW / name
        if p.exists():
            manifest = pd.read_csv(p)
            log.info(f"Loaded: {p.name}  ({len(manifest)} subjects)")
            break

    if manifest is None:
        log.error("No manifest found. Run build_manifest.py first.")
        sys.exit(1)

    log.info(f"Checking {len(manifest)} subjects …")
    rows = [
        check_subject_status(row["subject_id"])
        for _, row in tqdm(manifest.iterrows(), total=len(manifest))
    ]

    df = pd.DataFrame(rows)
    # Merge split info
    if "split" in manifest.columns:
        df = df.merge(manifest[["subject_id", "split"]],
                      on="subject_id", how="left")

    # Save full report
    out_path = DATA_RAW / "data_status_report.csv"
    df.to_csv(out_path, index=False)

    # ── Print summary ────────────────────────────────────────────────────────
    N = len(df)
    print("\n" + "=" * 65)
    print("DATA STATUS REPORT")
    print("=" * 65)

    stages = [
        ("1. Preprocessing complete",  ["mr_preprocessed", "ct_preprocessed", "mr_mask"]),
        ("2. Registration (affine)",   ["ct_affine"]),
        ("3. Registration (bspline)",  ["ct_bspline"]),
        ("4. NPY cache ready",         ["mr_npy", "ct_npy", "mask_npy"]),
    ]

    for stage_name, cols in stages:
        present_cols = [c for c in cols if c in df.columns]
        mask = df[present_cols].all(axis=1)
        n    = int(mask.sum())
        pct  = 100 * n / N if N > 0 else 0
        print(f"\n  {stage_name}")
        print(f"  [{_progress_bar(pct)}] {n}/{N}  ({pct:.0f}%)")
        if n < N:
            missing_sids = df.loc[~mask, "subject_id"].tolist()
            preview = ", ".join(missing_sids[:3])
            suffix  = f" … +{len(missing_sids) - 3} more" if len(missing_sids) > 3 else ""
            print(f"  Missing : {preview}{suffix}")

    # ── Per-split breakdown ──────────────────────────────────────────────────
    if "split" in df.columns:
        print(f"\n  Per-split NPY readiness:")
        for split in ["train", "val", "test"]:
            sdf   = df[df["split"] == split]
            if sdf.empty:
                continue
            ready = sdf[["mr_npy", "ct_npy"]].all(axis=1).sum()
            print(f"    {split:6s}: {ready}/{len(sdf)} NPY ready")

    print(f"\n  Full report saved: {out_path}")
    print("=" * 65)

    # ── Week 3 readiness verdict ─────────────────────────────────────────────
    req_cols     = [c for c in ["mr_npy", "ct_npy", "mask_npy", "ct_affine"]
                    if c in df.columns]
    npy_ready    = df[req_cols].all(axis=1).sum() if req_cols else 0
    train_count  = len(df[df.get("split", pd.Series()) == "train"]) if "split" in df.columns else N

    print(f"\nWeek 3 training readiness: {npy_ready}/{N} subjects")

    if npy_ready < 10:
        print("⚠️  WARNING: Need at least 10 subjects for training.")
        print("   Ask your teammate to run:")
        print("     python scripts/run_preprocessing_batch.py --no-hdbet")
        print("     python scripts/run_classical.py --no-bspline")
        print("     python scripts/build_npy_cache.py")
    else:
        print("✅ READY for Week 3 VoxelMorph training!")
        print(f"   {npy_ready} subjects available  "
              f"(train: {train_count})")


if __name__ == "__main__":
    main()
