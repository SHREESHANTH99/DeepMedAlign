"""
build_npy_cache.py
-------------------
Converts preprocessed NIfTI volumes to compressed NumPy .npy files.

Why this matters for Week 3 training speed
──────────────────────────────────────────
Without NPY cache:
  Training loop → load *.nii.gz → decompress → ~2s per batch
  1500 epochs × 180 subjects = 270 000 loads × 2s = 150 hours I/O

With NPY cache:
  Training loop → load *.npy → ~0.01s per batch
  1500 epochs × 180 subjects = 270 000 loads × 0.01s = 45 min I/O

The .npy cache is the difference between training finishing overnight
and training finishing next week.

Output per subject:
  data/processed/<sid>/<sid>_mr.npy     float32, shape = FIXED_SHAPE
  data/processed/<sid>/<sid>_ct.npy     float32, shape = FIXED_SHAPE
  data/processed/<sid>/<sid>_mask.npy   float32, shape = FIXED_SHAPE (binary)

Run:
    python scripts/build_npy_cache.py
    python scripts/build_npy_cache.py --split train
    python scripts/build_npy_cache.py --limit 5
    python scripts/build_npy_cache.py --force
    python scripts/build_npy_cache.py --verify
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC, FIXED_SHAPE
from src.utils import get_logger

log = get_logger("build_npy_cache")


# ---------------------------------------------------------------------------
# Core conversion function
# ---------------------------------------------------------------------------

def nii_to_npy(
    nii_path:       str,
    npy_path:       str,
    expected_shape: tuple = FIXED_SHAPE,
    force:          bool  = False,
) -> bool:
    """Load a NIfTI file and save as float32 .npy.

    If the volume does not match expected_shape it is cropped / zero-padded.

    Parameters
    ----------
    nii_path       : input NIfTI path (.nii or .nii.gz)
    npy_path       : output .npy path
    expected_shape : (D, H, W) target shape — must match FIXED_SHAPE
    force          : overwrite existing .npy file

    Returns
    -------
    True  if the file was written
    False if skipped (already exists and force=False) or input missing
    """
    if not force and Path(npy_path).exists():
        return False   # already cached

    if not Path(nii_path).exists():
        log.warning(f"NIfTI not found: {nii_path}")
        return False

    import nibabel as nib
    arr = nib.load(nii_path).get_fdata().astype("float32")

    # Crop / zero-pad to fixed spatial shape
    if arr.shape != expected_shape:
        out = np.zeros(expected_shape, dtype="float32")
        s   = tuple(min(a, b) for a, b in zip(arr.shape, expected_shape))
        out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]
        arr = out

    np.save(npy_path, arr)
    return True


# ---------------------------------------------------------------------------
# Per-subject cache builder
# ---------------------------------------------------------------------------

def build_cache_for_subject(sid: str, force: bool = False) -> dict:
    """Build .npy cache for one subject.

    Returns a status dict — never raises.
    """
    out    = DATA_PROC / sid
    result = {"subject_id": sid}

    if not out.exists():
        result["cache_status"] = "skip_no_processed_dir"
        return result

    saved  = []
    errors = []

    targets = [
        ("mr",   f"{sid}_mr_norm.nii.gz",      f"{sid}_mr.npy"),
        ("ct",   f"{sid}_ct_norm.nii.gz",       f"{sid}_ct.npy"),
        ("mask", f"{sid}_mr_brain_mask.nii.gz", f"{sid}_mask.npy"),
    ]

    for key, nii_name, npy_name in targets:
        nii_path = str(out / nii_name)
        npy_path = str(out / npy_name)
        try:
            was_saved = nii_to_npy(nii_path, npy_path, force=force)
            result[f"{key}_npy"] = npy_path
            if was_saved:
                saved.append(key)
                size_mb = Path(npy_path).stat().st_size // (1024 * 1024)
                log.info(f"  {npy_name}  ({size_mb} MB)")
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            log.error(f"  {npy_name} FAILED: {exc}")

    if errors:
        result["cache_status"] = f"partial_error: {errors}"
    elif len(saved) == 0:
        result["cache_status"] = "skip_already_cached"
    else:
        result["cache_status"] = "ok"
        result["files_saved"]  = len(saved)

    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_npy_files(sid: str) -> dict:
    """Verify .npy files are loadable and have the correct shape.

    Call after build_cache_for_subject to confirm the cache is valid.

    Returns
    -------
    dict with 'verify_ok' bool and per-volume shape / range info
    """
    out    = DATA_PROC / sid
    result = {"subject_id": sid, "verify_ok": True}

    for key, npy_name in [
        ("mr",   f"{sid}_mr.npy"),
        ("ct",   f"{sid}_ct.npy"),
        ("mask", f"{sid}_mask.npy"),
    ]:
        npy_path = str(out / npy_name)
        if not Path(npy_path).exists():
            result["verify_ok"]    = False
            result[f"{key}_issue"] = "file missing"
            continue
        try:
            arr = np.load(npy_path)
            result[f"{key}_shape"] = str(arr.shape)
            result[f"{key}_min"]   = round(float(arr.min()), 4)
            result[f"{key}_max"]   = round(float(arr.max()), 4)
            if arr.shape != FIXED_SHAPE:
                result["verify_ok"]    = False
                result[f"{key}_issue"] = f"wrong shape {arr.shape} (expected {FIXED_SHAPE})"
        except Exception as exc:
            result["verify_ok"]    = False
            result[f"{key}_issue"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert preprocessed NIfTI files to .npy cache for fast DL training."
    )
    ap.add_argument("--split",  default=None,
                    choices=["train", "val", "test"],
                    help="Cache only one split.")
    ap.add_argument("--limit",  type=int, default=None,
                    help="Cache only the first N subjects.")
    ap.add_argument("--force",  action="store_true",
                    help="Rebuild .npy files even if they already exist.")
    ap.add_argument("--verify", action="store_true",
                    help="Verify every .npy file after building.")
    args = ap.parse_args()

    # Load manifest
    manifest = None
    for name in ["manifest_processed.csv", "manifest.csv"]:
        p = DATA_RAW / name
        if p.exists():
            manifest = pd.read_csv(p)
            log.info(f"Loaded manifest: {p.name}  ({len(manifest)} subjects)")
            break

    if manifest is None:
        log.error("No manifest found. Run build_manifest.py first.")
        sys.exit(1)

    if args.split:
        manifest = manifest[manifest["split"] == args.split]
    if args.limit:
        manifest = manifest.head(args.limit)

    log.info(f"Building NPY cache for {len(manifest)} subjects …")
    t_start = time.time()

    cache_results  = []
    verify_results = []
    n_built = n_skip = n_error = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="Caching"):
        sid = row["subject_id"]
        log.info(f"Caching: {sid}")

        r = build_cache_for_subject(sid, force=args.force)
        cache_results.append(r)

        status = r.get("cache_status", "")
        if status == "ok":
            n_built += 1
        elif status.startswith("skip"):
            n_skip += 1
        else:
            n_error += 1

        if args.verify:
            verify_results.append(verify_npy_files(sid))

    elapsed = time.time() - t_start

    # Save report
    pd.DataFrame(cache_results).to_csv(
        DATA_RAW / "npy_cache_report.csv", index=False)

    print("\n" + "=" * 60)
    print("NPY CACHE BUILD COMPLETE")
    print("=" * 60)
    print(f"  Built   : {n_built}")
    print(f"  Skipped : {n_skip}  (already cached)")
    print(f"  Errors  : {n_error}")
    print(f"  Time    : {elapsed:.1f}s")

    if args.verify and verify_results:
        vdf      = pd.DataFrame(verify_results)
        ok_count = vdf["verify_ok"].sum()
        print(f"\n  Verification: {ok_count}/{len(vdf)} OK")
        bad = vdf[~vdf["verify_ok"]]
        if len(bad) > 0:
            print("  Failed:")
            for _, b in bad.iterrows():
                print(f"    {b['subject_id']}")

    print("=" * 60)
    print("\nWeek 3 PyTorch Dataset will load from these .npy files.")
    print("Loading speed: ~0.01s per volume  (vs ~2s for .nii.gz)")


if __name__ == "__main__":
    main()
