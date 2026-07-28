"""
run_preprocessing_batch.py
---------------------------
Extended batch preprocessing runner with:
  - Resume capability (skip already-processed subjects)
  - Per-subject error handling (one failure does NOT stop the batch)
  - Progress saved to CSV after every subject
  - Detailed logging to logs/r1/preprocessing_run.log

This is the script your teammate runs on the machine with data.

Run:
    python scripts/run_preprocessing_batch.py
    python scripts/run_preprocessing_batch.py --limit 5
    python scripts/run_preprocessing_batch.py --split train
    python scripts/run_preprocessing_batch.py --resume
    python scripts/run_preprocessing_batch.py --force
    python scripts/run_preprocessing_batch.py --no-hdbet
"""

import sys
import time
import argparse
import logging
import traceback
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC
from src.utils import get_logger, ensure_dir

log = get_logger("run_preprocessing_batch")


# ---------------------------------------------------------------------------
# File logger for long batch runs
# ---------------------------------------------------------------------------

def _setup_file_logger() -> None:
    ensure_dir(Path("logs/r1"))
    fh = logging.FileHandler("logs/r1/preprocessing_run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)


# ---------------------------------------------------------------------------
# Per-subject helpers
# ---------------------------------------------------------------------------

def is_already_preprocessed(sid: str) -> bool:
    """Return True if all three primary preprocessed files exist."""
    out = DATA_PROC / sid
    return (
        (out / f"{sid}_mr_norm.nii.gz").exists()
        and (out / f"{sid}_ct_norm.nii.gz").exists()
        and (out / f"{sid}_mr_brain_mask.nii.gz").exists()
    )


def process_one_subject(row: pd.Series, use_hdbet: bool = False) -> dict:
    """Run full MRI + CT preprocessing for one subject.

    Catches all exceptions so a single failure does not abort the batch.

    Parameters
    ----------
    row      : one row from the manifest DataFrame
    use_hdbet: whether to use HD-BET for skull stripping (requires GPU)

    Returns
    -------
    dict with at minimum keys: subject_id, preprocess_status
    """
    sid    = row["subject_id"]
    result = {"subject_id": sid}
    t0     = time.time()

    try:
        import os
        # Verify inputs
        for col in ["mr", "ct"]:
            p = row.get(col, "")
            if not p or not Path(p).exists():
                raise FileNotFoundError(f"{col} file not found: {p}")
            if os.path.getsize(p) < 1000:
                raise ValueError(f"{col} file is an empty placeholder: {p}")

        out_dir = str(DATA_PROC / sid)

        # Lazy import to avoid crashing on machines that lack nibabel at import time
        from src.preprocess_mri import preprocess_mri_full
        from src.preprocess_ct  import preprocess_ct_full

        log.info(f"{sid}: MRI pipeline …")
        mr_result = preprocess_mri_full(row["mr"], out_dir, sid,
                                        use_hdbet=use_hdbet)

        log.info(f"{sid}: CT pipeline …")
        ct_result = preprocess_ct_full(row["ct"], out_dir, sid)

        elapsed = time.time() - t0
        result.update({
            "mr_preprocessed":   mr_result.get("mr_norm", ""),
            "mr_mask":           mr_result.get("mr_mask", ""),
            "ct_preprocessed":   ct_result.get("ct_norm", ""),
            "ct_mask":           ct_result.get("ct_mask", ""),
            "preprocess_status": "ok",
            "preprocess_time_s": round(elapsed, 1),
        })
        log.info(f"{sid}: done in {elapsed:.1f}s")

    except FileNotFoundError as exc:
        result["preprocess_status"] = "skip_file_missing"
        result["preprocess_error"]  = str(exc)
        log.warning(f"{sid}: skipped — {exc}")

    except ValueError as exc:
        result["preprocess_status"] = "skip_dummy"
        result["preprocess_error"]  = str(exc)
        log.warning(f"{sid}: skipped (dummy file) — {exc}")

    except Exception as exc:
        result["preprocess_status"] = "error"
        result["preprocess_error"]  = str(exc)
        result["traceback"]         = traceback.format_exc()[-500:]
        log.error(f"{sid}: FAILED — {exc}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch MRI+CT preprocessing runner with resume support."
    )
    ap.add_argument("--split",    default=None,
                    choices=["train", "val", "test"],
                    help="Process only one split.")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Process only the first N subjects.")
    ap.add_argument("--resume",   action="store_true",
                    help="Skip subjects that are already preprocessed.")
    ap.add_argument("--force",    action="store_true",
                    help="Re-preprocess even if outputs already exist.")
    ap.add_argument("--hdbet",    action="store_true",
                    help="Use HD-BET skull stripper (needs GPU).")
    ap.add_argument("--no-hdbet", action="store_true",
                    help="Use fallback skull stripper (CPU, default).")
    args = ap.parse_args()

    _setup_file_logger()
    use_hdbet = args.hdbet and not args.no_hdbet

    # Load manifest
    manifest = None
    for name in ["manifest_final.csv", "manifest_v2.csv", "manifest.csv"]:
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
        log.info(f"Filtered to split={args.split}: {len(manifest)} subjects")
    if args.limit:
        manifest = manifest.head(args.limit)
        log.info(f"Limited to first {args.limit} subjects")

    ensure_dir(DATA_PROC)

    results = []
    n_ok = n_skip = n_error = 0
    t_start = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="Preprocessing"):
        sid = row["subject_id"]

        # Resume mode
        if args.resume and not args.force and is_already_preprocessed(sid):
            log.info(f"{sid}: already done — skipping.")
            n_skip += 1
            results.append({"subject_id": sid,
                             "preprocess_status": "skip_already_done"})
            continue

        r = process_one_subject(row, use_hdbet=use_hdbet)
        results.append(r)

        status = r.get("preprocess_status", "")
        if status == "ok":
            n_ok += 1
        elif status.startswith("skip"):
            n_skip += 1
        else:
            n_error += 1

        # Save progress after every subject so a crash doesn't lose work
        pd.DataFrame(results).to_csv(
            DATA_RAW / "preprocessing_progress.csv", index=False)

    elapsed = time.time() - t_start

    # ── Final merged manifest ────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    base_cols = ["subject_id", "mr", "ct", "mask", "has_mask", "split"]
    keep_cols = [c for c in base_cols if c in manifest.columns]
    merged    = manifest[keep_cols].merge(result_df, on="subject_id",
                                          how="left")
    merged.to_csv(DATA_RAW / "manifest_processed.csv", index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING BATCH COMPLETE")
    print("=" * 60)
    print(f"  OK      : {n_ok}")
    print(f"  Skipped : {n_skip}")
    print(f"  Errors  : {n_error}")
    print(f"  Time    : {elapsed:.1f}s")
    if n_ok > 0:
        print(f"  Avg/subj: {elapsed / n_ok:.1f}s")
    print(f"\n  manifest_processed.csv saved")
    print(f"  Full log: logs/r1/preprocessing_run.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
