"""
run_preprocess.py
-----------------
Batch preprocessing runner for all SynthRAD2023 subjects.
Runs MRI pipeline then CT pipeline for each subject.

Run:
    python scripts/run_preprocess.py                  # all subjects
    python scripts/run_preprocess.py --subj 1BA001    # single subject
    python scripts/run_preprocess.py --n 5            # first N subjects
    python scripts/run_preprocess.py --no-hdbet       # skip HD-BET (use fallback)
    python scripts/run_preprocess.py --dry-run        # list subjects only

Outputs go to: data/processed/<subj_id>/
Skips subjects that already have mr_norm.nii.gz (resume-safe).
"""

import sys
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SYNTHRAD, DATA_PROC
from src.preprocess_mri import preprocess_mri_full
from src.preprocess_ct import preprocess_ct_full
from src.utils import get_logger, ensure_dir

log = get_logger("run_preprocess")


def get_subjects(subj_arg=None, n=None):
    """Return list of subject directories."""
    if subj_arg:
        d = SYNTHRAD / subj_arg
        if not d.is_dir():
            log.error(f"Subject not found: {d}")
            sys.exit(1)
        return [d]

    candidates = sorted([d for d in SYNTHRAD.iterdir() if d.is_dir()])
    if n:
        candidates = candidates[:n]
    return candidates


def already_done(out_dir: Path, subj_id: str) -> bool:
    """Return True if subject has already been fully processed."""
    return (out_dir / f"{subj_id}_mr_norm.nii.gz").exists() and \
           (out_dir / f"{subj_id}_ct_norm.nii.gz").exists()


def process_subject(subj_dir: Path, use_hdbet: bool) -> dict:
    """Run MRI then CT pipeline for one subject. Returns status dict."""
    subj_id = subj_dir.name
    out_dir = DATA_PROC / subj_id
    ensure_dir(out_dir)

    mr_path = subj_dir / "mr.nii.gz"
    ct_path = subj_dir / "ct.nii.gz"

    if not mr_path.exists() or not ct_path.exists():
        log.warning(f"[{subj_id}] Missing mr.nii.gz or ct.nii.gz — skipping")
        return {"subj": subj_id, "status": "skipped_missing"}

    if already_done(out_dir, subj_id):
        log.info(f"[{subj_id}] Already processed — skipping")
        return {"subj": subj_id, "status": "skipped_done"}

    try:
        # ── MRI pipeline ──────────────────────────────────────────────
        log.info(f"[{subj_id}] Starting MRI pipeline...")
        mr_outputs = preprocess_mri_full(
            raw_path=str(mr_path),
            out_dir=str(out_dir),
            subj_id=subj_id,
            use_hdbet=use_hdbet,
            device="cpu",
        )

        # ── CT pipeline ───────────────────────────────────────────────
        log.info(f"[{subj_id}] Starting CT pipeline...")
        ct_outputs = preprocess_ct_full(
            raw_path=str(ct_path),
            out_dir=str(out_dir),
            subj_id=subj_id,
            mask_path=mr_outputs.get("mr_mask"),
        )

        log.info(f"[{subj_id}] ✓ Done")
        return {"subj": subj_id, "status": "ok", **mr_outputs, **ct_outputs}

    except Exception as e:
        log.error(f"[{subj_id}] FAILED: {e}")
        log.debug(traceback.format_exc())
        return {"subj": subj_id, "status": "error", "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj",     default=None, help="Single subject ID")
    ap.add_argument("--n",        type=int, default=None, help="Process first N subjects")
    ap.add_argument("--no-hdbet", action="store_true", help="Use fallback skull strip")
    ap.add_argument("--dry-run",  action="store_true", help="List subjects only, no processing")
    args = ap.parse_args()

    subjects   = get_subjects(args.subj, args.n)
    use_hdbet  = not args.no_hdbet

    log.info(f"Subjects to process : {len(subjects)}")
    log.info(f"HD-BET skull strip  : {'yes' if use_hdbet else 'no (fallback)'}")
    log.info(f"Output dir          : {DATA_PROC}")

    if args.dry_run:
        for s in subjects:
            print(f"  {s.name}")
        return

    # ── Batch loop ────────────────────────────────────────────────────
    results = {"ok": [], "skipped_done": [], "skipped_missing": [], "error": []}

    for i, subj_dir in enumerate(subjects, 1):
        log.info(f"── [{i}/{len(subjects)}] {subj_dir.name} ──")
        r = process_subject(subj_dir, use_hdbet=use_hdbet)
        results[r["status"]].append(r["subj"])

    # ── Summary ───────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info(f"DONE  : {len(results['ok'])}")
    log.info(f"SKIP (already processed): {len(results['skipped_done'])}")
    log.info(f"SKIP (missing files)    : {len(results['skipped_missing'])}")
    log.info(f"ERROR : {len(results['error'])}")
    if results["error"]:
        log.error(f"Failed subjects: {results['error']}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
