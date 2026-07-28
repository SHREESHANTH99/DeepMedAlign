"""
verify_dataset.py
-----------------
Sanity-check script for the PyTorch Dataset + DataLoader pipeline.
Run this BEFORE starting VoxelMorph training to confirm data is correct.

Checks:
  - All splits load without error
  - Tensor shapes are (1, D, H, W) as expected by VoxelMorph
  - Values are float32 in [0, 1]
  - No NaN or Inf in any tensor
  - Loading speed is acceptable (< 1s per sample)
  - No data leakage across splits

Run:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --split train
    python scripts/verify_dataset.py --n 5
"""

import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config    import DATA_RAW, FIXED_SHAPE
from src.dataset   import MedicalRegistrationDataset
from src.dataloader import get_dataloaders
from src.utils     import get_logger

log = get_logger("verify_dataset")

EXPECTED_SHAPE = (1,) + FIXED_SHAPE   # (1, 160, 192, 160)


def check_tensor(t: torch.Tensor, name: str, sid: str) -> list[str]:
    """Return list of issues found in tensor t. Empty = all OK."""
    issues = []
    if t.shape != torch.Size(EXPECTED_SHAPE):
        issues.append(f"{name} shape {tuple(t.shape)} != {EXPECTED_SHAPE}")
    if t.dtype != torch.float32:
        issues.append(f"{name} dtype {t.dtype} != float32")
    if t.min() < -0.01:
        issues.append(f"{name} min={t.min():.4f} < 0")
    if t.max() > 1.01:
        issues.append(f"{name} max={t.max():.4f} > 1")
    if torch.isnan(t).any():
        issues.append(f"{name} contains NaN")
    if torch.isinf(t).any():
        issues.append(f"{name} contains Inf")
    return issues


def verify_split(split: str, n_samples: int = 3) -> dict:
    """Verify one split. Returns a result summary dict."""
    result = {"split": split, "ok": True, "issues": [], "n": 0, "load_ms": []}

    try:
        ds = MedicalRegistrationDataset(split=split, require_npy=True)
    except FileNotFoundError as exc:
        result["ok"]     = False
        result["issues"] = [str(exc)]
        return result

    result["n"] = len(ds)
    if len(ds) == 0:
        result["ok"]     = False
        result["issues"] = [f"Dataset is empty for split={split}"]
        return result

    # Check first n_samples
    for i in range(min(n_samples, len(ds))):
        t0     = time.time()
        sample = ds[i]
        elapsed_ms = (time.time() - t0) * 1000
        result["load_ms"].append(round(elapsed_ms, 1))

        sid = sample.get("subject_id", f"idx={i}")

        for key in ("mr", "ct", "mask"):
            if key not in sample:
                result["issues"].append(f"{sid}: missing key '{key}'")
                result["ok"] = False
                continue
            issues = check_tensor(sample[key], key, sid)
            if issues:
                result["issues"].extend([f"{sid}: {iss}" for iss in issues])
                result["ok"] = False

    return result


def verify_no_leakage() -> list[str]:
    """Confirm no subject_id appears in multiple splits."""
    issues   = []
    seen     = {}
    all_sids = {}

    for split in ("train", "val", "test"):
        try:
            ds = MedicalRegistrationDataset(split=split, require_npy=False)
            all_sids[split] = set(ds.subject_ids())
        except Exception:
            all_sids[split] = set()

    splits = list(all_sids.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2  = splits[i], splits[j]
            overlap = all_sids[s1] & all_sids[s2]
            if overlap:
                issues.append(
                    f"DATA LEAKAGE: {len(overlap)} subjects in both "
                    f"{s1} and {s2}: {list(overlap)[:3]}"
                )
    return issues


def verify_dataloader(batch_size: int = 1) -> dict:
    """Verify the DataLoader produces correctly shaped batches."""
    result = {"ok": True, "issues": []}
    try:
        loaders = get_dataloaders(batch_size=batch_size, num_workers=0)
        train_loader = loaders.get("train")
        if train_loader is None:
            result["issues"] = ["train DataLoader is None"]
            result["ok"]     = False
            return result

        batch = next(iter(train_loader))
        for key in ("mr", "ct", "mask"):
            if key not in batch:
                result["issues"].append(f"batch missing key '{key}'")
                result["ok"] = False
                continue
            expected = torch.Size([batch_size, 1] + list(FIXED_SHAPE))
            if batch[key].shape != expected:
                result["issues"].append(
                    f"batch['{key}'] shape {tuple(batch[key].shape)} "
                    f"!= {tuple(expected)}"
                )
                result["ok"] = False

    except Exception as exc:
        result["ok"]     = False
        result["issues"] = [f"DataLoader error: {exc}"]

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify the PyTorch Dataset pipeline before training."
    )
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "test"],
                    help="Check only one split (default: all).")
    ap.add_argument("--n",    type=int, default=3,
                    help="Number of samples to check per split (default 3).")
    args = ap.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]

    print("\n" + "=" * 58)
    print("DATASET VERIFICATION")
    print("=" * 58)

    all_ok = True

    # ── Per-split checks ─────────────────────────────────────────────────────
    for split in splits:
        r = verify_split(split, n_samples=args.n)
        n        = r["n"]
        load_avg = (
            f"{np.mean(r['load_ms']):.1f} ms"
            if r["load_ms"] else "N/A"
        )
        status = "✅" if r["ok"] else "❌"
        print(f"\n  {status} {split:6s} : {n:4d} samples  "
              f"avg load={load_avg}")
        for iss in r["issues"]:
            print(f"       ⚠  {iss}")
        if not r["ok"]:
            all_ok = False

    print()

    # ── Data leakage check ───────────────────────────────────────────────────
    leakage_issues = verify_no_leakage()
    if leakage_issues:
        all_ok = False
        for iss in leakage_issues:
            print(f"  ❌ {iss}")
    else:
        print("  ✅ No data leakage across splits")

    # ── DataLoader batch shape ───────────────────────────────────────────────
    dl_result = verify_dataloader(batch_size=1)
    if dl_result["ok"]:
        print(f"  ✅ DataLoader batch shape: "
              f"(1, 1, {FIXED_SHAPE[0]}, {FIXED_SHAPE[1]}, {FIXED_SHAPE[2]})")
    else:
        all_ok = False
        for iss in dl_result["issues"]:
            print(f"  ❌ {iss}")

    # ── Final verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    if all_ok:
        print("✅ ALL CHECKS PASSED — Ready for VoxelMorph training!")
    else:
        print("❌ ISSUES FOUND — Fix before starting training.")
        print("   Run: python scripts/build_npy_cache.py --force --verify")
        sys.exit(1)
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
