"""
scripts/evaluate_voxelmorph.py
------------------------------
Evaluate a trained VoxelMorph checkpoint on the test split.

Primary metrics  : Dice, HD95, Jacobian neg%
Secondary metric : NCC  (sanity check only -- cross-modality, not meaningful as primary)

Usage
-----
    python scripts/evaluate_voxelmorph.py
    python scripts/evaluate_voxelmorph.py --checkpoint models/voxelmorph_last.pth
    python scripts/evaluate_voxelmorph.py --compare-baseline
    python scripts/evaluate_voxelmorph.py --tta            # logs warning, TTA not yet implemented
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.voxelmorph_model import VoxelMorph, SpatialTransformer
from src.dataloader        import get_dataloaders
from src.metrics           import (
    dice_coefficient,
    hausdorff95,
    jacobian_stats,
    normalised_cross_correlation,
)
from src.utils             import get_logger

log = get_logger("evaluate_voxelmorph")


def _load_checkpoint(path: Path, device: torch.device):
    """Load model weights; infer enc/dec feature sizes and diffeomorphic flag."""
    ckpt = torch.load(path, map_location=device)

    cfg           = ckpt.get("config", {})
    diffeomorphic = cfg.get("diffeomorphic", True)
    large         = cfg.get("large", False)

    enc = (32, 64, 64, 64) if large else (16, 32, 32, 32)
    dec = (64, 64, 64, 32) if large else (32, 32, 32, 16)

    model = VoxelMorph(
        enc_features  = enc,
        dec_features  = dec,
        diffeomorphic = diffeomorphic,
    ).to(device)

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    log.info(f"Loaded checkpoint: {path}  (epoch {epoch}, diffeomorphic={diffeomorphic})")
    return model


def _print_summary(rows: list, label: str):
    """Print mean +- std for primary + secondary metrics."""
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  {label} -- Per-subject summary (mean +- std)")
    print(f"{'='*60}")

    for col, unit, primary in [
        ("dice",        "",    True),
        ("hd95",        "mm",  True),
        ("jac_neg_pct", "%",   True),
        ("ncc",         "",    False),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        tag  = "" if primary else "  [secondary / sanity check -- cross-modality NCC]"
        print(f"  {col:<14}: {vals.mean():.4f} +- {vals.std():.4f} {unit}{tag}")
    print()


def _compare_baseline(vm_df: pd.DataFrame, baseline_path: Path):
    """Side-by-side table: VoxelMorph vs B-spline baseline."""
    if not baseline_path.exists():
        log.warning(f"Baseline file not found: {baseline_path}  -- skipping comparison.")
        return

    bl = pd.read_csv(baseline_path)
    print(f"\n{'='*70}")
    print("  Side-by-side: VoxelMorph  vs  B-spline Baseline")
    print(f"{'='*70}")
    print(f"  {'Metric':<14} {'VoxelMorph':>14} {'B-spline':>14}")
    print(f"  {'-'*44}")

    for col, unit in [("dice", ""), ("hd95", "mm")]:
        vm_val  = vm_df[col].dropna().mean()  if col in vm_df.columns  else float("nan")
        bl_val  = bl[col].dropna().mean()      if col in bl.columns     else float("nan")
        winner  = "VM wins" if (col == "dice" and vm_val > bl_val) or \
                               (col == "hd95"  and vm_val < bl_val) else "BL wins"
        print(f"  {col:<14} {vm_val:>13.4f}  {bl_val:>13.4f}   {winner}")
    print()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if args.tta:
        warnings.warn("TTA not yet implemented, running without it", UserWarning)

    model = _load_checkpoint(Path(args.checkpoint), device)

    # Nearest-neighbour transformer for binary mask warping (no interpolation artifacts)
    vol_size = (160, 192, 160)
    mask_transformer = SpatialTransformer(vol_size, mode="nearest").to(device)

    loaders     = get_dataloaders(augment=False, manifest=args.manifest)
    test_loader = loaders.get("test")
    if test_loader is None:
        log.error("Test DataLoader is None -- check manifest / dataset.")
        sys.exit(1)

    log.info(f"Evaluating {len(test_loader.dataset)} test subjects ...")

    rows    = []
    skipped = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            mr      = batch["mr"].to(device)        # (1,1,D,H,W)
            ct      = batch["ct"].to(device)
            mr_mask = batch["mask"].to(device)      # MRI brain mask
            ct_mask = batch.get("ct_mask")          # CT  brain mask

            subject_id = batch.get("subject_id", [f"subj_{i:03d}"])[0]

            # Hard skip if CT mask is missing -- never substitute MR mask as proxy
            if ct_mask is None:
                log.error(
                    f"{subject_id}: ct_mask missing from batch -- "
                    f"run scripts/generate_ct_mask_npy.py then retry. SKIPPING."
                )
                skipped += 1
                continue

            warped_ct, dvf = model(mr, ct)

            # Warp CT mask with nearest-neighbour (binary -- no interpolation artifacts)
            warped_ct_mask = mask_transformer(ct_mask.to(device).float(), dvf)

            # Convert to numpy
            mr_mask_np        = mr_mask[0, 0].cpu().numpy().astype(bool)
            warped_ct_mask_np = (warped_ct_mask[0, 0].cpu().numpy() > 0.5).astype(bool)
            mr_np             = mr[0, 0].cpu().numpy()
            warped_ct_np      = warped_ct[0, 0].cpu().numpy()
            dvf_np            = dvf[0].cpu().numpy()   # (3, D, H, W)

            dice   = dice_coefficient(mr_mask_np, warped_ct_mask_np)
            hd95   = hausdorff95(mr_mask_np, warped_ct_mask_np, voxel_size=1.0)
            jstats = jacobian_stats(dvf_np)
            ncc    = normalised_cross_correlation(mr_np, warped_ct_np, mask=mr_mask_np)

            row = {
                "subject_id":  subject_id,
                "dice":        round(dice, 4),
                "hd95":        round(hd95, 3),
                "ncc":         round(ncc,  4),    # secondary / sanity check
                **{k: round(v, 4) for k, v in jstats.items()},
            }
            rows.append(row)
            log.info(
                f"[{i+1:03d}/{len(test_loader.dataset)}] {subject_id} | "
                f"dice={dice:.4f}  hd95={hd95:.2f}mm  "
                f"jac_neg%={jstats['jac_neg_pct']:.2f}  ncc={ncc:.4f}[secondary]"
            )

    n_total = len(test_loader.dataset)
    log.info(f"Dice/HD95 computed on {len(rows)}/{n_total} subjects ({skipped} skipped — missing ct_mask)")

    out_path = ROOT / "results" / "voxelmorph_test_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info(f"Results saved -> {out_path}")

    _print_summary(rows, label=f"checkpoint: {args.checkpoint}")

    if args.compare_baseline:
        bl_path = ROOT / "results" / "baseline_metrics_bspline.csv"
        _compare_baseline(pd.DataFrame(rows), bl_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VoxelMorph on the test split.")
    p.add_argument("--checkpoint",       default="models/voxelmorph_best.pth")
    p.add_argument("--manifest",         default=None)
    p.add_argument("--tta",              action="store_true", default=False,
                   help="Test-time adaptation (NOT YET IMPLEMENTED)")
    p.add_argument("--compare-baseline", action="store_true", default=False,
                   help="Compare against results/baseline_metrics_bspline.csv")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
