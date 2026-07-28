"""
train_voxelmorph.py
--------------------
Training loop for VoxelMorph MRI-CT deformable registration.

What this script does:
  1. Loads train/val DataLoaders from R1's pipeline
  2. Trains VoxelMorph for EPOCHS iterations
  3. Logs MIND loss + reg loss + NCC every epoch
  4. Saves best model checkpoint (lowest val loss)
  5. Saves training CSV for R3 visualisation

Run:
    python scripts/train_voxelmorph.py
    python scripts/train_voxelmorph.py --epochs 50 --lr 1e-4
    python scripts/train_voxelmorph.py --resume models/voxelmorph_best.pth
    python scripts/train_voxelmorph.py --device cpu
"""

import sys
import time
import argparse
import csv
import platform
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config           import MODELS, RESULTS, EPOCHS, LR, LAMBDA_SMOOTH
from src.voxelmorph_model import VoxelMorph, SpatialTransformer
from src.losses           import total_loss
from src.dataloader       import get_dataloaders
from src.metrics          import normalised_cross_correlation as ncc
from src.utils            import get_logger

log = get_logger("train_voxelmorph")


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_reg, sigma,
                    mask_tf=None, lambda_dice=1.0, lambda_jacobian=1.0):
    model.train()
    total = mi_sum = reg_sum = dice_sum = jac_sum = 0.0
    n = 0

    for batch in loader:
        mr   = batch["mr"].to(device)
        ct   = batch["ct"].to(device)
        mask = batch["mask"].to(device) if mask_tf else None

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            warped_ct, dvf = model(mr, ct)
            w_mask = mask_tf(mask.float(), dvf) if mask_tf else None
            loss, losses = total_loss(warped_ct, mr, dvf, lambda_reg, sigma,
                                      warped_mask=w_mask, target_mask=mask,
                                      lambda_dice=lambda_dice,
                                      lambda_jacobian=lambda_jacobian)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total    += losses["total"]
        mi_sum   += losses["mi"]
        reg_sum  += losses["reg"]
        jac_sum  += losses.get("jac_loss", 0.0)
        dice_sum += losses.get("dice_loss", 0.0)
        n        += 1

    avg = {"train_loss": total/max(n,1), "train_mi": mi_sum/max(n,1),
           "train_reg": reg_sum/max(n,1), "train_jac_loss": jac_sum/max(n,1)}
    if mask_tf:
        avg["train_dice_loss"] = dice_sum / max(n, 1)
    return avg


@torch.no_grad()
def validate(model, loader, device, lambda_reg, sigma, mask_tf=None, lambda_dice=1.0, lambda_jacobian=1.0):
    model.eval()
    total = mi_sum = reg_sum = ncc_sum = dice_sum = jac_sum = 0.0
    n = 0

    for batch in loader:
        mr   = batch["mr"].to(device)
        ct   = batch["ct"].to(device)
        mask = batch["mask"].to(device) if mask_tf else None

        with torch.cuda.amp.autocast():
            warped_ct, dvf = model(mr, ct)
            w_mask = mask_tf(mask.float(), dvf) if mask_tf else None
            loss, losses = total_loss(warped_ct, mr, dvf, lambda_reg, sigma,
                                      warped_mask=w_mask, target_mask=mask,
                                      lambda_dice=lambda_dice,
                                      lambda_jacobian=lambda_jacobian)

        ncc_val = ncc(
            mr[0, 0].cpu().numpy(),
            warped_ct[0, 0].cpu().numpy(),
        )

        total    += losses["total"]
        mi_sum   += losses["mi"]
        reg_sum  += losses["reg"]
        jac_sum  += losses.get("jac_loss", 0.0)
        dice_sum += losses.get("dice_loss", 0.0)
        ncc_sum  += ncc_val
        n        += 1

    avg = {"val_loss": total/max(n,1), "val_mi": mi_sum/max(n,1),
           "val_reg": reg_sum/max(n,1), "val_jac_loss": jac_sum/max(n,1), "val_ncc": ncc_sum/max(n,1)}
    if mask_tf:
        avg["val_dice_loss"] = dice_sum / max(n, 1)
    return avg


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    log.info(f"Checkpoint saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--lambda-reg", type=float, default=LAMBDA_SMOOTH,
                    dest="lambda_reg")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint to resume from")
    default_workers = 4 if platform.system() == "Linux" else 0
    ap.add_argument("--workers",       type=int, default=default_workers)
    ap.add_argument("--diffeomorphic", action="store_true",
                    help="Use diffeomorphic (no-fold) integration in model")
    ap.add_argument("--cosine",        action="store_true",
                    help="Use CosineAnnealingWarmRestarts instead of ReduceLROnPlateau")
    # Method 1: Two-Stage Curriculum — switch sigma without touching losses.py
    ap.add_argument("--sigma",  type=float, default=0.1,
                    help="MI loss sigma: 0.1=fast Stage-1, 0.05=precise Stage-2")
    # Method 2: Larger Network — doubles U-Net capacity for higher NCC ceiling
    ap.add_argument("--large",  action="store_true",
                    help="Use (32,64,64,64) U-Net instead of default (16,32,32,32)")
    # Method 3: Elastic Augmentation — forces learning of complex deformations
    ap.add_argument("--elastic", action="store_true",
                    help="Add random elastic deformation to training augmentation")
    ap.add_argument("--lambda-dice", type=float, default=1.0, dest="lambda_dice",
                    help="Weight for soft Dice mask loss (default 1.0)")
    ap.add_argument("--no-dice-loss", action="store_true",
                    help="Disable soft Dice mask loss")
    ap.add_argument("--lambda-jacobian", type=float, default=1.0, dest="lambda_jacobian",
                    help="Weight for Jacobian determinant folding loss (default 1.0)")
    ap.add_argument("--out-prefix", default="voxelmorph",
                    help="Checkpoint filename prefix (default: voxelmorph -> voxelmorph_best.pth)")
    args = ap.parse_args()

    device = get_device(args.device)
    log.info(f"Device: {device}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        batch_size=1,
        num_workers=args.workers,
        augment=True,
        elastic=args.elastic,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    if train_loader is None or len(train_loader.dataset) == 0:
        log.error("Train dataset is empty. Run build_npy_cache.py first.")
        sys.exit(1)

    log.info(f"Train: {len(train_loader.dataset)} subjects")
    log.info(f"Val  : {len(val_loader.dataset)} subjects")
    log.info(f"DataLoader workers: {args.workers}")

    # ── Model ────────────────────────────────────────────────────────────────
    enc = (32, 64, 64, 64) if args.large else (16, 32, 32, 32)
    dec = (64, 64, 64, 32) if args.large else (32, 32, 32, 16)
    log.info(f"Network: {'LARGE' if args.large else 'standard'}  enc={enc}")
    use_dice = not args.no_dice_loss
    log.info(f"MI sigma: {args.sigma}  elastic: {args.elastic}  dice_loss: {use_dice} (λ={args.lambda_dice})  jac_loss: λ={args.lambda_jacobian}")
    mask_tf = SpatialTransformer((160, 192, 160), mode="bilinear").to(device) if use_dice else None
    model     = VoxelMorph(enc_features=enc, dec_features=dec,
                           diffeomorphic=args.diffeomorphic).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=1e-6)
        log.info("Scheduler: CosineAnnealingWarmRestarts (T_0=100, T_mult=2)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=50, factor=0.5, min_lr=1e-6)
        log.info("Scheduler: ReduceLROnPlateau (patience=50, factor=0.5)")

    # ── AMP Scaler (GPU only) ─────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    log.info(f"AMP (Mixed Precision): {'ENABLED' if use_amp else 'disabled'}")

    # ── torch.compile (PyTorch 2.0+, best on Linux/Kaggle) ────────────────────
    if hasattr(torch, "compile") and device.type == "cuda" and platform.system() == "Linux":
        try:
            model = torch.compile(model)
            log.info("torch.compile: ENABLED (15-30%% extra speedup)")
        except Exception as e:
            log.warning(f"torch.compile skipped: {e}")
    else:
        log.info("torch.compile: skipped (Windows / CPU / older PyTorch)")

    start_epoch = 0
    best_val    = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        log.info(f"Resumed from epoch {start_epoch}, val_loss={best_val:.4f}")

    # ── Output paths ─────────────────────────────────────────────────────────
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    best_path = str(MODELS / f"{args.out_prefix}_best.pth")
    last_path = str(MODELS / f"{args.out_prefix}_last.pth")
    log_path  = str(RESULTS / "training_log.csv")

    # ── CSV log ──────────────────────────────────────────────────────────────
    csv_fields = ["epoch", "train_loss", "train_mi", "train_reg", "train_jac_loss",
                  "val_loss", "val_mi", "val_reg", "val_jac_loss", "val_ncc", "lr"]
    if use_dice:
        csv_fields.extend(["train_dice_loss", "val_dice_loss"])
    write_header = not Path(log_path).exists()
    csv_file     = open(log_path, "a", newline="")
    writer       = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"Training for {args.epochs} epochs")
    log.info(f"lambda_reg={args.lambda_reg}  lr={args.lr}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0         = time.time()
        train_logs = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args.lambda_reg, args.sigma,
            mask_tf=mask_tf, lambda_dice=args.lambda_dice, lambda_jacobian=args.lambda_jacobian)
        val_logs   = validate(
            model, val_loader, device, args.lambda_reg, args.sigma,
            mask_tf=mask_tf, lambda_dice=args.lambda_dice, lambda_jacobian=args.lambda_jacobian)

        if args.cosine:
            scheduler.step()
        else:
            scheduler.step(val_logs["val_loss"])
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr":    cur_lr,
            **train_logs,
            **val_logs,
        }
        writer.writerow(row)
        csv_file.flush()

        # Save best model
        if val_logs["val_loss"] < best_val:
            best_val = val_logs["val_loss"]
            save_checkpoint(model, optimizer, epoch, best_val, best_path)

        # Save latest model every 50 epochs
        if epoch % 50 == 0:
            save_checkpoint(model, optimizer, epoch,
                            val_logs["val_loss"], last_path)

        elapsed = time.time() - t0
        dice_str = f"  dice={train_logs['train_dice_loss']:.4f}" if use_dice else ""
        print(
            f"Epoch {epoch:4d} | "
            f"train={train_logs['train_loss']:.6f} "
            f"(mi={train_logs['train_mi']:.6f} "
            f"reg={train_logs['train_reg']:.6f} "
            f"jac={train_logs['train_jac_loss']:.6f}{dice_str}) | "
            f"val={val_logs['val_loss']:.6f}  "
            f"ncc={val_logs['val_ncc']:.4f} | "
            f"{elapsed:.1f}s"
        )

    csv_file.close()
    log.info(f"Training complete. Best val loss: {best_val:.4f}")
    log.info(f"Best model: {best_path}")
    log.info(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
