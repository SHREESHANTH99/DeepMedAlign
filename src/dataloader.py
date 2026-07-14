"""
src/dataloader.py
------------------
DataLoader factory for MRI-CT registration training.

Returns train / val / test DataLoaders with the correct settings for
VoxelMorph 3D training:
  - batch_size = 1  (full 3D brain per sample — GPU memory constraint)
  - num_workers = 2 (safe on Windows; increase on Linux if faster)
  - pin_memory  = True (speeds up CPU→GPU tensor transfer)
  - train shuffled, val/test ordered (deterministic evaluation)

Usage
-----
    from src.dataloader import get_dataloaders

    loaders = get_dataloaders(batch_size=1, num_workers=2)
    for batch in loaders["train"]:
        mr   = batch["mr"]    # (B, 1, D, H, W)
        ct   = batch["ct"]    # (B, 1, D, H, W)
        mask = batch["mask"]  # (B, 1, D, H, W)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.config    import BATCH_SIZE
from src.dataset   import MedicalRegistrationDataset
from src.augmentation import get_train_transforms, get_val_transforms
from src.utils     import get_logger

log = get_logger("dataloader")

# Windows safe default
_DEFAULT_NUM_WORKERS = 0   # 0 = main process (avoids multiprocessing issues)


def get_dataloaders(
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = _DEFAULT_NUM_WORKERS,
    pin_memory:  bool = False,
    manifest:    Optional[str] = None,
    augment:     bool = True,
    elastic:     bool = False,
) -> Dict[str, DataLoader]:
    """Build and return train / val / test DataLoaders.

    Parameters
    ----------
    batch_size  : samples per batch (default 1 for 3D volumes)
    num_workers : DataLoader worker processes (default 0 for Windows safety)
    pin_memory  : pin CPU tensors for faster GPU transfer (default False)
    manifest    : path to manifest_processed.csv (None = auto-detect)
    augment     : apply augmentation to train split (default True)

    Returns
    -------
    dict with keys 'train', 'val', 'test' each containing a DataLoader
    """
    train_tf = get_train_transforms(elastic=elastic) if augment else get_val_transforms()
    val_tf   = get_val_transforms()

    datasets = {}
    for split, tf in [("train", train_tf),
                      ("val",   val_tf),
                      ("test",  val_tf)]:
        try:
            datasets[split] = MedicalRegistrationDataset(
                split=split,
                manifest=manifest,
                transform=tf,
                require_npy=True,
            )
        except FileNotFoundError as exc:
            log.warning(f"Could not build {split} dataset: {exc}")
            datasets[split] = None

    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        if ds is None or len(ds) == 0:
            log.warning(f"{split}: no samples — DataLoader skipped.")
            loaders[split] = None
            continue

        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = False,
        )
        log.info(
            f"DataLoader [{split}]: {len(ds)} samples, "
            f"batch={batch_size}, workers={num_workers}, "
            f"shuffle={shuffle}"
        )

    return loaders


def get_single_loader(
    split:       str  = "train",
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = _DEFAULT_NUM_WORKERS,
    augment:     bool = True,
    manifest:    Optional[str] = None,
) -> DataLoader:
    """Convenience function to get a single split's DataLoader.

    Parameters
    ----------
    split       : 'train', 'val', or 'test'
    batch_size  : samples per batch
    num_workers : DataLoader workers
    augment     : apply augmentation (only meaningful for train)
    manifest    : path to manifest CSV

    Returns
    -------
    DataLoader for the requested split
    """
    tf = get_train_transforms() if (augment and split == "train") \
        else get_val_transforms()

    ds = MedicalRegistrationDataset(
        split=split,
        manifest=manifest,
        transform=tf,
        require_npy=True,
    )

    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = False,
        drop_last   = False,
    )
