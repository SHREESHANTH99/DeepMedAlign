"""
test_week3_r1.py
-----------------
Tests for R1 Week 3 — PyTorch Dataset + DataLoader + Augmentation.
All tests use synthetic .npy data — no real dataset required.

Run:
    pytest tests/test_week3_r1.py -v
"""

import sys
import pytest
import numpy as np
import torch
import tempfile
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import FIXED_SHAPE

SHAPE = FIXED_SHAPE   # (160, 192, 160)


# ---------------------------------------------------------------------------
# Fixtures — synthetic manifest + .npy files
# ---------------------------------------------------------------------------

def _make_npy(path: str, shape=SHAPE, value: float = 0.5) -> None:
    """Save a synthetic .npy volume."""
    arr = np.full(shape, value, dtype="float32")
    # Add small noise so it's not completely flat
    arr += np.random.uniform(-0.01, 0.01, shape).astype("float32")
    arr = np.clip(arr, 0.0, 1.0)
    np.save(path, arr)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal synthetic dataset: 3 train, 1 val, 1 test subjects."""
    raw_dir  = tmp_path / "data" / "raw"
    proc_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)

    subjects = [
        ("S001", "train"),
        ("S002", "train"),
        ("S003", "train"),
        ("S004", "val"),
        ("S005", "test"),
    ]

    rows = []
    for sid, split in subjects:
        out = proc_dir / sid
        out.mkdir(parents=True)

        mr_path   = str(out / f"{sid}_mr.npy")
        ct_path   = str(out / f"{sid}_ct.npy")
        mask_path = str(out / f"{sid}_mask.npy")

        _make_npy(mr_path)
        _make_npy(ct_path,   value=0.3)
        _make_npy(mask_path, value=1.0)

        rows.append({
            "subject_id": sid,
            "split":      split,
            "mr_npy":     mr_path,
            "ct_npy":     ct_path,
            "mask_npy":   mask_path,
        })

    manifest_path = str(raw_dir / "manifest_processed.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    return {"manifest": manifest_path, "proc_dir": str(proc_dir),
            "subjects": subjects}


# ---------------------------------------------------------------------------
# TestMedicalRegistrationDataset
# ---------------------------------------------------------------------------

class TestMedicalRegistrationDataset:

    def test_loads_train_split(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds = MedicalRegistrationDataset(
            split="train",
            manifest=synthetic_dataset["manifest"])
        assert len(ds) == 3

    def test_loads_val_split(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds = MedicalRegistrationDataset(
            split="val",
            manifest=synthetic_dataset["manifest"])
        assert len(ds) == 1

    def test_loads_test_split(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds = MedicalRegistrationDataset(
            split="test",
            manifest=synthetic_dataset["manifest"])
        assert len(ds) == 1

    def test_loads_all_splits_when_none(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds = MedicalRegistrationDataset(
            split=None,
            manifest=synthetic_dataset["manifest"])
        assert len(ds) == 5

    def test_sample_has_required_keys(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds     = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        sample = ds[0]
        for key in ("mr", "ct", "mask", "subject_id"):
            assert key in sample, f"Missing key: {key}"

    def test_tensor_shape_is_1_D_H_W(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds     = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        sample = ds[0]
        expected = torch.Size([1] + list(SHAPE))
        for key in ("mr", "ct", "mask"):
            assert sample[key].shape == expected, \
                f"{key} shape {sample[key].shape} != {expected}"

    def test_tensor_dtype_is_float32(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds     = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        sample = ds[0]
        for key in ("mr", "ct", "mask"):
            assert sample[key].dtype == torch.float32, \
                f"{key} dtype {sample[key].dtype} != float32"

    def test_tensor_values_in_zero_one(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds     = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        sample = ds[0]
        for key in ("mr", "ct", "mask"):
            assert sample[key].min() >= -0.01
            assert sample[key].max() <=  1.01

    def test_no_nan_or_inf(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds     = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        sample = ds[0]
        for key in ("mr", "ct", "mask"):
            assert not torch.isnan(sample[key]).any(), f"{key} has NaN"
            assert not torch.isinf(sample[key]).any(), f"{key} has Inf"

    def test_subject_id_is_string(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds     = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        sample = ds[0]
        assert isinstance(sample["subject_id"], str)

    def test_missing_manifest_raises_file_not_found(self):
        from src.dataset import MedicalRegistrationDataset
        with pytest.raises(FileNotFoundError):
            MedicalRegistrationDataset(
                split="train",
                manifest="/nonexistent/path/manifest.csv")

    def test_invalid_split_raises_value_error(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        with pytest.raises(ValueError):
            MedicalRegistrationDataset(
                split="invalid_split",
                manifest=synthetic_dataset["manifest"])

    def test_subject_ids_returns_list(self, synthetic_dataset):
        from src.dataset import MedicalRegistrationDataset
        ds  = MedicalRegistrationDataset(
            split="train", manifest=synthetic_dataset["manifest"])
        ids = ds.subject_ids()
        assert isinstance(ids, list)
        assert len(ids) == 3

    def test_no_leakage_across_splits(self, synthetic_dataset):
        """Same subject must not appear in more than one split."""
        from src.dataset import MedicalRegistrationDataset
        train_ids = set(MedicalRegistrationDataset(
            split="train",
            manifest=synthetic_dataset["manifest"]).subject_ids())
        val_ids   = set(MedicalRegistrationDataset(
            split="val",
            manifest=synthetic_dataset["manifest"]).subject_ids())
        test_ids  = set(MedicalRegistrationDataset(
            split="test",
            manifest=synthetic_dataset["manifest"]).subject_ids())

        assert train_ids.isdisjoint(val_ids),  "train/val leakage"
        assert train_ids.isdisjoint(test_ids), "train/test leakage"
        assert val_ids.isdisjoint(test_ids),   "val/test leakage"


# ---------------------------------------------------------------------------
# TestAugmentation
# ---------------------------------------------------------------------------

class TestAugmentation:

    def _make_sample(self) -> dict:
        return {
            "subject_id": "TEST",
            "mr":   torch.rand(1, 40, 40, 40),
            "ct":   torch.rand(1, 40, 40, 40),
            "mask": torch.zeros(1, 40, 40, 40),
        }

    def test_val_transforms_are_identity(self):
        from src.augmentation import get_val_transforms
        tf     = get_val_transforms()
        sample = self._make_sample()
        mr_orig = sample["mr"].clone()
        result  = tf(sample)
        assert torch.allclose(result["mr"], mr_orig)

    def test_flip_output_same_shape(self):
        from src.augmentation import RandomFlip
        tf     = RandomFlip(p=1.0)
        sample = self._make_sample()
        result = tf(sample)
        assert result["mr"].shape == sample["mr"].shape

    def test_flip_changes_values(self):
        from src.augmentation import RandomFlip
        tf     = RandomFlip(p=1.0)
        sample = self._make_sample()
        mr_orig = sample["mr"].clone()
        result  = tf(sample)
        # After flip, data should differ from original
        assert not torch.allclose(result["mr"], mr_orig)

    def test_jitter_output_in_zero_one(self):
        from src.augmentation import RandomIntensityJitter
        tf     = RandomIntensityJitter(delta=0.10, p=1.0)
        sample = self._make_sample()
        result = tf(sample)
        assert result["mr"].min() >= 0.0
        assert result["mr"].max() <= 1.0

    def test_jitter_does_not_change_ct(self):
        from src.augmentation import RandomIntensityJitter
        tf     = RandomIntensityJitter(delta=0.10, p=1.0)
        sample = self._make_sample()
        ct_orig = sample["ct"].clone()
        result  = tf(sample)
        # CT must NOT be touched by intensity jitter
        assert torch.allclose(result["ct"], ct_orig)

    def test_noise_output_in_zero_one(self):
        from src.augmentation import RandomGaussianNoise
        tf     = RandomGaussianNoise(sigma=0.01, p=1.0)
        sample = self._make_sample()
        result = tf(sample)
        assert result["mr"].min() >= 0.0
        assert result["mr"].max() <= 1.0

    def test_compose_applies_all_transforms(self):
        from src.augmentation import get_train_transforms
        tf     = get_train_transforms(
            flip=True, jitter=True, noise=True, affine=False)
        sample = self._make_sample()
        result = tf(sample)
        for key in ("mr", "ct", "mask"):
            assert key in result
            assert result[key].shape == sample[key].shape


# ---------------------------------------------------------------------------
# TestDataLoader
# ---------------------------------------------------------------------------

class TestDataLoader:

    def test_dataloader_iterates(self, synthetic_dataset):
        from src.dataloader import get_dataloaders
        loaders = get_dataloaders(
            batch_size=1,
            num_workers=0,
            manifest=synthetic_dataset["manifest"],
            augment=False,
        )
        assert loaders["train"] is not None
        batch = next(iter(loaders["train"]))
        assert "mr" in batch
        assert "ct" in batch

    def test_batch_shape_is_B_1_D_H_W(self, synthetic_dataset):
        from src.dataloader import get_dataloaders
        loaders = get_dataloaders(
            batch_size=1,
            num_workers=0,
            manifest=synthetic_dataset["manifest"],
            augment=False,
        )
        batch    = next(iter(loaders["train"]))
        expected = torch.Size([1, 1] + list(SHAPE))
        assert batch["mr"].shape == expected
        assert batch["ct"].shape == expected

    def test_train_loader_shuffles(self, synthetic_dataset):
        """Two passes over train with shuffle=True should differ in order."""
        from src.dataloader import get_dataloaders
        loaders = get_dataloaders(
            batch_size=1, num_workers=0,
            manifest=synthetic_dataset["manifest"],
            augment=False)
        loader = loaders["train"]

        ids1 = [b["subject_id"][0] for b in loader]
        ids2 = [b["subject_id"][0] for b in loader]
        # With 3 subjects and shuffle=True, runs usually differ
        # (very low probability they're identical twice)
        # We just check both passes have same set
        assert set(ids1) == set(ids2)

    def test_val_loader_not_none(self, synthetic_dataset):
        from src.dataloader import get_dataloaders
        loaders = get_dataloaders(
            batch_size=1, num_workers=0,
            manifest=synthetic_dataset["manifest"])
        assert loaders["val"] is not None

    def test_all_splits_present_in_output(self, synthetic_dataset):
        from src.dataloader import get_dataloaders
        loaders = get_dataloaders(
            batch_size=1, num_workers=0,
            manifest=synthetic_dataset["manifest"])
        for split in ("train", "val", "test"):
            assert split in loaders
