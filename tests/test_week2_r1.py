"""
test_week2_r1.py
-----------------
Tests for R1 Week 2 data pipeline scripts.
All tests use synthetic data or dummy paths — no real dataset required.

Run:
    pytest tests/test_week2_r1.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, FIXED_SHAPE


# ---------------------------------------------------------------------------
# TestNiiToNpy
# ---------------------------------------------------------------------------

class TestNiiToNpy:

    def test_converts_successfully(self, tmp_path):
        import nibabel as nib
        from scripts.build_npy_cache import nii_to_npy

        arr = np.random.rand(40, 40, 40).astype("float32")
        nib.save(nib.Nifti1Image(arr, np.eye(4)),
                 str(tmp_path / "t.nii.gz"))
        saved = nii_to_npy(str(tmp_path / "t.nii.gz"),
                            str(tmp_path / "t.npy"),
                            expected_shape=(40, 40, 40))
        assert saved is True
        assert (tmp_path / "t.npy").exists()

    def test_loaded_values_match_original(self, tmp_path):
        import nibabel as nib
        from scripts.build_npy_cache import nii_to_npy

        arr = np.random.rand(40, 40, 40).astype("float32")
        nib.save(nib.Nifti1Image(arr, np.eye(4)),
                 str(tmp_path / "t.nii.gz"))
        nii_to_npy(str(tmp_path / "t.nii.gz"),
                   str(tmp_path / "t.npy"),
                   expected_shape=(40, 40, 40))
        loaded = np.load(str(tmp_path / "t.npy"))
        assert np.allclose(arr, loaded, atol=1e-5)

    def test_skips_when_npy_exists_and_no_force(self, tmp_path):
        import nibabel as nib
        from scripts.build_npy_cache import nii_to_npy

        arr = np.ones((10, 10, 10), dtype="float32")
        nib.save(nib.Nifti1Image(arr, np.eye(4)),
                 str(tmp_path / "t.nii.gz"))
        np.save(str(tmp_path / "t.npy"), arr)  # pre-existing file

        saved = nii_to_npy(str(tmp_path / "t.nii.gz"),
                            str(tmp_path / "t.npy"),
                            expected_shape=(10, 10, 10),
                            force=False)
        assert saved is False

    def test_force_overwrites_existing(self, tmp_path):
        import nibabel as nib
        from scripts.build_npy_cache import nii_to_npy

        arr = np.random.rand(10, 10, 10).astype("float32")
        nib.save(nib.Nifti1Image(arr, np.eye(4)),
                 str(tmp_path / "t.nii.gz"))
        np.save(str(tmp_path / "t.npy"), np.zeros((10, 10, 10)))

        saved = nii_to_npy(str(tmp_path / "t.nii.gz"),
                            str(tmp_path / "t.npy"),
                            expected_shape=(10, 10, 10),
                            force=True)
        assert saved is True
        loaded = np.load(str(tmp_path / "t.npy"))
        assert np.allclose(arr, loaded, atol=1e-5)

    def test_missing_nii_returns_false(self, tmp_path):
        from scripts.build_npy_cache import nii_to_npy

        result = nii_to_npy(str(tmp_path / "no_such.nii.gz"),
                             str(tmp_path / "out.npy"),
                             expected_shape=(10, 10, 10))
        assert result is False

    def test_output_cropped_to_target_shape(self, tmp_path):
        """Volume larger than target shape must be cropped."""
        import nibabel as nib
        from scripts.build_npy_cache import nii_to_npy

        arr = np.random.rand(80, 80, 80).astype("float32")
        nib.save(nib.Nifti1Image(arr, np.eye(4)),
                 str(tmp_path / "large.nii.gz"))
        nii_to_npy(str(tmp_path / "large.nii.gz"),
                   str(tmp_path / "large.npy"),
                   expected_shape=(40, 40, 40),
                   force=True)
        assert np.load(str(tmp_path / "large.npy")).shape == (40, 40, 40)

    def test_output_padded_to_target_shape(self, tmp_path):
        """Volume smaller than target shape must be zero-padded."""
        import nibabel as nib
        from scripts.build_npy_cache import nii_to_npy

        arr = np.random.rand(20, 20, 20).astype("float32")
        nib.save(nib.Nifti1Image(arr, np.eye(4)),
                 str(tmp_path / "small.nii.gz"))
        nii_to_npy(str(tmp_path / "small.nii.gz"),
                   str(tmp_path / "small.npy"),
                   expected_shape=(40, 40, 40),
                   force=True)
        loaded = np.load(str(tmp_path / "small.npy"))
        assert loaded.shape == (40, 40, 40)
        # Original data should be in the top-left corner
        assert np.allclose(loaded[:20, :20, :20], arr, atol=1e-5)
        # Padding should be zeros
        assert loaded[20:, :, :].sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestFindPreprocessedFiles
# ---------------------------------------------------------------------------

class TestFindPreprocessedFiles:

    def test_returns_all_required_keys(self):
        from scripts.update_manifest_preprocessed import find_preprocessed_files
        result = find_preprocessed_files("FAKE_SID")
        required = ["mr_preprocessed", "mr_mask", "ct_preprocessed",
                    "ct_mask", "ct_affine", "preprocess_status",
                    "mr_npy", "ct_npy", "mask_npy"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_missing_subject_file_paths_are_empty_string(self):
        from scripts.update_manifest_preprocessed import find_preprocessed_files
        result = find_preprocessed_files("DEFINITELY_NOT_REAL_9999")
        file_keys = ["mr_preprocessed", "ct_preprocessed",
                     "mr_mask", "ct_mask", "ct_affine",
                     "mr_npy", "ct_npy", "mask_npy"]
        for key in file_keys:
            assert result[key] == "", \
                f"{key} should be '' for missing subject, got {result[key]!r}"

    def test_missing_subject_status_is_missing(self):
        from scripts.update_manifest_preprocessed import find_preprocessed_files
        result = find_preprocessed_files("FAKE_SID_999")
        assert result["preprocess_status"] == "missing"

    def test_does_not_raise_on_any_input(self):
        """Function must never raise, even with weird inputs."""
        from scripts.update_manifest_preprocessed import find_preprocessed_files
        for sid in ["", " ", "../../etc/passwd", "1" * 200]:
            try:
                find_preprocessed_files(sid)
            except Exception as exc:
                pytest.fail(f"Raised on sid={sid!r}: {exc}")


# ---------------------------------------------------------------------------
# TestProcessOneSubject
# ---------------------------------------------------------------------------

class TestProcessOneSubject:

    def test_missing_files_return_skip_or_error(self):
        from scripts.run_preprocessing_batch import process_one_subject
        row    = pd.Series({"subject_id": "FAKE",
                             "mr": "nonexistent/mr.nii.gz",
                             "ct": "nonexistent/ct.nii.gz"})
        result = process_one_subject(row)
        assert ("skip" in result["preprocess_status"]
                or "error" in result["preprocess_status"])

    def test_result_always_contains_subject_id(self):
        from scripts.run_preprocessing_batch import process_one_subject
        row    = pd.Series({"subject_id": "SID_123",
                             "mr": "fake/mr.nii.gz",
                             "ct": "fake/ct.nii.gz"})
        result = process_one_subject(row)
        assert result["subject_id"] == "SID_123"

    def test_never_raises(self):
        """process_one_subject must never propagate an exception."""
        from scripts.run_preprocessing_batch import process_one_subject
        row = pd.Series({"subject_id": "CRASH_TEST",
                          "mr": "", "ct": ""})
        try:
            process_one_subject(row)
        except Exception as exc:
            pytest.fail(f"process_one_subject raised: {exc}")


# ---------------------------------------------------------------------------
# TestManifestIntegrity
# ---------------------------------------------------------------------------

class TestManifestIntegrity:

    def test_manifest_csv_exists(self):
        p = DATA_RAW / "manifest.csv"
        assert p.exists(), "Run: python scripts/build_manifest.py first"

    def test_manifest_has_required_columns(self):
        p = DATA_RAW / "manifest.csv"
        if not p.exists():
            pytest.skip("manifest.csv not found")
        df = pd.read_csv(p)
        for col in ["subject_id", "split", "mr", "ct"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_splits_are_valid_values(self):
        p = DATA_RAW / "manifest.csv"
        if not p.exists():
            pytest.skip("manifest.csv not found")
        df     = pd.read_csv(p)
        splits = set(df["split"].unique())
        assert splits.issubset({"train", "val", "test"}), \
            f"Unexpected splits: {splits - {'train', 'val', 'test'}}"

    def test_no_data_leakage_across_splits(self):
        """Critical: same subject must appear in exactly one split."""
        p = DATA_RAW / "manifest.csv"
        if not p.exists():
            pytest.skip("manifest.csv not found")
        df = pd.read_csv(p)
        for sid in df["subject_id"]:
            count = len(df[df["subject_id"] == sid])
            assert count == 1, \
                f"DATA LEAKAGE: {sid} appears in {count} splits"

    def test_train_is_largest_split(self):
        p = DATA_RAW / "manifest.csv"
        if not p.exists():
            pytest.skip("manifest.csv not found")
        df     = pd.read_csv(p)
        counts = df["split"].value_counts()
        assert counts.get("train", 0) >= counts.get("val", 0)
        assert counts.get("train", 0) >= counts.get("test", 0)
