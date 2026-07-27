"""
scripts/generate_ct_mask_npy.py
--------------------------------
Convert {sid}_ct_mask.nii.gz -> {sid}_ct_mask.npy for every subject.
Crops/pads to (160, 192, 160) to match the other .npy volumes.
Safe to re-run: skips subjects that already have ct_mask.npy.
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHAPE  = (160, 192, 160)
MANIFEST = ROOT / "data" / "raw" / "manifest_processed.csv"
PROC_DIR = ROOT / "data" / "processed"

df = pd.read_csv(MANIFEST)
converted = skipped = missing = 0

for _, row in df.iterrows():
    sid      = str(row["subject_id"])
    nii_path = PROC_DIR / sid / f"{sid}_ct_mask.nii.gz"
    npy_path = PROC_DIR / sid / f"{sid}_ct_mask.npy"

    if npy_path.exists():
        skipped += 1
        continue

    if not nii_path.exists():
        print(f"[MISSING] {sid}: {nii_path} not found")
        missing += 1
        continue

    arr = nib.load(nii_path).get_fdata().astype("float32")
    arr = (arr > 0).astype("float32")   # binarize

    # Crop / pad to fixed shape
    out = np.zeros(SHAPE, dtype="float32")
    s   = tuple(min(a, b) for a, b in zip(arr.shape, SHAPE))
    out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]

    np.save(npy_path, out)
    converted += 1
    print(f"[OK] {sid} -> ct_mask.npy  (nonzero={out.sum():.0f})")

print(f"\nDone. converted={converted}  already_existed={skipped}  missing_nii={missing}")
