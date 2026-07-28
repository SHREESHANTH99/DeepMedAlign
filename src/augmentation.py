"""
src/augmentation.py
--------------------
Data augmentation transforms for MRI-CT registration training.

Design rules
------------
* Applied ONLY to the train split — val/test use identity transforms.
* All augmentations are spatially consistent: the same random transform
  is applied to MRI, CT, and mask together.
* Medical images are already in [0, 1] after preprocessing.
* Never flip along Z (axial) — brain anatomy is not symmetric top/bottom.

Usage
-----
    from src.augmentation import get_train_transforms, get_val_transforms

    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    # Both are callable: transform(sample_dict) -> sample_dict
"""

from __future__ import annotations

import random
import numpy as np
import torch
from typing import Callable, Dict


# Type alias
Sample = Dict[str, object]


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

class RandomFlip:
    """Randomly flip MRI, CT, and mask along the left-right axis (axis=1).

    Brain anatomy is symmetric left/right so this is a valid augmentation.
    Never flip along Z (axial, axis=0) or Y (coronal, axis=2).

    Parameters
    ----------
    p : probability of flipping (default 0.5)
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            for key in ("mr", "ct", "mask"):
                if key in sample and torch.is_tensor(sample[key]):
                    # tensor shape: (1, D, H, W) — flip along W (dim=3)
                    sample[key] = torch.flip(sample[key], dims=[3])
        return sample


class RandomIntensityJitter:
    """Multiply MRI intensity by a random scale factor in [1-delta, 1+delta].

    Applied to MRI only — CT HU values are physically meaningful.

    Parameters
    ----------
    delta : maximum relative intensity shift (default 0.10 = ±10%)
    p     : probability of applying (default 0.5)
    """

    def __init__(self, delta: float = 0.10, p: float = 0.5) -> None:
        self.delta = delta
        self.p     = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            scale        = 1.0 + random.uniform(-self.delta, self.delta)
            sample["mr"] = torch.clamp(sample["mr"] * scale, 0.0, 1.0)
        return sample


class RandomGaussianNoise:
    """Add zero-mean Gaussian noise to MRI.

    Applied to MRI only.

    Parameters
    ----------
    sigma : noise standard deviation (default 0.01)
    p     : probability of applying (default 0.5)
    """

    def __init__(self, sigma: float = 0.01, p: float = 0.5) -> None:
        self.sigma = sigma
        self.p     = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            noise        = torch.randn_like(sample["mr"]) * self.sigma
            sample["mr"] = torch.clamp(sample["mr"] + noise, 0.0, 1.0)
        return sample


class RandomAffinePerturbation:
    """Apply a small random affine perturbation to all volumes.

    Uses scipy for the transform so the mask stays consistent.

    Parameters
    ----------
    max_angle_deg : maximum rotation per axis in degrees (default 5°)
    max_shift_px  : maximum translation in voxels (default 5)
    p             : probability of applying (default 0.3)
    """

    def __init__(
        self,
        max_angle_deg: float = 5.0,
        max_shift_px:  int   = 5,
        p:             float = 0.3,
    ) -> None:
        self.max_angle = max_angle_deg
        self.max_shift = max_shift_px
        self.p         = p

    def _random_angle(self) -> float:
        return random.uniform(-self.max_angle, self.max_angle)

    def _random_shift(self) -> int:
        return random.randint(-self.max_shift, self.max_shift)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.p:
            return sample

        try:
            from scipy.ndimage import rotate, shift

            # Same random params for all volumes
            angle  = self._random_angle()
            shifts = [self._random_shift() for _ in range(3)]
            axes   = (0, 1)   # rotate in axial-coronal plane

            for key in ("mr", "ct", "mask"):
                if key not in sample or not torch.is_tensor(sample[key]):
                    continue
                arr = sample[key].numpy()[0]   # (D, H, W)
                order = 0 if key == "mask" else 1  # NN for mask, linear for imgs
                arr = rotate(arr, angle, axes=axes,
                             reshape=False, order=order, mode="constant")
                arr = shift(arr, shifts, order=order, mode="constant")
                arr = np.clip(arr, 0.0, 1.0).astype("float32")
                sample[key] = torch.from_numpy(arr).unsqueeze(0)

        except ImportError:
            pass   # scipy not available — skip silently

        return sample


class RandomElasticDeformation:
    """Apply random elastic deformation to MRI, CT, and mask consistently.

    Generates a smooth random displacement field using Gaussian-filtered
    noise. This forces the AI to learn to un-warp severely deformed brains,
    making it much better at aligning real-world variations.

    Parameters
    ----------
    alpha : controls deformation strength (default 300.0)
    sigma : controls deformation smoothness (default 20.0, higher = smoother)
    p     : probability of applying (default 0.3)
    """

    def __init__(self, alpha: float = 300.0, sigma: float = 20.0, p: float = 0.3) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p     = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.p:
            return sample

        try:
            from scipy.ndimage import gaussian_filter, map_coordinates

            # Use MRI shape to build the displacement field
            ref = sample["mr"].numpy()[0]   # (D, H, W)
            shape = ref.shape

            # One shared random displacement field for all axes
            dx = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
            dy = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
            dz = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha

            z, y, x = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
            indices = (
                np.clip(z + dz, 0, shape[0] - 1).ravel(),
                np.clip(y + dy, 0, shape[1] - 1).ravel(),
                np.clip(x + dx, 0, shape[2] - 1).ravel(),
            )

            for key in ("mr", "ct", "mask"):
                if key not in sample or not torch.is_tensor(sample[key]):
                    continue
                arr   = sample[key].numpy()[0]
                order = 0 if key == "mask" else 1
                warped = map_coordinates(arr, indices, order=order, mode="reflect")
                warped = np.clip(warped.reshape(shape), 0.0, 1.0).astype("float32")
                sample[key] = torch.from_numpy(warped).unsqueeze(0)

        except ImportError:
            pass   # scipy not available — skip silently

        return sample


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = ["Compose("]
        for t in self.transforms:
            lines.append(f"  {t.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_train_transforms(
    flip:    bool = True,
    jitter:  bool = True,
    noise:   bool = True,
    affine:  bool = False,
    elastic: bool = False,
) -> Compose:
    """Return the full augmentation pipeline for the train split.

    Parameters
    ----------
    flip    : include random left-right flip
    jitter  : include random MRI intensity jitter
    noise   : include random Gaussian noise on MRI
    affine  : include random affine perturbation (needs scipy)
    elastic : include random elastic deformation (needs scipy)
    """
    transforms = []
    if flip:
        transforms.append(RandomFlip(p=0.5))
    if jitter:
        transforms.append(RandomIntensityJitter(delta=0.10, p=0.5))
    if noise:
        transforms.append(RandomGaussianNoise(sigma=0.01, p=0.5))
    if affine:
        transforms.append(RandomAffinePerturbation(
            max_angle_deg=5.0, max_shift_px=5, p=0.3))
    if elastic:
        transforms.append(RandomElasticDeformation(
            alpha=300.0, sigma=20.0, p=0.3))
    return Compose(transforms)


def get_val_transforms() -> Compose:
    """Return the identity transform for val/test splits (no augmentation)."""
    return Compose([])

