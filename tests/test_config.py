from pathlib import Path

from src.config import (
    DATA_RAW,
    DATA_PROC,
    DOCS,
    FIXED_SHAPE,
    CT_HU_CLIP,
    CT_HU_BRAIN,
    LOGS,
    MODELS_DIR,
    RESULTS,
    ROOT,
    SCRIPTS,
    SYNTHRAD,
    VOXEL_SPACING,
)


def test_all_dirs_created():
    for directory in (
        DATA_RAW,
        DATA_PROC,
        MODELS_DIR,
        RESULTS,
        LOGS,
        DOCS,
        SCRIPTS,
    ):
        assert directory.exists()
        assert directory.is_dir()


def test_constants_correct_types():
    assert isinstance(ROOT, Path)
    assert isinstance(SYNTHRAD, Path)
    assert isinstance(VOXEL_SPACING, tuple)
    assert isinstance(FIXED_SHAPE, tuple)
    assert isinstance(CT_HU_CLIP, tuple)
    assert isinstance(CT_HU_BRAIN, tuple)


def test_voxel_spacing_values():
    assert len(VOXEL_SPACING) == 3
    assert all(value > 0 for value in VOXEL_SPACING)


def test_fixed_shape_values():
    assert len(FIXED_SHAPE) == 3
    assert all(isinstance(value, int) and value > 0 for value in FIXED_SHAPE)


def test_ct_hu_clip_range():
    low, high = CT_HU_CLIP
    assert low < high


def test_src_package_importable():
    import src  # noqa: F401
