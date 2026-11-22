"""Tests for configuration loading, validation, and device utilities."""

from pathlib import Path

import pytest

from solar_filament.config import PatchConfig, load_config
from solar_filament.utils.device import resolve_device
from solar_filament.utils.seed import seed_everything


def test_load_default_config():
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.seed == 42
    assert cfg.data.image_size == 2048
    assert cfg.data.val_size == pytest.approx(0.15)
    assert cfg.patches.patch_size == 768
    assert cfg.patches.overlap == 192
    assert cfg.patches.stride == 576
    assert cfg.model.num_classes == 2
    assert cfg.model.image_mean == [0.326]
    assert cfg.model.image_std == [0.217]
    assert cfg.training.use_amp is False
    assert cfg.training.grad_clip_norm == pytest.approx(1.0)
    assert cfg.inference.score_threshold == pytest.approx(0.50)
    assert cfg.inference.min_mask_area == 20


def test_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        _ = PatchConfig(patch_size=512, overlap=512).stride


def test_unknown_config_key_raises(tmp_path):
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("training:\n  nonexistent_param: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(bad_cfg)


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    auto = resolve_device("auto")
    assert auto.type in {"cuda", "mps", "cpu"}


def test_seed_everything_repeatable():
    import random
    import numpy as np
    import torch

    seed_everything(123)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    seed_everything(123)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert a == b
