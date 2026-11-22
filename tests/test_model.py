"""Tests for grayscale conv1 adaptation, checkpoint extraction/loading, and model builder."""

import pytest
import torch
import torch.nn as nn

from solar_filament.config import ModelConfig
from solar_filament.models.mask_rcnn import (
    adapt_conv1_to_grayscale,
    build_mask_rcnn,
    extract_state_dict,
    load_model_checkpoint,
    strip_module_prefix,
)


def test_adapt_conv1_to_grayscale_averages_rgb_weights():
    rgb_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        rgb_conv.weight[:, 0].fill_(1.0)
        rgb_conv.weight[:, 1].fill_(2.0)
        rgb_conv.weight[:, 2].fill_(6.0)

    gray_conv = adapt_conv1_to_grayscale(rgb_conv, average_weights=True)
    assert gray_conv.in_channels == 1
    assert gray_conv.out_channels == 64
    assert gray_conv.kernel_size == (7, 7)
    assert gray_conv.stride == (2, 2)
    assert gray_conv.padding == (3, 3)
    assert gray_conv.bias is None
    # Mean of (1, 2, 6) == 3.0
    assert torch.allclose(gray_conv.weight, torch.full((64, 1, 7, 7), 3.0))


def test_extract_state_dict_and_strip_module_prefix():
    raw = {"module.layer.weight": torch.ones(2, 2)}
    wrapped = {"epoch": 1, "best_val_loss": 0.41655, "model_state_dict": raw}

    extracted = extract_state_dict(wrapped)
    assert set(extracted.keys()) == {"module.layer.weight"}

    stripped = strip_module_prefix(extracted)
    assert set(stripped.keys()) == {"layer.weight"}

    assert extract_state_dict(raw) == raw

    with pytest.raises(ValueError, match="Could not find model weights"):
        extract_state_dict({"epoch": 1, "notes": "no tensors"})


def test_load_model_checkpoint_roundtrip(tmp_path):
    model_a = nn.Linear(4, 2)
    with torch.no_grad():
        model_a.weight.fill_(0.25)
        model_a.bias.fill_(0.5)

    ckpt_path = tmp_path / "best_model.pth"
    torch.save(
        {
            "epoch": 1,
            "best_val_loss": 0.41655,
            "model_state_dict": model_a.state_dict(),
        },
        ckpt_path,
    )

    model_b = nn.Linear(4, 2)
    meta = load_model_checkpoint(model_b, ckpt_path, device="cpu", strict=True)
    assert meta["epoch"] == 1
    assert meta["best_val_loss"] == pytest.approx(0.41655)
    assert torch.allclose(model_b.weight, model_a.weight)
    assert torch.allclose(model_b.bias, model_a.bias)


def test_build_mask_rcnn_architecture():
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torchvision unavailable in this environment: {exc}")

    cfg = ModelConfig(pretrained_coco=False, num_classes=2, min_size=768, max_size=768)
    model = build_mask_rcnn(cfg, pretrained=False)
    assert model.backbone.body.conv1.in_channels == 1
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
    assert model.transform.image_mean == [0.326]
    assert model.transform.image_std == [0.217]
