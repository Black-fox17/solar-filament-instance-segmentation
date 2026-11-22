"""Grayscale-adapted Mask R-CNN (ResNet-50 FPN) builder and checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from solar_filament.config import ModelConfig


def adapt_conv1_to_grayscale(conv1: nn.Conv2d, average_weights: bool = True) -> nn.Conv2d:
    """Convert a 3-channel RGB stem Conv2d to a 1-channel grayscale Conv2d.

    When ``average_weights`` is True (used during COCO-pretrained initialization),
    weights are initialized as the channel-wise mean of the original RGB weights.
    """
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        dilation=conv1.dilation,
        groups=conv1.groups,
        bias=conv1.bias is not None,
    )
    if average_weights:
        with torch.no_grad():
            new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
            if conv1.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv1.bias)
    return new_conv


def build_mask_rcnn(
    config: ModelConfig | None = None,
    pretrained: bool | None = None,
) -> nn.Module:
    """Build the 1-channel grayscale Mask R-CNN ResNet-50 FPN model.

    Matches Cells 39, 44, and 47 of the research notebook:
    - Backbone: ResNet-50 FPN (COCO-pretrained when ``pretrained=True``)
    - Input stem: 1-channel Conv2d (RGB weights averaged when pretrained)
    - Box & mask heads: 2 classes (0 = background, 1 = filament)
    - Normalization: ``image_mean=[0.326]``, ``image_std=[0.217]``
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    cfg = config or ModelConfig()
    use_pretrained = cfg.pretrained_coco if pretrained is None else pretrained

    if use_pretrained:
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            box_detections_per_img=cfg.box_detections_per_img,
        )
    else:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            box_detections_per_img=cfg.box_detections_per_img,
        )

    model.backbone.body.conv1 = adapt_conv1_to_grayscale(
        model.backbone.body.conv1,
        average_weights=use_pretrained,
    )

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, cfg.num_classes)

    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, cfg.num_classes)

    model.transform.image_mean = list(cfg.image_mean)
    model.transform.image_std = list(cfg.image_std)

    return model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a model ``state_dict`` from a raw or wrapped training checkpoint."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    available = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
    raise ValueError(f"Could not find model weights in checkpoint. Available: {available}")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove leading ``module.`` prefixes saved by ``torch.nn.DataParallel``."""
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights from disk into ``model`` and return checkpoint metadata."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Train a model first (`python scripts/train.py`) or place a trained checkpoint there."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)

    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        for key in ("epoch", "best_val_loss", "patience_counter"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    return metadata
