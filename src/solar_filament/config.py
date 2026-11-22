"""Typed configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    dataset_root: Path = Path("data/MAGFiLO_1.0_Kaggle_2026")
    train_images_dir: Path = Path("data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    train_annotations_json: Path = Path(
        "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    test_images_dir: Path = Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    checkpoint_dir: Path = Path("checkpoints")
    best_checkpoint: Path = Path("checkpoints/best_model.pth")
    submission_csv: Path = Path("outputs/submission.csv")


@dataclass
class DataConfig:
    image_size: int = 2048
    val_size: float = 0.15
    split_seed: int = 42
    cache_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class PatchConfig:
    patch_size: int = 768
    overlap: int = 192

    @property
    def stride(self) -> int:
        stride = self.patch_size - self.overlap
        if stride <= 0:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly smaller than patch_size ({self.patch_size})"
            )
        return stride


@dataclass
class ModelConfig:
    backbone: str = "resnet50_fpn"
    pretrained_coco: bool = True
    num_classes: int = 2
    min_size: int = 768
    max_size: int = 768
    box_detections_per_img: int = 100
    image_mean: list[float] = field(default_factory=lambda: [0.326])
    image_std: list[float] = field(default_factory=lambda: [0.217])


@dataclass
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    grad_clip_norm: float = 1.0
    patience: int = 4
    min_delta: float = 0.001
    use_amp: bool = False
    max_skipped_batch_ratio: float = 0.01
    log_interval: int = 100


@dataclass
class InferenceConfig:
    batch_size: int = 4
    score_threshold: float = 0.50
    mask_threshold: float = 0.50
    min_mask_area: int = 20
    bbox_iou_threshold: float = 0.30
    bbox_containment_threshold: float = 0.50


@dataclass
class ProjectConfig:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _validate_section_keys(section_name: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in config section '{section_name}': {sorted(unknown)}")


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a ProjectConfig from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping at top level, got {type(raw).__name__}")

    top_allowed = {"seed", "device", "paths", "data", "patches", "model", "training", "inference"}
    _validate_section_keys("root", raw, top_allowed)

    paths_raw = raw.get("paths", {}) or {}
    _validate_section_keys("paths", paths_raw, set(PathsConfig.__dataclass_fields__.keys()))
    paths_cfg = PathsConfig(**{k: Path(v) for k, v in paths_raw.items()})

    data_raw = raw.get("data", {}) or {}
    _validate_section_keys("data", data_raw, set(DataConfig.__dataclass_fields__.keys()))
    data_cfg = DataConfig(**data_raw)

    patches_raw = raw.get("patches", {}) or {}
    _validate_section_keys("patches", patches_raw, set(PatchConfig.__dataclass_fields__.keys()))
    patches_cfg = PatchConfig(**patches_raw)
    _ = patches_cfg.stride  # Validate stride > 0

    model_raw = raw.get("model", {}) or {}
    _validate_section_keys("model", model_raw, set(ModelConfig.__dataclass_fields__.keys()))
    model_cfg = ModelConfig(**model_raw)

    training_raw = raw.get("training", {}) or {}
    _validate_section_keys("training", training_raw, set(TrainingConfig.__dataclass_fields__.keys()))
    training_cfg = TrainingConfig(**training_raw)

    inference_raw = raw.get("inference", {}) or {}
    _validate_section_keys(
        "inference", inference_raw, set(InferenceConfig.__dataclass_fields__.keys())
    )
    inference_cfg = InferenceConfig(**inference_raw)

    return ProjectConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        paths=paths_cfg,
        data=data_cfg,
        patches=patches_cfg,
        model=model_cfg,
        training=training_cfg,
        inference=inference_cfg,
    )
