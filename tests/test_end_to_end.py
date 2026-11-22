"""End-to-end synthetic pipeline integration tests (no Kaggle dataset required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from solar_filament.cli import main as cli_main
from solar_filament.config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    PatchConfig,
    PathsConfig,
    ProjectConfig,
    TrainingConfig,
)
from solar_filament.data import (
    AnnotationIndex,
    SolarFilamentDataset,
    detection_collate_fn,
    group_records_by_filename,
    split_by_filename,
)
from solar_filament.inference import run_directory_inference
from solar_filament.models import load_model_checkpoint
from solar_filament.submission import (
    build_submission_rows,
    validate_submission_csv,
    write_submission_csv,
)
from solar_filament.training import train_model


class _DummyDetectionModel(nn.Module):
    """Lightweight stand-in matching Torchvision Mask R-CNN's train/eval contract.

    Includes a real BatchNorm2d layer so we verify that validate_one_epoch keeps
    BatchNorm running statistics frozen while model.train() is active.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, images: list[torch.Tensor], targets=None):
        batch = torch.stack(images)  # [B, 1, H, W]
        feat = self.bn(self.conv(batch))

        if self.training:
            base = (feat**2).mean()
            return {
                "loss_classifier": base * 0.2 + 0.1,
                "loss_box_reg": base * 0.2 + 0.05,
                "loss_mask": base * 0.4 + 0.2,
                "loss_objectness": base * 0.1 + 0.02,
                "loss_rpn_box_reg": base * 0.1 + 0.01,
            }

        # Eval mode: emit one consistent synthetic filament mask per patch
        preds = []
        _, _, h, w = batch.shape
        for _ in images:
            mask = torch.zeros((1, 1, h, w), dtype=torch.float32, device=batch.device)
            mask[0, 0, 8:20, 8:20] = 0.92  # 144 px >= min_mask_area (20)
            preds.append(
                {
                    "boxes": torch.tensor([[8.0, 8.0, 20.0, 20.0]], device=batch.device),
                    "labels": torch.tensor([1], dtype=torch.int64, device=batch.device),
                    "scores": torch.tensor([0.91], dtype=torch.float32, device=batch.device),
                    "masks": mask,
                }
            )
        return preds


def _toy_encode(mask: np.ndarray) -> str:
    ys, xs = np.where(mask > 0)
    return ";".join(f"{int(y)},{int(x)}" for y, x in zip(ys, xs))


def _toy_decode(encoded: str, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for pair in encoded.split(";"):
        if pair:
            y_str, x_str = pair.split(",")
            mask[int(y_str), int(x_str)] = 1
    return mask


def _build_synthetic_workspace(tmp_path: Path) -> tuple[Path, ProjectConfig]:
    root = tmp_path / "MAGFiLO_1.0_Kaggle_2026"
    train_dir = root / "train" / "train_images"
    test_dir = root / "test" / "test_images"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    train_files = [f"sun_train_{i}.jpeg" for i in range(4)]
    test_files = [f"sun_test_{i}.jpeg" for i in range(2)]

    for fname in train_files:
        Image.fromarray(rng.integers(20, 220, size=(64, 64), dtype=np.uint8), mode="L").save(
            train_dir / fname
        )
    for fname in test_files:
        Image.fromarray(rng.integers(20, 220, size=(64, 64), dtype=np.uint8), mode="L").save(
            test_dir / fname
        )

    # Multi-annotator COCO JSON (sun_train_0 has 2 annotator passes)
    images = [
        {"id": "01-sun0", "file_name": "sun_train_0.jpeg", "height": 64, "width": 64},
        {"id": "02-sun0", "file_name": "sun_train_0.jpeg", "height": 64, "width": 64},
        {"id": "01-sun1", "file_name": "sun_train_1.jpeg", "height": 64, "width": 64},
        {"id": "01-sun2", "file_name": "sun_train_2.jpeg", "height": 64, "width": 64},
        {"id": "01-sun3", "file_name": "sun_train_3.jpeg", "height": 64, "width": 64},
    ]
    annotations = [
        {
            "id": f"ann-{idx}",
            "image_id": img["id"],
            "category_id": 1 + (idx % 3),
            "bbox": [8.0, 8.0, 14.0, 14.0],
            "area": 196.0,
            "iscrowd": 0,
            "segmentation": [[8, 8, 22, 8, 22, 22, 8, 22]],
        }
        for idx, img in enumerate(images)
    ]
    ann_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ann_path.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "Left", "supercategory": "filament"},
                    {"id": 2, "name": "Right", "supercategory": "filament"},
                    {"id": 3, "name": "Unidentifiable", "supercategory": "filament"},
                    {"id": 4, "name": "Ambiguous", "supercategory": "filament"},
                ],
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )

    ckpt_dir = tmp_path / "checkpoints"
    cfg = ProjectConfig(
        seed=42,
        device="cpu",
        paths=PathsConfig(
            dataset_root=root,
            train_images_dir=train_dir,
            train_annotations_json=ann_path,
            test_images_dir=test_dir,
            checkpoint_dir=ckpt_dir,
            best_checkpoint=ckpt_dir / "best_model.pth",
            submission_csv=tmp_path / "outputs" / "submission.csv",
        ),
        data=DataConfig(image_size=64, val_size=0.25, split_seed=42, cache_size=2, num_workers=0),
        patches=PatchConfig(patch_size=32, overlap=8),
        model=ModelConfig(pretrained_coco=False, min_size=32, max_size=32),
        training=TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            patience=4,
            use_amp=False,
            log_interval=0,
        ),
        inference=InferenceConfig(
            batch_size=2,
            score_threshold=0.50,
            mask_threshold=0.50,
            min_mask_area=20,
            bbox_iou_threshold=0.30,
            bbox_containment_threshold=0.50,
        ),
    )
    return root, cfg


def test_end_to_end_audit_train_checkpoint_infer_validate(tmp_path):
    root, cfg = _build_synthetic_workspace(tmp_path)

    # 1. Run CLI audit end-to-end
    audit_json = tmp_path / "audit.json"
    cli_main(
        [
            "audit",
            "--config",
            "configs/default.yaml",
            "--dataset-root",
            str(root),
            "--output-json",
            str(audit_json),
        ]
    )
    audit_data = json.loads(audit_json.read_text(encoding="utf-8"))
    assert audit_data["annotations"]["total_records"] == 5
    assert audit_data["annotations"]["unique_filenames"] == 4

    # 2. Build leakage-free train/val datasets & loaders
    index = AnnotationIndex.from_json(cfg.paths.train_annotations_json)
    grouped = group_records_by_filename(index)
    train_files, val_files = split_by_filename(grouped, val_size=cfg.data.val_size, random_state=42)

    train_ds = SolarFilamentDataset(
        filenames=train_files,
        filename_to_ids=grouped,
        annotation_index=index,
        image_dir=cfg.paths.train_images_dir,
        image_size=cfg.data.image_size,
        patch_size=cfg.patches.patch_size,
        overlap=cfg.patches.overlap,
        training=True,
        cache_size=2,
    )
    val_ds = SolarFilamentDataset(
        filenames=val_files,
        filename_to_ids=grouped,
        annotation_index=index,
        image_dir=cfg.paths.train_images_dir,
        image_size=cfg.data.image_size,
        patch_size=cfg.patches.patch_size,
        overlap=cfg.patches.overlap,
        training=False,
        cache_size=2,
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)

    # 3. Train for 2 epochs, saving best_model.pth, latest_model.pth, training_history.json
    model = _DummyDetectionModel()
    history = train_model(model, train_loader, val_loader, cfg, device=torch.device("cpu"))
    assert len(history) == 2
    assert cfg.paths.best_checkpoint.exists()
    assert (cfg.paths.checkpoint_dir / "latest_model.pth").exists()
    assert (cfg.paths.checkpoint_dir / "training_history.json").exists()
    assert all(h["skipped_train_batches"] == 0 for h in history)

    # 4. Reload checkpoint into a fresh model instance and run full tiled inference
    reloaded = _DummyDetectionModel()
    meta = load_model_checkpoint(reloaded, cfg.paths.best_checkpoint, device="cpu", strict=True)
    assert meta["epoch"] in {1, 2}

    results = run_directory_inference(
        model=reloaded,
        test_dir=cfg.paths.test_images_dir,
        config=cfg,
        device=torch.device("cpu"),
    )
    assert len(results) == 2
    assert all(len(r.masks) > 0 for r in results)

    # 5. Build, write, and strictly validate submission CSV
    rows = build_submission_rows(results, encoder=_toy_encode)
    write_submission_csv(rows, cfg.paths.submission_csv)
    summary = validate_submission_csv(
        csv_path=cfg.paths.submission_csv,
        image_size=cfg.data.image_size,
        min_mask_area=cfg.inference.min_mask_area,
        test_dir=cfg.paths.test_images_dir,
        decoder=_toy_decode,
        show_progress=False,
    )
    assert summary["overlapping_instances"] == 0
    assert summary["unique_predicted_images"] == 2
    assert summary["images_without_predictions"] == 0
