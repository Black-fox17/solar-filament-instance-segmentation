"""Unified command-line interface for dataset auditing, training, inference, and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from solar_filament.config import ProjectConfig, load_config
from solar_filament.data.annotations import (
    AnnotationIndex,
    audit_annotations,
    group_records_by_filename,
    split_by_filename,
)
from solar_filament.data.dataset import SolarFilamentDataset, detection_collate_fn
from solar_filament.data.patches import audit_patch_coverage
from solar_filament.inference.predictor import run_directory_inference
from solar_filament.models.mask_rcnn import build_mask_rcnn, load_model_checkpoint
from solar_filament.submission.validator import (
    build_submission_rows,
    validate_submission_csv,
    write_submission_csv,
)
from solar_filament.training.trainer import train_model
from solar_filament.utils.device import resolve_device
from solar_filament.utils.logging import get_logger
from solar_filament.utils.seed import seed_everything

logger = get_logger(__name__)


def _apply_common_overrides(cfg: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "dataset_root", None):
        root = Path(args.dataset_root)
        cfg.paths.dataset_root = root
        cfg.paths.train_images_dir = root / "train" / "train_images"
        cfg.paths.train_annotations_json = (
            root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        )
        cfg.paths.test_images_dir = root / "test" / "test_images"
    return cfg


def run_audit(args: argparse.Namespace) -> None:
    cfg = _apply_common_overrides(load_config(args.config), args)
    ann_path = Path(args.annotations) if args.annotations else cfg.paths.train_annotations_json
    logger.info("Loading annotations from %s", ann_path)
    index = AnnotationIndex.from_json(ann_path)

    ann_report = audit_annotations(
        index,
        check_mask_overlaps=not args.skip_mask_overlaps,
    )
    patch_report = audit_patch_coverage(
        index,
        image_size=cfg.data.image_size,
        patch_size=cfg.patches.patch_size,
        overlap=cfg.patches.overlap,
    )

    combined = {"annotations": ann_report, "patches": patch_report}
    output_text = json.dumps(combined, indent=2)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        logger.info("Saved audit report to %s", out_path)
    else:
        print(output_text)


def run_train(args: argparse.Namespace) -> None:
    cfg = _apply_common_overrides(load_config(args.config), args)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.checkpoint_dir:
        cfg.paths.checkpoint_dir = Path(args.checkpoint_dir)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    index = AnnotationIndex.from_json(cfg.paths.train_annotations_json)
    filename_to_ids = group_records_by_filename(index)
    train_filenames, val_filenames = split_by_filename(
        filename_to_ids,
        val_size=cfg.data.val_size,
        random_state=cfg.data.split_seed,
    )
    logger.info(
        "Physical filename split | train=%d images | val=%d images",
        len(train_filenames),
        len(val_filenames),
    )

    train_dataset = SolarFilamentDataset(
        filenames=train_filenames,
        filename_to_ids=filename_to_ids,
        annotation_index=index,
        image_dir=cfg.paths.train_images_dir,
        image_size=cfg.data.image_size,
        patch_size=cfg.patches.patch_size,
        overlap=cfg.patches.overlap,
        training=True,
        cache_size=cfg.data.cache_size,
        seed=cfg.seed,
    )
    val_dataset = SolarFilamentDataset(
        filenames=val_filenames,
        filename_to_ids=filename_to_ids,
        annotation_index=index,
        image_dir=cfg.paths.train_images_dir,
        image_size=cfg.data.image_size,
        patch_size=cfg.patches.patch_size,
        overlap=cfg.patches.overlap,
        training=False,
        cache_size=cfg.data.cache_size,
        seed=cfg.seed,
    )

    pin_memory = cfg.data.pin_memory and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )

    model = build_mask_rcnn(cfg.model, pretrained=cfg.model.pretrained_coco)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
    )


def run_predict(args: argparse.Namespace) -> None:
    cfg = _apply_common_overrides(load_config(args.config), args)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.paths.best_checkpoint
    test_dir = Path(args.test_dir) if args.test_dir else cfg.paths.test_images_dir
    output_csv = Path(args.output_csv) if args.output_csv else cfg.paths.submission_csv

    device = resolve_device(cfg.device)
    logger.info("Loading model onto %s from %s", device, checkpoint_path)

    model = build_mask_rcnn(cfg.model, pretrained=False)
    metadata = load_model_checkpoint(model, checkpoint_path, device=device, strict=True)
    if metadata:
        logger.info("Checkpoint metadata: %s", metadata)

    results = run_directory_inference(model=model, test_dir=test_dir, config=cfg, device=device)
    rows = build_submission_rows(results)
    df = write_submission_csv(rows, output_csv)

    raw_total = sum(r.raw_candidate_count for r in results)
    merged_total = sum(r.merged_count for r in results)
    final_total = sum(len(r.masks) for r in results)
    logger.info(
        "Submission written to %s | images=%d | raw=%d | merged=%d | final=%d | csv_rows=%d",
        output_csv,
        len(results),
        raw_total,
        merged_total,
        final_total,
        len(df),
    )

    if not args.skip_validation:
        summary = validate_submission_csv(
            csv_path=output_csv,
            image_size=cfg.data.image_size,
            min_mask_area=cfg.inference.min_mask_area,
            test_dir=test_dir,
        )
        logger.info("Submission validation passed: %s", summary)


def run_validate_submission(args: argparse.Namespace) -> None:
    cfg = _apply_common_overrides(load_config(args.config), args)
    csv_path = Path(args.csv) if args.csv else cfg.paths.submission_csv
    test_dir = Path(args.test_dir) if args.test_dir else None

    summary = validate_submission_csv(
        csv_path=csv_path,
        image_size=cfg.data.image_size,
        min_mask_area=cfg.inference.min_mask_area,
        test_dir=test_dir,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-filament",
        description="Solar Filament Instance Segmentation (MAGFiLO 1.0 / Kaggle 2026) CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Audit COCO annotations and patch coverage.")
    p_audit.add_argument("--config", default="configs/default.yaml")
    p_audit.add_argument("--dataset-root", default=None)
    p_audit.add_argument("--annotations", default=None)
    p_audit.add_argument("--skip-mask-overlaps", action="store_true")
    p_audit.add_argument("--output-json", default=None)
    p_audit.set_defaults(func=run_audit)

    p_train = sub.add_parser("train", help="Train grayscale Mask R-CNN in FP32.")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument("--dataset-root", default=None)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--checkpoint-dir", default=None)
    p_train.set_defaults(func=run_train)

    p_pred = sub.add_parser("predict", help="Run tiled inference and generate submission CSV.")
    p_pred.add_argument("--config", default="configs/default.yaml")
    p_pred.add_argument("--dataset-root", default=None)
    p_pred.add_argument("--device", default=None)
    p_pred.add_argument("--checkpoint", default=None)
    p_pred.add_argument("--test-dir", default=None)
    p_pred.add_argument("--output-csv", default=None)
    p_pred.add_argument("--skip-validation", action="store_true")
    p_pred.set_defaults(func=run_predict)

    p_val = sub.add_parser("validate-submission", help="Validate a competition submission CSV.")
    p_val.add_argument("--config", default="configs/default.yaml")
    p_val.add_argument("--csv", default=None)
    p_val.add_argument("--test-dir", default=None)
    p_val.set_defaults(func=run_validate_submission)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
