"""Full-resolution tiled inference pipeline for 2048x2048 solar images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

from solar_filament.config import ProjectConfig
from solar_filament.data.patches import generate_patch_coords
from solar_filament.inference.postprocess import (
    PatchCandidate,
    extract_patch_predictions,
    make_masks_exclusive,
    merge_patch_candidates,
)


@dataclass
class ImagePredictionResult:
    """Inference output and diagnostics for one full-resolution image."""

    image_path: Path
    masks: list[np.ndarray]
    raw_candidate_count: int
    merged_count: int


def extract_image_patches(
    image: np.ndarray,
    image_size: int = 2048,
    patch_size: int = 768,
    overlap: int = 192,
) -> list[tuple[torch.Tensor, int, int]]:
    """Slice a 2D uint8 grayscale image into normalized ``[1, patch_size, patch_size]`` tensors."""
    if image.shape != (image_size, image_size):
        raise ValueError(f"Expected image shape ({image_size}, {image_size}), got {image.shape}")

    coords = generate_patch_coords(image_size=image_size, patch_size=patch_size, overlap=overlap)
    patches: list[tuple[torch.Tensor, int, int]] = []
    for px, py in coords:
        crop = image[py : py + patch_size, px : px + patch_size]
        tensor = torch.from_numpy(crop.copy()).float().unsqueeze(0) / 255.0
        patches.append((tensor, px, py))
    return patches


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    config: ProjectConfig,
    device: torch.device,
) -> ImagePredictionResult:
    """Run tiled Mask R-CNN inference and full-resolution reconstruction on one image."""
    path = Path(image_path)
    with Image.open(path) as img:
        image = np.asarray(img.convert("L"), dtype=np.uint8)

    image_size = config.data.image_size
    patch_size = config.patches.patch_size
    overlap = config.patches.overlap
    icfg = config.inference

    patches = extract_image_patches(
        image=image,
        image_size=image_size,
        patch_size=patch_size,
        overlap=overlap,
    )

    candidates: list[PatchCandidate] = []
    model.eval()

    for start in range(0, len(patches), icfg.batch_size):
        batch_items = patches[start : start + icfg.batch_size]
        batch_tensors = [item[0].to(device) for item in batch_items]

        with torch.inference_mode():
            predictions = model(batch_tensors)

        for prediction, (_, patch_x, patch_y) in zip(predictions, batch_items):
            candidates.extend(
                extract_patch_predictions(
                    prediction=prediction,
                    patch_x=patch_x,
                    patch_y=patch_y,
                    score_threshold=icfg.score_threshold,
                    mask_threshold=icfg.mask_threshold,
                    min_mask_area=icfg.min_mask_area,
                )
            )

    merged_masks, merged_scores = merge_patch_candidates(
        candidates=candidates,
        image_size=image_size,
        bbox_iou_threshold=icfg.bbox_iou_threshold,
        bbox_containment_threshold=icfg.bbox_containment_threshold,
        min_mask_area=icfg.min_mask_area,
    )

    final_masks = make_masks_exclusive(
        masks=merged_masks,
        scores=merged_scores,
        image_size=image_size,
        min_mask_area=icfg.min_mask_area,
    )

    return ImagePredictionResult(
        image_path=path,
        masks=final_masks,
        raw_candidate_count=len(candidates),
        merged_count=len(merged_masks),
    )


def list_test_images(test_dir: str | Path) -> list[Path]:
    """Return sorted test image paths (``.jpeg``, ``.jpg``, ``.png``)."""
    directory = Path(test_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Test image directory not found: {directory}")

    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )
    if not paths:
        raise FileNotFoundError(f"No test images (.jpeg/.jpg/.png) found in {directory}")
    return paths


def run_directory_inference(
    model: nn.Module,
    test_dir: str | Path,
    config: ProjectConfig,
    device: torch.device,
) -> list[ImagePredictionResult]:
    """Run full-resolution inference across every image in ``test_dir``."""
    test_paths = list_test_images(test_dir)
    results: list[ImagePredictionResult] = []
    for image_path in tqdm(test_paths, desc="Generating predictions"):
        results.append(predict_image(model=model, image_path=image_path, config=config, device=device))
    return results
