"""Tiled inference and full-resolution instance reconstruction."""

from solar_filament.inference.postprocess import (
    PatchCandidate,
    bbox_iou_and_containment,
    extract_patch_predictions,
    make_masks_exclusive,
    merge_patch_candidates,
)
from solar_filament.inference.predictor import (
    extract_image_patches,
    predict_image,
    run_directory_inference,
)

__all__ = [
    "PatchCandidate",
    "bbox_iou_and_containment",
    "extract_image_patches",
    "extract_patch_predictions",
    "make_masks_exclusive",
    "merge_patch_candidates",
    "predict_image",
    "run_directory_inference",
]
