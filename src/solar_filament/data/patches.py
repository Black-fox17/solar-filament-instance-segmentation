"""Sliding-window patch geometry, mask clipping, and coverage auditing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from solar_filament.data.annotations import AnnotationIndex, group_records_by_filename


def compute_patch_positions(image_size: int, patch_size: int, overlap: int) -> list[int]:
    """Compute 1D patch start coordinates with right/bottom edge anchoring.

    For the default configuration (``image_size=2048``, ``patch_size=768``,
    ``overlap=192``), this yields ``[0, 576, 1152, 1280]`` (stride 576, 4x4 = 16 patches).
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if image_size < patch_size:
        raise ValueError(
            f"image_size ({image_size}) must be >= patch_size ({patch_size})"
        )
    if overlap < 0 or overlap >= patch_size:
        raise ValueError(
            f"overlap ({overlap}) must satisfy 0 <= overlap < patch_size ({patch_size})"
        )

    stride = patch_size - overlap
    final_start = image_size - patch_size
    positions = list(range(0, final_start + 1, stride))
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def generate_patch_coords(
    image_size: int = 2048,
    patch_size: int = 768,
    overlap: int = 192,
) -> list[tuple[int, int]]:
    """Return row-major ``(x, y)`` top-left coordinates for all patches in an image."""
    positions = compute_patch_positions(image_size, patch_size, overlap)
    return [(x, y) for y in positions for x in positions]


def extract_patch_targets(
    full_masks: list[np.ndarray],
    px: int,
    py: int,
    patch_size: int,
) -> tuple[list[np.ndarray], list[list[float]]]:
    """Clip full-image instance masks to a patch and compute tight ``[x1, y1, x2, y2]`` boxes.

    Matches the final dataset logic (Cell 34): every instance with at least one
    pixel inside the patch is kept, with its bounding box recomputed tightly
    around the clipped mask (exclusive ``x2``, ``y2``).
    """
    clipped_masks: list[np.ndarray] = []
    boxes: list[list[float]] = []

    for full_mask in full_masks:
        patch_mask = full_mask[py : py + patch_size, px : px + patch_size]
        if not np.any(patch_mask):
            continue

        ys, xs = np.where(patch_mask > 0)
        if len(xs) == 0:
            continue

        clipped_masks.append((patch_mask > 0).astype(np.uint8))
        boxes.append(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
        )

    return clipped_masks, boxes


def audit_patch_coverage(
    index: AnnotationIndex,
    image_size: int = 2048,
    patch_size: int = 768,
    overlap: int = 192,
) -> dict[str, Any]:
    """Audit full-containment rates and strict-filtering losses for a patch grid.

    Reproduces the patch geometry experiments from Cells 18, 30, 32, and 33.
    """
    coords = generate_patch_coords(image_size, patch_size, overlap)
    patches = [(x, y, x + patch_size, y + patch_size) for x, y in coords]

    # 1. Per-instance containment across all patches (Cells 18 & 30)
    containment_stats: dict[str, int] = defaultdict(int)
    for img_id in index.get_img_ids():
        for ann in index.get_anns_for_image(img_id):
            x, y, w, h = ann["bbox"]
            x2, y2 = x + w, y + h

            fully_contained = False
            intersects = False
            for px1, py1, px2, py2 in patches:
                inter_w = max(0.0, min(x2, px2) - max(x, px1))
                inter_h = max(0.0, min(y2, py2) - max(y, py1))
                if inter_w > 0 and inter_h > 0:
                    intersects = True
                if x >= px1 and y >= py1 and x2 <= px2 and y2 <= py2:
                    fully_contained = True
                    break

            if fully_contained:
                containment_stats["fully_contained"] += 1
            elif intersects:
                containment_stats["boundary_crossing"] += 1
            else:
                containment_stats["not_intersecting"] += 1

    total_instances = sum(containment_stats.values())

    # 2. Strict-filtering audit: drop any patch that cuts any annotation bbox (Cells 32 & 33)
    filename_to_ids = group_records_by_filename(index)
    total_patches = 0
    valid_patches = 0
    strict_covered_instances = 0
    images_without_valid_patches = 0

    for record_ids in filename_to_ids.values():
        boxes = []
        for img_id in record_ids:
            for ann in index.get_anns_for_image(img_id):
                x, y, w, h = ann["bbox"]
                boxes.append((x, y, x + w, y + h))

        image_valid_indices = []
        for idx, (px1, py1, px2, py2) in enumerate(patches):
            valid = True
            for x1, y1, x2, y2 in boxes:
                intersects = max(x1, px1) < min(x2, px2) and max(y1, py1) < min(y2, py2)
                contained = x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2
                if intersects and not contained:
                    valid = False
                    break
            if valid:
                image_valid_indices.append(idx)

        total_patches += len(patches)
        valid_patches += len(image_valid_indices)
        if not image_valid_indices:
            images_without_valid_patches += 1

        for x1, y1, x2, y2 in boxes:
            if any(
                x1 >= patches[i][0]
                and y1 >= patches[i][1]
                and x2 <= patches[i][2]
                and y2 <= patches[i][3]
                for i in image_valid_indices
            ):
                strict_covered_instances += 1

    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "overlap": overlap,
        "stride": patch_size - overlap,
        "positions": compute_patch_positions(image_size, patch_size, overlap),
        "patches_per_image": len(patches),
        "total_instances": total_instances,
        "fully_contained_instances": containment_stats["fully_contained"],
        "boundary_crossing_instances": containment_stats["boundary_crossing"],
        "containment_rate_pct": (
            round(100.0 * containment_stats["fully_contained"] / total_instances, 2)
            if total_instances
            else 0.0
        ),
        "strict_filtering": {
            "total_patches": total_patches,
            "valid_patches": valid_patches,
            "valid_patch_pct": round(100.0 * valid_patches / total_patches, 2)
            if total_patches
            else 0.0,
            "covered_instances": strict_covered_instances,
            "instance_coverage_pct": (
                round(100.0 * strict_covered_instances / total_instances, 2)
                if total_instances
                else 0.0
            ),
            "images_without_valid_patches": images_without_valid_patches,
        },
    }
