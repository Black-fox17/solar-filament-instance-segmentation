"""Postprocessing: candidate extraction, Union-Find merging, and overlap resolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PatchCandidate:
    """Compact full-image detection candidate extracted from a single patch."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) with exclusive x2, y2
    mask: np.ndarray  # Cropped boolean mask of shape (y2 - y1, x2 - x1)
    score: float


def bbox_iou_and_containment(
    box_a: tuple[int, int, int, int] | tuple[float, float, float, float],
    box_b: tuple[int, int, int, int] | tuple[float, float, float, float],
) -> tuple[float, float]:
    """Compute bounding-box IoU and smaller-box containment ratio.

    Boxes use ``(x1, y1, x2, y2)`` with exclusive ``x2`` and ``y2``.
    Containment is ``intersection / min(area_a, area_b)``.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    intersection = iw * ih

    area_a = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    area_b = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    union = area_a + area_b - intersection

    iou = intersection / union if union > 0.0 else 0.0
    smaller_area = min(area_a, area_b)
    containment = intersection / smaller_area if smaller_area > 0.0 else 0.0
    return iou, containment


def extract_patch_predictions(
    prediction: dict[str, Any],
    patch_x: int,
    patch_y: int,
    score_threshold: float = 0.50,
    mask_threshold: float = 0.50,
    min_mask_area: int = 20,
) -> list[PatchCandidate]:
    """Filter raw Mask R-CNN patch predictions and project them to full-image coordinates."""
    candidates: list[PatchCandidate] = []
    pred_masks = prediction["masks"]
    pred_scores = prediction["scores"]

    for i in range(len(pred_scores)):
        score_val = pred_scores[i]
        score = float(score_val.item()) if hasattr(score_val, "item") else float(score_val)
        if score < score_threshold:
            continue

        raw_mask = pred_masks[i, 0]
        mask_np = (
            raw_mask.detach().cpu().numpy()
            if hasattr(raw_mask, "detach")
            else np.asarray(raw_mask)
        )
        binary_mask = mask_np >= mask_threshold
        if int(binary_mask.sum()) < min_mask_area:
            continue

        ys, xs = np.where(binary_mask)
        local_x1 = int(xs.min())
        local_y1 = int(ys.min())
        local_x2 = int(xs.max()) + 1
        local_y2 = int(ys.max()) + 1

        cropped = binary_mask[local_y1:local_y2, local_x1:local_x2].copy()
        candidates.append(
            PatchCandidate(
                bbox=(
                    patch_x + local_x1,
                    patch_y + local_y1,
                    patch_x + local_x2,
                    patch_y + local_y2,
                ),
                mask=cropped,
                score=score,
            )
        )

    return candidates


def merge_patch_candidates(
    candidates: list[PatchCandidate],
    image_size: int = 2048,
    bbox_iou_threshold: float = 0.30,
    bbox_containment_threshold: float = 0.50,
    min_mask_area: int = 20,
) -> tuple[list[np.ndarray], list[float]]:
    """Merge duplicate detections across overlapping patches using Union-Find.

    Two candidates are linked if their bounding-box IoU >= ``bbox_iou_threshold``
    or smaller-box containment >= ``bbox_containment_threshold``. Each connected
    component is rasterized into a full-resolution boolean mask by logical OR,
    assigned the maximum confidence score in its component.
    """
    if not candidates:
        return [], []

    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n):
        for j in range(i + 1, n):
            iou, containment = bbox_iou_and_containment(
                candidates[i].bbox,
                candidates[j].bbox,
            )
            if iou >= bbox_iou_threshold or containment >= bbox_containment_threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged_masks: list[np.ndarray] = []
    merged_scores: list[float] = []

    for group_indices in groups.values():
        merged_mask = np.zeros((image_size, image_size), dtype=bool)
        group_score = 0.0

        for idx in group_indices:
            cand = candidates[idx]
            x1, y1, x2, y2 = cand.bbox
            merged_mask[y1:y2, x1:x2] |= cand.mask
            group_score = max(group_score, cand.score)

        if int(merged_mask.sum()) < min_mask_area:
            continue

        merged_masks.append(merged_mask)
        merged_scores.append(group_score)

    return merged_masks, merged_scores


def make_masks_exclusive(
    masks: list[np.ndarray],
    scores: list[float],
    image_size: int = 2048,
    min_mask_area: int = 20,
) -> list[np.ndarray]:
    """Resolve remaining pixel overlaps so every foreground pixel belongs to at most one instance.

    Higher-confidence masks retain ownership of overlapping pixels; lower-confidence
    masks have already-claimed pixels subtracted and are dropped if their remaining
    area falls below ``min_mask_area``.
    """
    if not masks:
        return []

    order = np.argsort(scores)[::-1]
    claimed = np.zeros((image_size, image_size), dtype=bool)
    exclusive_masks: list[np.ndarray] = []

    for idx in order:
        mask = masks[int(idx)].astype(bool)
        exclusive = mask & ~claimed
        if int(exclusive.sum()) < min_mask_area:
            continue
        exclusive_masks.append(exclusive)
        claimed |= exclusive

    return exclusive_masks
