"""Tests for inference postprocessing: candidate extraction, merging, and overlap resolution."""

import numpy as np
import pytest
import torch

from solar_filament.inference.postprocess import (
    PatchCandidate,
    bbox_iou_and_containment,
    extract_patch_predictions,
    make_masks_exclusive,
    merge_patch_candidates,
)


def test_bbox_iou_and_containment():
    # Identical boxes
    iou, cont = bbox_iou_and_containment((0, 0, 10, 10), (0, 0, 10, 10))
    assert iou == pytest.approx(1.0)
    assert cont == pytest.approx(1.0)

    # Disjoint boxes
    iou, cont = bbox_iou_and_containment((0, 0, 10, 10), (20, 20, 30, 30))
    assert iou == pytest.approx(0.0)
    assert cont == pytest.approx(0.0)

    # Small box fully contained inside large box
    iou, cont = bbox_iou_and_containment((0, 0, 100, 100), (10, 10, 20, 20))
    assert iou == pytest.approx(100 / 10000)
    assert cont == pytest.approx(1.0)


def test_extract_patch_predictions_filters_and_offsets():
    # Patch size 32x32 at global offset (100, 200)
    m_keep = np.zeros((32, 32), dtype=np.float32)
    m_keep[5:15, 6:16] = 0.9  # area 100 >= 20

    m_low_score = np.ones((32, 32), dtype=np.float32) * 0.9
    m_too_small = np.zeros((32, 32), dtype=np.float32)
    m_too_small[0:2, 0:2] = 0.9  # area 4 < 20

    prediction = {
        "scores": torch.tensor([0.85, 0.30, 0.95]),
        "masks": torch.from_numpy(np.stack([m_keep, m_low_score, m_too_small])[:, None, :, :]),
    }

    cands = extract_patch_predictions(
        prediction=prediction,
        patch_x=100,
        patch_y=200,
        score_threshold=0.50,
        mask_threshold=0.50,
        min_mask_area=20,
    )
    assert len(cands) == 1
    assert cands[0].bbox == (106, 205, 116, 215)
    assert cands[0].mask.shape == (10, 10)
    assert cands[0].score == pytest.approx(0.85)


def test_merge_patch_candidates_merges_duplicates_and_keeps_disjoint():
    # Candidate A and B overlap strongly -> should merge into 1 mask with max score 0.92
    # Candidate C is disjoint -> stays separate with score 0.70
    c_a = PatchCandidate(bbox=(10, 10, 20, 20), mask=np.ones((10, 10), dtype=bool), score=0.80)
    c_b = PatchCandidate(bbox=(12, 10, 22, 20), mask=np.ones((10, 10), dtype=bool), score=0.92)
    c_c = PatchCandidate(bbox=(40, 40, 50, 50), mask=np.ones((10, 10), dtype=bool), score=0.70)

    masks, scores = merge_patch_candidates(
        [c_a, c_b, c_c],
        image_size=64,
        bbox_iou_threshold=0.30,
        bbox_containment_threshold=0.50,
        min_mask_area=20,
    )
    assert len(masks) == 2
    assert sorted(round(s, 2) for s in scores) == [0.70, 0.92]
    # Union of [10..20, 10..20] and [10..20, 12..22] has area 10 * 12 = 120
    areas = sorted(int(m.sum()) for m in masks)
    assert areas == [100, 120]


def test_make_masks_exclusive_resolves_overlaps_by_confidence():
    m_high = np.zeros((64, 64), dtype=bool)
    m_high[10:20, 10:20] = True  # 100 px

    # Partially overlaps m_high (leaves 60 px exclusive >= 20 -> kept)
    m_mid = np.zeros((64, 64), dtype=bool)
    m_mid[10:20, 16:26] = True  # 40 px overlap, 60 px exclusive

    # Completely covered by m_high (leaves 0 px < 20 -> dropped)
    m_low = np.zeros((64, 64), dtype=bool)
    m_low[12:18, 12:18] = True

    exclusive = make_masks_exclusive(
        masks=[m_low, m_mid, m_high],
        scores=[0.60, 0.75, 0.95],
        image_size=64,
        min_mask_area=20,
    )
    assert len(exclusive) == 2
    assert int(exclusive[0].sum()) == 100
    assert int(exclusive[1].sum()) == 60
    # Zero pixel-level overlap
    assert int((exclusive[0] & exclusive[1]).sum()) == 0
