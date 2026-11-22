"""Tests for patch grid geometry, mask clipping, bounding boxes, and coverage audits."""

import numpy as np
import pytest

from solar_filament.data.annotations import AnnotationIndex
from solar_filament.data.patches import (
    audit_patch_coverage,
    compute_patch_positions,
    extract_patch_targets,
    generate_patch_coords,
)


def test_default_patch_positions_match_notebook():
    positions = compute_patch_positions(image_size=2048, patch_size=768, overlap=192)
    assert positions == [0, 576, 1152, 1280]

    coords = generate_patch_coords(image_size=2048, patch_size=768, overlap=192)
    assert len(coords) == 16
    assert coords[0] == (0, 0)
    assert coords[-1] == (1280, 1280)


def test_patch_positions_edge_cases():
    # Exact fit: single patch
    assert compute_patch_positions(image_size=768, patch_size=768, overlap=192) == [0]
    # Exact divisibility with 512 / 128 -> stride 384 -> 0, 384, 768, 1152, 1536 (5x5=25)
    assert compute_patch_positions(image_size=2048, patch_size=512, overlap=128) == [
        0,
        384,
        768,
        1152,
        1536,
    ]
    # 1024 / 256 -> stride 768 -> 0, 768, 1024 (3x3=9)
    assert compute_patch_positions(image_size=2048, patch_size=1024, overlap=256) == [0, 768, 1024]

    with pytest.raises(ValueError):
        compute_patch_positions(image_size=512, patch_size=768, overlap=128)
    with pytest.raises(ValueError):
        compute_patch_positions(image_size=2048, patch_size=768, overlap=768)


def test_extract_patch_targets_clips_masks_and_computes_tight_boxes():
    # Two masks in a 64x64 image:
    # Mask 1 crosses right boundary of patch [0:32, 0:32] (x in [25..40], y in [10..20])
    # Mask 2 lies completely outside patch [0:32, 0:32] (x in [40..50], y in [40..50])
    m1 = np.zeros((64, 64), dtype=np.uint8)
    m1[10:20, 25:40] = 1
    m2 = np.zeros((64, 64), dtype=np.uint8)
    m2[40:50, 40:50] = 1

    masks, boxes = extract_patch_targets([m1, m2], px=0, py=0, patch_size=32)
    assert len(masks) == 1
    assert masks[0].shape == (32, 32)
    assert masks[0].dtype == np.uint8
    assert int(masks[0].sum()) == 10 * 7  # rows 10..19, cols 25..31
    # Exclusive [x1, y1, x2, y2]
    assert boxes[0] == [25.0, 10.0, 32.0, 20.0]


def test_audit_patch_coverage_on_synthetic_data():
    coco_data = {
        "categories": [{"id": 1, "name": "Left", "supercategory": "filament"}],
        "images": [{"id": "1", "file_name": "img1.jpeg", "height": 64, "width": 64}],
        "annotations": [
            # Fully contained in patch (0,0)-(32,32)
            {"id": "a1", "image_id": "1", "category_id": 1, "bbox": [4, 4, 10, 10], "area": 100},
            # Too wide for any 32x32 patch -> boundary crossing
            {"id": "a2", "image_id": "1", "category_id": 1, "bbox": [10, 10, 40, 5], "area": 200},
        ],
    }
    report = audit_patch_coverage(
        AnnotationIndex(coco_data),
        image_size=64,
        patch_size=32,
        overlap=8,
    )
    assert report["total_instances"] == 2
    assert report["fully_contained_instances"] == 1
    assert report["boundary_crossing_instances"] == 1
    assert report["containment_rate_pct"] == pytest.approx(50.0)
