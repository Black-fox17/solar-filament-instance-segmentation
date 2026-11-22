"""Tests for COCO annotation parsing, multi-annotator grouping, splitting, and audits."""

import json

import numpy as np
import pytest

from solar_filament.data.annotations import (
    AnnotationIndex,
    audit_annotations,
    decode_annotation_mask,
    group_records_by_filename,
    split_by_filename,
)


def _make_synthetic_coco() -> dict:
    return {
        "categories": [
            {"id": 1, "name": "Left", "supercategory": "filament"},
            {"id": 2, "name": "Right", "supercategory": "filament"},
            {"id": 3, "name": "Unidentifiable", "supercategory": "filament"},
            {"id": 4, "name": "Ambiguous", "supercategory": "filament"},
        ],
        "images": [
            {"id": "01-imgA", "file_name": "imgA.jpeg", "height": 64, "width": 64},
            {"id": "02-imgA", "file_name": "imgA.jpeg", "height": 64, "width": 64},
            {"id": "01-imgB", "file_name": "imgB.jpeg", "height": 64, "width": 64},
            {"id": "01-imgC", "file_name": "imgC.jpeg", "height": 64, "width": 64},
            {"id": "01-imgD", "file_name": "imgD.jpeg", "height": 64, "width": 64},
        ],
        "annotations": [
            {
                "id": "a1",
                "image_id": "01-imgA",
                "category_id": 1,
                "bbox": [4.0, 4.0, 10.0, 10.0],
                "area": 100.0,
                "iscrowd": 0,
                "segmentation": [[4, 4, 14, 4, 14, 14, 4, 14]],
            },
            {
                "id": "a2",
                "image_id": "02-imgA",
                "category_id": 2,
                "bbox": [5.0, 5.0, 10.0, 10.0],
                "area": 100.0,
                "iscrowd": 0,
                "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]],
            },
            # Two overlapping annotations inside the same record (01-imgB)
            {
                "id": "b1",
                "image_id": "01-imgB",
                "category_id": 2,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            },
            {
                "id": "b2",
                "image_id": "01-imgB",
                "category_id": 2,
                "bbox": [12.0, 12.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
                "segmentation": [[12, 12, 32, 12, 32, 32, 12, 32]],
            },
            {
                "id": "c1",
                "image_id": "01-imgC",
                "category_id": 3,
                "bbox": [20.0, 20.0, 8.0, 4.0],
                "area": 32.0,
                "iscrowd": 0,
                "segmentation": [[20, 20, 28, 20, 28, 24, 20, 24]],
            },
            {
                "id": "d1",
                "image_id": "01-imgD",
                "category_id": 1,
                "bbox": [30.0, 30.0, 10.0, 10.0],
                "area": 100.0,
                "iscrowd": 0,
                "segmentation": [[30, 30, 40, 30, 40, 40, 30, 40]],
            },
        ],
    }


def test_annotation_index_and_grouping(tmp_path):
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(_make_synthetic_coco()), encoding="utf-8")

    index = AnnotationIndex.from_json(json_path)
    grouped = group_records_by_filename(index)

    assert len(index.imgs) == 5
    assert len(grouped) == 4
    assert grouped["imgA.jpeg"] == ["01-imgA", "02-imgA"]


def test_split_by_filename_prevents_leakage():
    index = AnnotationIndex(_make_synthetic_coco())
    grouped = group_records_by_filename(index)

    train_files, val_files = split_by_filename(grouped, val_size=0.25, random_state=42)
    assert len(train_files) == 3
    assert len(val_files) == 1
    assert set(train_files).isdisjoint(set(val_files))

    # Every record for imgA.jpeg must land on the same side of the split
    train_ids = {i for f in train_files for i in grouped[f]}
    val_ids = {i for f in val_files for i in grouped[f]}
    assert train_ids.isdisjoint(val_ids)
    assert ("01-imgA" in train_ids) == ("02-imgA" in train_ids)


def test_decode_polygon_mask_shape_and_dtype():
    ann = {"id": "p1", "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]]}
    mask = decode_annotation_mask(ann, height=64, width=64)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    assert mask[15, 15] == 1
    assert mask[0, 0] == 0


def test_audit_annotations_detects_overlaps_and_multiplicity():
    index = AnnotationIndex(_make_synthetic_coco())
    report = audit_annotations(index, check_mask_overlaps=True, high_overlap_iou_threshold=0.50)

    assert report["total_records"] == 5
    assert report["unique_filenames"] == 4
    assert report["duplicate_filename_groups"] == 1
    assert report["annotation_multiplicity"] == {1: 3, 2: 1}
    assert report["category_counts"]["Ambiguous"] == 0
    assert report["mask_integrity"]["empty_masks"] == 0
    assert report["mask_integrity"]["invalid_masks"] == 0
    assert report["mask_integrity"]["records_with_overlapping_instances"] == 1
    assert len(report["mask_integrity"]["high_overlap_pairs"]) == 1
    assert report["mask_integrity"]["high_overlap_pairs"][0]["image_id"] == "01-imgB"
