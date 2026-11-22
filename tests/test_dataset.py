"""Tests for SolarFilamentDataset indexing, epoch annotator selection, and collation."""

import numpy as np
import torch
from PIL import Image

from solar_filament.data.annotations import AnnotationIndex, group_records_by_filename
from solar_filament.data.dataset import SolarFilamentDataset, detection_collate_fn


def _create_synthetic_dataset(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((64, 64), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(img_dir / "sun1.jpeg")

    coco_data = {
        "categories": [{"id": 1, "name": "Left", "supercategory": "filament"}],
        "images": [
            {"id": "rec1", "file_name": "sun1.jpeg", "height": 64, "width": 64},
            {"id": "rec2", "file_name": "sun1.jpeg", "height": 64, "width": 64},
        ],
        "annotations": [
            # rec1 has a filament in top-left patch (0, 0)
            {
                "id": "a1",
                "image_id": "rec1",
                "category_id": 1,
                "bbox": [4, 4, 10, 10],
                "area": 100,
                "segmentation": [[4, 4, 14, 4, 14, 14, 4, 14]],
            },
            # rec2 has a filament in bottom-right patch (32, 32)
            {
                "id": "a2",
                "image_id": "rec2",
                "category_id": 1,
                "bbox": [40, 40, 10, 10],
                "area": 100,
                "segmentation": [[40, 40, 50, 40, 50, 50, 40, 50]],
            },
        ],
    }
    index = AnnotationIndex(coco_data)
    grouped = group_records_by_filename(index)
    return img_dir, index, grouped


def test_dataset_shapes_positive_and_empty_patches(tmp_path):
    img_dir, index, grouped = _create_synthetic_dataset(tmp_path)
    # 64x64 with patch_size=32, overlap=0 -> 4 patches: (0,0), (32,0), (0,32), (32,32)
    val_ds = SolarFilamentDataset(
        filenames=["sun1.jpeg"],
        filename_to_ids=grouped,
        annotation_index=index,
        image_dir=img_dir,
        image_size=64,
        patch_size=32,
        overlap=0,
        training=False,
        cache_size=2,
    )
    assert len(val_ds) == 4
    # Validation always picks record_ids[0] ("rec1")
    assert val_ds.record_selection["sun1.jpeg"] == "rec1"

    # Patch 0 (0,0) has 1 instance; patch 3 (32,32) is empty for rec1
    img0, target0 = val_ds[0]
    assert img0.shape == (1, 32, 32)
    assert img0.dtype == torch.float32
    assert 0.0 <= float(img0.min()) <= float(img0.max()) <= 1.0
    assert target0["masks"].shape == (1, 32, 32)
    assert target0["boxes"].shape == (1, 4)
    assert target0["labels"].tolist() == [1]
    assert (target0["boxes"][:, 2:] > target0["boxes"][:, :2]).all()

    img3, target3 = val_ds[3]
    assert img3.shape == (1, 32, 32)
    assert target3["masks"].shape == (0, 32, 32)
    assert target3["boxes"].shape == (0, 4)
    assert target3["labels"].shape == (0,)
    assert target3["area"].shape == (0,)


def test_training_dataset_set_epoch_switches_and_clears_cache(tmp_path):
    img_dir, index, grouped = _create_synthetic_dataset(tmp_path)
    train_ds = SolarFilamentDataset(
        filenames=["sun1.jpeg"],
        filename_to_ids=grouped,
        annotation_index=index,
        image_dir=img_dir,
        image_size=64,
        patch_size=32,
        overlap=0,
        training=True,
        cache_size=2,
        seed=42,
    )
    _ = train_ds[0]
    assert len(train_ds.cache) == 1

    seen = set()
    for ep in range(10):
        train_ds.set_epoch(ep)
        assert len(train_ds.cache) == 0
        seen.add(train_ds.record_selection["sun1.jpeg"])
    assert seen == {"rec1", "rec2"}


def test_detection_collate_fn(tmp_path):
    img_dir, index, grouped = _create_synthetic_dataset(tmp_path)
    ds = SolarFilamentDataset(
        filenames=["sun1.jpeg"],
        filename_to_ids=grouped,
        annotation_index=index,
        image_dir=img_dir,
        image_size=64,
        patch_size=32,
        overlap=0,
        training=False,
    )
    images, targets = detection_collate_fn([ds[0], ds[1]])
    assert len(images) == 2
    assert len(targets) == 2
