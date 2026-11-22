"""PyTorch Dataset for tiled grayscale solar filament instance segmentation."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from solar_filament.data.annotations import AnnotationIndex
from solar_filament.data.patches import extract_patch_targets, generate_patch_coords


def detection_collate_fn(batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Collate function for torchvision detection models (returns tuples of images and targets)."""
    return tuple(zip(*batch))


class SolarFilamentDataset(Dataset):
    """Patch-based dataset with epoch-level single-annotator sampling and LRU caching.

    For physical images with multiple annotator records (2 or 3 passes in MAGFiLO 1.0),
    ``set_epoch(epoch)`` selects one annotation record per physical image for the entire
    epoch during training, and deterministically picks ``record_ids[0]`` during validation.
    All filament categories (Left, Right, Unidentifiable) are collapsed to class label 1.
    """

    def __init__(
        self,
        filenames: Sequence[str],
        filename_to_ids: dict[str, list[Any]],
        annotation_index: AnnotationIndex,
        image_dir: str | Path,
        image_size: int = 2048,
        patch_size: int = 768,
        overlap: int = 192,
        training: bool = True,
        cache_size: int = 4,
        seed: int = 42,
    ) -> None:
        self.filenames = list(filenames)
        self.filename_to_ids = filename_to_ids
        self.annotation_index = annotation_index
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.training = training
        self.cache_size = cache_size
        self.seed = seed

        self.patch_coords = generate_patch_coords(image_size, patch_size, overlap)
        self.patch_index: list[tuple[str, int]] = [
            (filename, patch_idx)
            for filename in self.filenames
            for patch_idx in range(len(self.patch_coords))
        ]

        self.cache: OrderedDict[tuple[str, Any], tuple[np.ndarray, list[np.ndarray]]] = (
            OrderedDict()
        )
        self.epoch = 0
        self.record_selection: dict[str, Any] = {}
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Select one annotation record per physical image for the given epoch."""
        self.epoch = epoch
        if self.training:
            rng = random.Random(self.seed + epoch)
            self.record_selection = {
                filename: rng.choice(self.filename_to_ids[filename])
                for filename in self.filenames
            }
        else:
            self.record_selection = {
                filename: self.filename_to_ids[filename][0] for filename in self.filenames
            }
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.patch_index)

    def _load_image_and_masks(
        self, filename: str, image_id: Any
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        key = (filename, image_id)
        if self.cache_size > 0 and key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        image_path = self.image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            image = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

        if image.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected image shape ({self.image_size}, {self.image_size}) for {filename}, "
                f"got {image.shape}"
            )

        anns = self.annotation_index.get_anns_for_image(image_id)
        full_masks = [self.annotation_index.ann_to_mask(ann) for ann in anns]

        result = (image, full_masks)
        if self.cache_size > 0:
            self.cache[key] = result
            self.cache.move_to_end(key)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return result

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        filename, patch_idx = self.patch_index[index]
        image_id = self.record_selection[filename]

        image, full_masks = self._load_image_and_masks(filename, image_id)
        px, py = self.patch_coords[patch_idx]
        image_patch = image[py : py + self.patch_size, px : px + self.patch_size]

        clipped_masks, clipped_boxes = extract_patch_targets(
            full_masks=full_masks,
            px=px,
            py=py,
            patch_size=self.patch_size,
        )

        if clipped_masks:
            masks_tensor = torch.as_tensor(np.stack(clipped_masks), dtype=torch.uint8)
            boxes_tensor = torch.as_tensor(clipped_boxes, dtype=torch.float32)
        else:
            masks_tensor = torch.zeros((0, self.patch_size, self.patch_size), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        num_instances = len(masks_tensor)
        target: dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": torch.ones((num_instances,), dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (
                masks_tensor.sum(dim=(1, 2)).float()
                if num_instances > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((num_instances,), dtype=torch.int64),
        }

        image_tensor = torch.from_numpy(image_patch.copy()).unsqueeze(0)
        return image_tensor, target
