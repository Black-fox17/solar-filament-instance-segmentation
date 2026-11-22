"""COCO annotation parsing, multi-annotator grouping, splitting, and integrity audits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split


def decode_annotation_mask(ann: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Decode a COCO annotation segmentation (polygon list or RLE) into a uint8 binary mask.

    Uses ``pycocotools.mask`` when available to match ``COCO.annToMask`` exactly.
    Falls back to PIL polygon rasterization for polygon lists when ``pycocotools``
    is not installed.
    """
    segm = ann.get("segmentation")
    if segm is None:
        raise ValueError(f"Annotation {ann.get('id')} is missing 'segmentation'.")

    try:
        from pycocotools import mask as mask_utils

        if isinstance(segm, list):
            rles = mask_utils.frPyObjects(segm, height, width)
            rle = mask_utils.merge(rles)
            return mask_utils.decode(rle).astype(np.uint8)
        if isinstance(segm, dict):
            if isinstance(segm.get("counts"), list):
                rle = mask_utils.frPyObjects(segm, height, width)
            else:
                rle = segm
            return mask_utils.decode(rle).astype(np.uint8)
        raise TypeError(f"Unsupported segmentation type: {type(segm).__name__}")
    except ImportError as exc:
        if isinstance(segm, dict):
            raise ImportError(
                "pycocotools is required to decode RLE segmentations. "
                "Install it with: pip install pycocotools"
            ) from exc
        if not isinstance(segm, list):
            raise TypeError(f"Unsupported segmentation type: {type(segm).__name__}") from exc

        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in segm:
            if len(poly) >= 6:
                coords = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
                draw.polygon(coords, outline=1, fill=1)
        return np.asarray(canvas, dtype=np.uint8)


class AnnotationIndex:
    """In-memory index over a COCO-format annotation JSON.

    Provides the subset of COCO lookup operations used across EDA, dataset loading,
    and auditing without requiring global mutable state.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        self.imgs: dict[Any, dict[str, Any]] = {img["id"]: img for img in data.get("images", [])}
        self.cats: dict[int, dict[str, Any]] = {
            cat["id"]: cat for cat in data.get("categories", [])
        }
        self.anns: dict[Any, dict[str, Any]] = {
            ann["id"]: ann for ann in data.get("annotations", [])
        }
        self.img_to_anns: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            self.img_to_anns[ann["image_id"]].append(ann)

    @classmethod
    def from_json(cls, path: str | Path) -> AnnotationIndex:
        json_path = Path(path)
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation JSON not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def get_img_ids(self) -> list[Any]:
        return list(self.imgs.keys())

    def get_anns_for_image(self, image_id: Any) -> list[dict[str, Any]]:
        return list(self.img_to_anns.get(image_id, []))

    def ann_to_mask(self, ann: dict[str, Any]) -> np.ndarray:
        info = self.imgs[ann["image_id"]]
        return decode_annotation_mask(ann, height=int(info["height"]), width=int(info["width"]))


def group_records_by_filename(index: AnnotationIndex) -> dict[str, list[Any]]:
    """Group COCO image record IDs by their physical ``file_name``.

    In MAGFiLO 1.0, multiple annotator passes share the same physical JPEG file
    under distinct COCO ``id`` prefixes (1,154 records across 707 unique files).
    """
    grouped: dict[str, list[Any]] = defaultdict(list)
    for img_id, info in index.imgs.items():
        grouped[info["file_name"]].append(img_id)
    return dict(grouped)


def split_by_filename(
    filename_to_ids: dict[str, list[Any]],
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """Split unique physical filenames into disjoint train and validation lists.

    Splitting by physical filename prevents multi-annotator records of the same
    solar image from leaking across train and validation splits.
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    unique_filenames = list(filename_to_ids.keys())
    if len(unique_filenames) < 2:
        raise ValueError("Need at least 2 unique physical images to create a train/val split.")

    train_filenames, val_filenames = train_test_split(
        unique_filenames,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    train_set = set(train_filenames)
    val_set = set(val_filenames)
    if not train_set.isdisjoint(val_set):
        raise RuntimeError("Filename leakage detected between train and validation splits.")

    return list(train_filenames), list(val_filenames)


def audit_annotations(
    index: AnnotationIndex,
    check_mask_overlaps: bool = True,
    high_overlap_iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Run the dataset integrity and morphology audit from the research notebook."""
    filename_to_ids = group_records_by_filename(index)
    multiplicity = dict(sorted(Counter(len(ids) for ids in filename_to_ids.values()).items()))
    dimensions = dict(
        Counter((int(info["height"]), int(info["width"])) for info in index.imgs.values())
    )

    all_anns = list(index.anns.values())
    category_counts = Counter(ann["category_id"] for ann in all_anns)
    instances_per_record = [len(index.get_anns_for_image(i)) for i in index.imgs]

    areas = np.array([ann.get("area", 0.0) for ann in all_anns], dtype=np.float64)
    widths = np.array([ann["bbox"][2] for ann in all_anns], dtype=np.float64) if all_anns else np.array([])
    heights = np.array([ann["bbox"][3] for ann in all_anns], dtype=np.float64) if all_anns else np.array([])
    bbox_areas = widths * heights
    aspect_ratios = (
        np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), 1e-6)
        if all_anns
        else np.array([])
    )
    occupancy = areas / np.maximum(bbox_areas, 1.0) if all_anns else np.array([])

    report: dict[str, Any] = {
        "total_records": len(index.imgs),
        "unique_filenames": len(filename_to_ids),
        "duplicate_filename_groups": sum(1 for ids in filename_to_ids.values() if len(ids) > 1),
        "annotation_multiplicity": multiplicity,
        "image_dimensions": {f"{h}x{w}": count for (h, w), count in dimensions.items()},
        "total_annotations": len(all_anns),
        "records_with_zero_annotations": sum(n == 0 for n in instances_per_record),
        "non_positive_area_annotations": int((areas <= 0).sum()) if len(areas) else 0,
        "instances_per_record": {
            "min": int(min(instances_per_record)) if instances_per_record else 0,
            "median": float(np.median(instances_per_record)) if instances_per_record else 0.0,
            "mean": float(np.mean(instances_per_record)) if instances_per_record else 0.0,
            "max": int(max(instances_per_record)) if instances_per_record else 0,
        },
        "category_counts": {
            index.cats[cid]["name"]: int(category_counts.get(cid, 0))
            for cid in sorted(index.cats)
        },
        "morphology": {
            "area_median": float(np.median(areas)) if len(areas) else 0.0,
            "area_mean": float(np.mean(areas)) if len(areas) else 0.0,
            "area_p90": float(np.percentile(areas, 90)) if len(areas) else 0.0,
            "bbox_width_median": float(np.median(widths)) if len(widths) else 0.0,
            "bbox_height_median": float(np.median(heights)) if len(heights) else 0.0,
            "aspect_ratio_median": float(np.median(aspect_ratios)) if len(aspect_ratios) else 0.0,
            "bbox_occupancy_median_pct": float(np.median(occupancy) * 100.0) if len(occupancy) else 0.0,
        },
    }

    if check_mask_overlaps:
        empty_masks = 0
        invalid_masks = 0
        overlapping_records: list[dict[str, Any]] = []
        high_overlap_pairs: list[dict[str, Any]] = []

        for img_id, info in index.imgs.items():
            anns = index.get_anns_for_image(img_id)
            occupied = np.zeros((int(info["height"]), int(info["width"])), dtype=bool)
            record_overlap_pixels = 0
            decoded_masks: list[tuple[dict[str, Any], np.ndarray]] = []

            for ann in anns:
                try:
                    mask = index.ann_to_mask(ann).astype(bool)
                except Exception:
                    invalid_masks += 1
                    continue

                if not mask.any():
                    empty_masks += 1
                    continue

                record_overlap_pixels += int((occupied & mask).sum())
                occupied |= mask
                decoded_masks.append((ann, mask))

            if record_overlap_pixels > 0:
                overlapping_records.append(
                    {
                        "image_id": img_id,
                        "filename": info["file_name"],
                        "overlap_pixels": record_overlap_pixels,
                    }
                )
                for i in range(len(decoded_masks)):
                    ann_a, mask_a = decoded_masks[i]
                    for j in range(i + 1, len(decoded_masks)):
                        ann_b, mask_b = decoded_masks[j]
                        inter = int((mask_a & mask_b).sum())
                        if inter == 0:
                            continue
                        union = int((mask_a | mask_b).sum())
                        iou = inter / union if union > 0 else 0.0
                        if iou >= high_overlap_iou_threshold:
                            high_overlap_pairs.append(
                                {
                                    "image_id": img_id,
                                    "filename": info["file_name"],
                                    "ann_a": ann_a["id"],
                                    "ann_b": ann_b["id"],
                                    "iou": round(float(iou), 4),
                                }
                            )

        report["mask_integrity"] = {
            "empty_masks": empty_masks,
            "invalid_masks": invalid_masks,
            "records_with_overlapping_instances": len(overlapping_records),
            "overlapping_records": overlapping_records,
            "high_overlap_pairs": high_overlap_pairs,
        }

    return report
