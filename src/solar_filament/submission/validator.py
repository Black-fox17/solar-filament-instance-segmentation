"""Competition submission CSV generation and strict validation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from solar_filament.inference.predictor import ImagePredictionResult
from solar_filament.submission.rle import decode_mask_rle, encode_mask_rle

SUBMISSION_COLUMNS = ["filament_id", "segmentation_rle"]
_FILAMENT_ID_PATTERN = re.compile(r"^.+\_\d+$")


def build_submission_rows(
    results: list[ImagePredictionResult],
    encoder: Callable[[np.ndarray], str] = encode_mask_rle,
) -> list[dict[str, str]]:
    """Convert per-image exclusive binary masks into competition CSV rows."""
    rows: list[dict[str, str]] = []
    for result in results:
        stem = result.image_path.stem
        for instance_idx, binary_mask in enumerate(result.masks, start=1):
            rows.append(
                {
                    "filament_id": f"{stem}_{instance_idx}",
                    "segmentation_rle": encoder(binary_mask),
                }
            )
    return rows


def write_submission_csv(
    rows: list[dict[str, str]],
    output_csv: str | Path,
) -> pd.DataFrame:
    """Write submission rows to disk with exact required columns."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    df.to_csv(out_path, index=False)
    return df


def validate_submission_csv(
    csv_path: str | Path,
    image_size: int = 2048,
    min_mask_area: int = 20,
    test_dir: str | Path | None = None,
    decoder: Callable[[str, int, int], np.ndarray] = decode_mask_rle,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run structural, RLE decode, area, and zero-overlap checks on a submission CSV.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If any structural, RLE, area, or overlap check fails.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {path}")

    df = pd.read_csv(path)

    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Invalid columns: expected {SUBMISSION_COLUMNS}, got {list(df.columns)}"
        )
    if len(df) == 0:
        raise ValueError("Submission CSV contains 0 rows.")
    if not df["filament_id"].is_unique:
        raise ValueError("Duplicate filament_id values detected.")
    if not df["segmentation_rle"].notna().all():
        raise ValueError("Submission contains null segmentation_rle values.")
    if not (df["segmentation_rle"].astype(str).str.len() > 0).all():
        raise ValueError("Submission contains empty segmentation_rle strings.")
    if not df["filament_id"].astype(str).map(lambda s: bool(_FILAMENT_ID_PATTERN.match(s))).all():
        raise ValueError("Every filament_id must match '<image_stem>_<integer_index>'.")

    masks_by_image: dict[str, list[np.ndarray]] = defaultdict(list)
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Validating RLE masks")

    for row_idx, row in iterator:
        filament_id = str(row["filament_id"])
        image_id = filament_id.rsplit("_", 1)[0]
        decoded = decoder(str(row["segmentation_rle"]), image_size, image_size)

        if decoded.shape != (image_size, image_size):
            raise ValueError(
                f"Row {row_idx} ({filament_id}): expected mask shape "
                f"({image_size}, {image_size}), got {decoded.shape}"
            )
        if decoded.dtype != np.uint8:
            raise ValueError(
                f"Row {row_idx} ({filament_id}): expected uint8 mask, got {decoded.dtype}"
            )

        area = int((decoded > 0).sum())
        if area < min_mask_area:
            raise ValueError(
                f"Row {row_idx} ({filament_id}): mask area {area} < min_mask_area ({min_mask_area})"
            )

        masks_by_image[image_id].append(decoded > 0)

    overlap_details: list[dict[str, Any]] = []
    for image_id, masks in masks_by_image.items():
        occupied = np.zeros((image_size, image_size), dtype=bool)
        for instance_idx, mask in enumerate(masks, start=1):
            overlap = occupied & mask
            if overlap.any():
                overlap_details.append(
                    {
                        "image_id": image_id,
                        "instance": instance_idx,
                        "overlapping_pixels": int(overlap.sum()),
                    }
                )
            occupied |= mask

    if overlap_details:
        raise ValueError(
            f"Submission contains {len(overlap_details)} overlapping instance(s); "
            f"first example: {overlap_details[0]}"
        )

    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "unique_predicted_images": len(masks_by_image),
        "overlapping_instances": 0,
    }

    if test_dir is not None:
        test_path = Path(test_dir)
        test_stems = {
            p.stem
            for p in test_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        predicted_stems = set(masks_by_image.keys())
        summary["test_images"] = len(test_stems)
        summary["images_without_predictions"] = len(test_stems - predicted_stems)

    return summary
