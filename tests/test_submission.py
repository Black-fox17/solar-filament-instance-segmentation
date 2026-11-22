"""Tests for submission generation, validation, and COCO RLE round-tripping."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_filament.inference.predictor import ImagePredictionResult
from solar_filament.submission.rle import decode_mask_rle, encode_mask_rle
from solar_filament.submission.validator import (
    build_submission_rows,
    validate_submission_csv,
    write_submission_csv,
)


def _toy_encode(mask: np.ndarray) -> str:
    """Deterministic coordinate-string encoder for testing validator logic without pycocotools."""
    ys, xs = np.where(mask > 0)
    return ";".join(f"{int(y)},{int(x)}" for y, x in zip(ys, xs))


def _toy_decode(encoded: str, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for pair in encoded.split(";"):
        if not pair:
            continue
        y_str, x_str = pair.split(",")
        mask[int(y_str), int(x_str)] = 1
    return mask


def test_submission_build_write_and_validate_passes(tmp_path):
    m1 = np.zeros((32, 32), dtype=bool)
    m1[2:8, 2:8] = True  # 36 px
    m2 = np.zeros((32, 32), dtype=bool)
    m2[15:21, 15:21] = True  # 36 px

    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    (test_dir / "img001.jpeg").write_bytes(b"")
    (test_dir / "img002.jpeg").write_bytes(b"")

    results = [
        ImagePredictionResult(
            image_path=test_dir / "img001.jpeg",
            masks=[m1, m2],
            raw_candidate_count=5,
            merged_count=2,
        ),
        ImagePredictionResult(
            image_path=test_dir / "img002.jpeg",
            masks=[],
            raw_candidate_count=0,
            merged_count=0,
        ),
    ]

    rows = build_submission_rows(results, encoder=_toy_encode)
    csv_path = tmp_path / "submission.csv"
    df = write_submission_csv(rows, csv_path)
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert df["filament_id"].tolist() == ["img001_1", "img001_2"]

    summary = validate_submission_csv(
        csv_path=csv_path,
        image_size=32,
        min_mask_area=20,
        test_dir=test_dir,
        decoder=_toy_decode,
        show_progress=False,
    )
    assert summary["rows"] == 2
    assert summary["unique_predicted_images"] == 1
    assert summary["test_images"] == 2
    assert summary["images_without_predictions"] == 1
    assert summary["overlapping_instances"] == 0


def test_validator_rejects_overlapping_masks(tmp_path):
    m1 = np.zeros((32, 32), dtype=bool)
    m1[2:8, 2:8] = True
    m2 = np.zeros((32, 32), dtype=bool)
    m2[5:11, 5:11] = True  # overlaps m1 on [5:8, 5:8]

    csv_path = tmp_path / "overlap.csv"
    pd.DataFrame(
        [
            {"filament_id": "img001_1", "segmentation_rle": _toy_encode(m1)},
            {"filament_id": "img001_2", "segmentation_rle": _toy_encode(m2)},
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="overlapping"):
        validate_submission_csv(
            csv_path=csv_path,
            image_size=32,
            min_mask_area=20,
            decoder=_toy_decode,
            show_progress=False,
        )


def test_validator_rejects_bad_schema_and_small_area(tmp_path):
    bad_cols = tmp_path / "bad_cols.csv"
    pd.DataFrame([{"id": "img_1", "rle": "1,1"}]).to_csv(bad_cols, index=False)
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_submission_csv(bad_cols, image_size=32, decoder=_toy_decode, show_progress=False)

    small_mask = np.zeros((32, 32), dtype=bool)
    small_mask[0:2, 0:2] = True  # area 4 < 20
    small_csv = tmp_path / "small.csv"
    pd.DataFrame(
        [{"filament_id": "img001_1", "segmentation_rle": _toy_encode(small_mask)}]
    ).to_csv(small_csv, index=False)
    with pytest.raises(ValueError, match="min_mask_area"):
        validate_submission_csv(
            small_csv,
            image_size=32,
            min_mask_area=20,
            decoder=_toy_decode,
            show_progress=False,
        )


def test_pycocotools_rle_roundtrip():
    pytest.importorskip("pycocotools")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:25, 30:50] = 1
    rle_str = encode_mask_rle(mask)
    decoded = decode_mask_rle(rle_str, height=64, width=64)
    assert decoded.shape == (64, 64)
    assert decoded.dtype == np.uint8
    assert np.array_equal(mask, decoded)
