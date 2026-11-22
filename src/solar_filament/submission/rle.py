"""COCO compressed RLE encoding and decoding via pycocotools."""

from __future__ import annotations

import numpy as np


def _import_mask_utils():
    try:
        from pycocotools import mask as mask_utils

        return mask_utils
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for competition COCO RLE encoding and decoding. "
            "Install it with: pip install pycocotools"
        ) from exc


def encode_mask_rle(binary_mask: np.ndarray) -> str:
    """Encode a 2D binary mask into a UTF-8 COCO compressed RLE counts string.

    Converts the mask to Fortran-contiguous ``uint8`` as required by ``pycocotools``.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary_mask.shape}")
    mask_utils = _import_mask_utils()
    fortran_mask = np.asfortranarray((binary_mask > 0).astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else str(counts)


def decode_mask_rle(rle_counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode a UTF-8 COCO compressed RLE counts string into a 2D ``uint8`` mask."""
    mask_utils = _import_mask_utils()
    rle = {"size": [height, width], "counts": rle_counts.encode("utf-8")}
    decoded = mask_utils.decode(rle)
    return decoded.astype(np.uint8)
