"""Hardware-agnostic PyTorch device selection."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device string ('auto', 'cuda', 'cuda:0', 'mps', 'cpu').

    Raises:
        ValueError: If 'cuda' or 'mps' is explicitly requested but unavailable.
    """
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if req.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            f"Requested device '{requested}', but CUDA is not available in this environment."
        )

    if req == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError(
            f"Requested device '{requested}', but Apple MPS is not available in this environment."
        )

    return torch.device(requested)
