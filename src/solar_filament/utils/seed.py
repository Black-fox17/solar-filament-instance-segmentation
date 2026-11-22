"""Random seed utilities with honest reproducibility notes."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Note:
        Setting seeds makes splits, patch sampling, and weight initialization
        repeatable on the same hardware/software stack, but does not guarantee
        bit-exact reproducibility across different GPUs, CUDA/cuDNN versions,
        or DataLoader worker counts. Some torchvision detection operators
        (such as ROIAlign backward) also have nondeterministic CUDA kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
