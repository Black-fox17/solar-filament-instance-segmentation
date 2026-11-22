"""Train the grayscale-adapted Mask R-CNN ResNet-50 FPN model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_filament.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
