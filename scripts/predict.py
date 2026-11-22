"""Run tiled full-resolution inference and generate a validated submission CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_filament.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
