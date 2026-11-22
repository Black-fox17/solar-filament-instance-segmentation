"""Validate an existing competition submission CSV (schema, RLE decode, zero overlap)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_filament.cli import main

if __name__ == "__main__":
    main(["validate-submission", *sys.argv[1:]])
