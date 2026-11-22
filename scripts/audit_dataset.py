"""Run dataset integrity, morphology, and patch-coverage audits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_filament.cli import main

if __name__ == "__main__":
    main(["audit", *sys.argv[1:]])
