"""Entry point for the embedding-linear active-learning framework."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tpcrp.train import main as run_experiment


def main() -> None:
    run_experiment(default_selector="typiclust", default_framework="embedding")


if __name__ == "__main__":
    main()
