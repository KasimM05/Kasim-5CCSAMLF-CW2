"""Compatibility wrapper for the modified TPCRP experiment."""

from tpcrp.improvements import DiversifiedTPCRPSelector
from tpcrp.train import main as run_experiment

__all__ = ["DiversifiedTPCRPSelector", "main"]


def main() -> None:
    run_experiment(default_selector="diversified", default_framework="fully_supervised")


if __name__ == "__main__":
    main()
