"""Compatibility wrapper for the paper-faithful TypiClust experiment."""

from tpcrp.train import main


if __name__ == "__main__":
    main(default_selector="typiclust", default_framework="fully_supervised")
