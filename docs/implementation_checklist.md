# Algorithm Implementation Checklist

This file tracks whether the core "implement the algorithm" requirement is met.

## Paper-faithful TypiClust / TPCRP components

- SimCLR-style self-supervised representation learning: implemented
- ResNet-18 backbone built locally: implemented
- L2-normalized embedding extraction: implemented
- K-means / MiniBatchKMeans clustering: implemented
- Cluster balancing using the fewest-labeled-clusters rule: implemented
- Typicality via inverse mean k-nearest-neighbour distance: implemented
- Iterative active-learning selection loop: implemented
- CIFAR-10 data pipeline: implemented
- No author implementation imported into the repo: satisfied

## Coursework comparison infrastructure

- Fully supervised runner: implemented
- Embedding + linear probe runner: implemented
- Semi-supervised runner scaffold: implemented
- Shared cache for expensive SimCLR artefacts: implemented
- Colab notebook aligned to the cached workflow: implemented

## Remaining risks

- Full baseline matrix across all three frameworks still needs to be executed.
- The current semi-supervised path is a lightweight pseudo-label framework, not a full FlexMatch reproduction.
- Report-quality plots/tables still need to be generated from longer GPU runs.
