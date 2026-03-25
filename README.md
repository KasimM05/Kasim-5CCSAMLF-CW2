# Kasim-5CCSAMLF-CW2

Coursework repository for reproducing TypiClust / TPCRP-style active learning on CIFAR-10, writing the report, and evaluating a modified improvement.

## Coursework Rules We Need To Respect

- Implement all baselines across the three required frameworks, not just the random baseline.
- Reproduce the paper results to the best of our ability.
- Do not use implementation code from the paper authors or other external implementations.
- Reuse of older standard building blocks is allowed, for example an untrained `ResNet-18`, as long as we train and integrate it ourselves.
- The improvement section may change more than one aspect of the algorithm, but the report must justify and analyse those changes clearly.

The in-repo compliance notes live in `docs/coursework_rules.md`.

## Current Repo Layout

- `tpcrp/`: core implementation package for models, acquisition strategies, utilities, and training.
- `experiments/`: runnable entry points for baseline/improvement runs and the three frameworks.
- `results/`: organised output location for baseline and improvement artefacts.
- `reports/`: LaTeX report draft.
- `docs/`: coursework constraints and process notes.
- `data/`: local CIFAR-10 cache.
- `tools/`: maintenance and orchestration scripts, including notebook regeneration and matrix launching.
- `tpcrp_colab.ipynb`: Colab notebook for GPU runs with Drive-backed cache reuse.
- `tpcrp_kaggle.ipynb`: Kaggle notebook for GPU runs under `/kaggle/working`.

## What Is Implemented Now

- A paper-faithful TypiClust / TPCRP-style selection pipeline in `tpcrp/selector.py`.
- A selector registry in `tpcrp/acquisition.py` covering the main comparison methods used in the paper workflow: random, uncertainty, margin, entropy, DBAL, CoreSet, BALD, BADGE, TypiClust, and the modified diversified variant.
- A cached SimCLR pipeline in `tpcrp/train.py` that can save and reuse representation checkpoints and embedding arrays.
- Three runnable framework paths:
  - fully supervised;
  - embedding + linear probe;
  - lightweight semi-supervised pseudo-label runner.
- A first improvement variant in `tpcrp/improvements.py` that keeps cluster balancing but adds an intra-cluster diversity term.
- Compatibility wrappers at the repo root so older commands such as `python train.py` still work.

## Implementation Marks Check

For the coursework's implementation-focused marks, the core algorithm work is now in place:

- self-supervised representation learning;
- feature extraction on the unlabeled pool;
- K-means / MiniBatchKMeans clustering;
- cluster balancing via the fewest-labeled-clusters rule;
- typicality scoring via inverse mean k-nearest-neighbour distance;
- iterative active-learning querying;
- all implemented locally in this repository without importing author code.

That means the project is in a credible state for the "implement the algorithm" component. What is still missing for full coursework strength is the experimental coverage and report evidence, especially the complete baseline matrix across all three frameworks.

## Current Gap Against The Coursework

The codebase is now organised for baseline and improvement work, and the runner infrastructure can execute the required framework/selector combinations. However, the full matrix of required baselines across all three frameworks has not yet been completed with report-ready runs.

One more important caveat: the current semi-supervised runner is a lightweight pseudo-label framework, not a faithful FlexMatch reproduction. If strict paper-level parity is required for that framework, FlexMatch-equivalent training is still a remaining task.

## Running Experiments

Representation precompute for Colab/GPU reuse:

```bash
python experiments/run_fully_supervised.py --selector typiclust --rounds 0 --simclr-epochs 500 --cache-dir results/cache
```

Baseline matrix launcher:

```bash
python tools/run_experiment_matrix.py --frameworks fully_supervised --cache-dir results/cache --output-root results/baseline --reuse-representation --reuse-embeddings
```

Fully supervised smoke test:

```bash
python experiments/run_fully_supervised.py --selector typiclust --debug-subset 256 --query-size 10 --rounds 1 --simclr-epochs 1 --classifier-epochs 1 --simclr-batch-size 128 --classifier-batch-size 128 --num-workers 0
```

Embedding framework run:

```bash
python experiments/run_embedding_framework.py --selector typiclust --query-size 10 --rounds 5 --simclr-epochs 500 --classifier-epochs 20 --cache-dir results/cache --reuse-representation --reuse-embeddings
```

Semi-supervised runner:

```bash
python experiments/run_semi_supervised.py --selector typiclust --query-size 10 --rounds 5 --simclr-epochs 500 --semi-supervised-epochs 20 --cache-dir results/cache --reuse-representation --reuse-embeddings
```

Modified improvement run:

```bash
python experiments/run_improvement.py --query-size 10 --rounds 5 --simclr-epochs 500 --classifier-epochs 20 --cache-dir results/cache --reuse-representation --reuse-embeddings --diversity-weight 0.35
```

If `--output-dir` is omitted, runs are written under `results/baseline/` or `results/improvements/` automatically. Both the Colab and Kaggle notebooks at the repo root use this cached workflow, with platform-specific storage paths.
