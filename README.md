# Kasim-5CCSAMLF-CW2

This repository contains the completed coursework project for reproducing a TPCRP / TypiClust-style active-learning pipeline on CIFAR-10, evaluating the required baselines across three frameworks, and testing a modified improvement.

## What To Read First

- Report source: [`main.tex`](main.tex)
- Report figure: [`reports/figures/report_summary.png`](reports/figures/report_summary.png)
- Core implementation: [`tpcrp/`](tpcrp)
- Experiment entry points: [`experiments/`](experiments)
- Stored artefacts layout: [`results/README.md`](results/README.md)


## Project Scope

The coursework addresses active learning for image classification, where the goal is to achieve useful performance with a small labelled set by selecting informative examples from a larger unlabeled pool. This project reproduces the paper workflow on CIFAR-10 under a reduced but matched training budget and compares:

- fully supervised retraining after each query round;
- frozen self-supervised embeddings with a linear probe;
- a lightweight semi-supervised scaffold using labelled and unlabeled data together.

The repository also contains a modified selector that augments TPCRP with an intra-cluster diversity term.

## Implemented Components

The codebase includes the following completed components:

- SimCLR-based self-supervised representation learning;
- embedding extraction for the active pool and test set;
- K-means / MiniBatchKMeans clustering;
- cluster balancing based on the fewest-labelled-cluster rule;
- local typicality scoring using k-nearest-neighbour density;
- iterative active-learning querying;
- a selector registry covering the required baselines:
  - random
  - uncertainty
  - margin
  - entropy
  - DBAL
  - CoreSet
  - BALD
  - BADGE
  - TPCRP / TypiClust
- a modified diversified TPCRP variant.

The implementation is local to this repository and does not reuse the paper authors' released code.

## Repository Layout

- [`main.tex`](main.tex): sole LaTeX report source.
- [`tpcrp/`](tpcrp): models, selector logic, improvement variant, utilities, and shared training pipeline.
- [`experiments/`](experiments): runnable entry points for each framework and the improvement experiment.
- [`tools/`](tools): notebook generation, notebook appendix export, and matrix orchestration scripts.
- [`results/`](results): organised experiment artefacts and cached representations.
- [`reports/`](reports): non-LaTeX report assets, currently the summary figure and printable notebook appendix HTML.
- [`docs/`](docs): coursework rule notes and implementation checklist.
- [`tpcrp_kaggle.ipynb`](tpcrp_kaggle.ipynb): clean Kaggle notebook used for the final workflow.
- [`tpcrp_colab.ipynb`](tpcrp_colab.ipynb): Colab version of the same workflow.
- [`tpcrp-kaggle (1).ipynb`](tpcrp-kaggle%20(1).ipynb): executed Kaggle notebook with outputs, retained for appendix generation.

## Final Workflow Used For Results

The final experimental workflow was:

1. validate the notebook environment with a smoke test;
2. precompute a single cached SimCLR representation;
3. run the fully supervised baseline matrix;
4. run the embedding baseline matrix;
5. run the semi-supervised baseline matrix;
6. run the diversified TPCRP improvement.

The main reduced configuration used for the completed runs is:

- `200` SimCLR epochs
- query size `10`
- `3` active-learning rounds
- `10` supervised / linear-probe epochs per round
- `5` semi-supervised epochs per round

This configuration is intentionally lighter than the original paper setup, but it is matched across methods and is the configuration discussed in the report.

## Key Files For Reproduction

- Fully supervised runner: [`experiments/run_fully_supervised.py`](experiments/run_fully_supervised.py)
- Embedding runner: [`experiments/run_embedding_framework.py`](experiments/run_embedding_framework.py)
- Semi-supervised runner: [`experiments/run_semi_supervised.py`](experiments/run_semi_supervised.py)
- Improvement runner: [`experiments/run_improvement.py`](experiments/run_improvement.py)
- Shared training and cache logic: [`tpcrp/train.py`](tpcrp/train.py)
- TPCRP / TypiClust selector: [`tpcrp/selector.py`](tpcrp/selector.py)
- Diversified selector: [`tpcrp/improvements.py`](tpcrp/improvements.py)

Example command structure:

```bash
python experiments/run_fully_supervised.py \
  --selector typiclust \
  --query-size 10 \
  --rounds 3 \
  --simclr-epochs 200 \
  --classifier-epochs 10 \
  --reuse-representation \
  --reuse-embeddings
```

## Results and Artefacts

The stored artefacts are organised as follows:

- [`results/cache/`](results/cache): cached SimCLR checkpoint and embeddings.
- [`results/precompute/`](results/precompute): precompute metadata.
- [`results/baseline/fully_supervised/`](results/baseline/fully_supervised): fully supervised baseline outputs.
- [`results/baseline/embedding/`](results/baseline/embedding): embedding-framework outputs.
- [`results/baseline/semi_supervised/`](results/baseline/semi_supervised): semi-supervised outputs.
- [`results/improvements/`](results/improvements): modified-selector outputs.

Each run directory contains lightweight summary artefacts such as `run_summary.json` and `round_XX_metrics.csv`. Large generated caches, checkpoints, arrays, and archives are ignored by Git.
