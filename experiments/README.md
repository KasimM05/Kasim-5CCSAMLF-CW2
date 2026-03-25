# Experiments

- `run_baseline.py`: paper-faithful TypiClust baseline in the fully supervised framework.
- `run_improvement.py`: diversified TypiClust improvement in the fully supervised framework.
- `run_fully_supervised.py`: general fully supervised runner.
- `run_embedding_framework.py`: frozen-embedding + linear probe runner.
- `run_semi_supervised.py`: lightweight semi-supervised pseudo-label runner.
- `../tools/run_experiment_matrix.py`: batch launcher for the baseline matrix.

Recommended pattern:

```bash
python experiments/run_fully_supervised.py --cache-dir results/cache --output-dir results/baseline/fully_supervised/<run_name> ...
python experiments/run_embedding_framework.py --cache-dir results/cache --reuse-representation --reuse-embeddings --output-dir results/baseline/embedding/<run_name> ...
python experiments/run_semi_supervised.py --cache-dir results/cache --reuse-representation --reuse-embeddings --output-dir results/baseline/semi_supervised/<run_name> ...
python experiments/run_improvement.py --cache-dir results/cache --reuse-representation --reuse-embeddings --output-dir results/improvements/<run_name> ...
```

Keep baseline and modified runs separate so the report can compare them cleanly. The Colab and Kaggle notebooks at the repo root follow this exact cache-first workflow with platform-specific cache directories.

Matrix example:

```bash
python tools/run_experiment_matrix.py --frameworks fully_supervised --cache-dir results/cache --output-root results/baseline --reuse-representation --reuse-embeddings
```
