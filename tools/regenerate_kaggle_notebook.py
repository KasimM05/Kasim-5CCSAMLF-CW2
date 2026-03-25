"""Regenerate the Kaggle notebook from a source-of-truth cell list."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "tpcrp_kaggle.ipynb"


def build_notebook() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TPCRP / TypiClust Kaggle Runner\n",
                "\n",
                "This notebook is structured to be run top-to-bottom on Kaggle: validate the environment, run a quick smoke test, build the SimCLR cache once, execute the baseline frameworks, and finally run the improvement experiment.\n",
                "\n",
                "The long-run cells use a deadline-safe reduced configuration: 200 SimCLR epochs, 3 active-learning rounds, and shorter classifier training. The full baseline set is still kept because the coursework Q&A expects all baselines across all three frameworks.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Clone the repo\n",
                "\n",
                "Enable a GPU in the notebook settings before running these cells. If Kaggle internet is disabled, enable it first so the repository can be cloned.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import shutil\n",
                "import subprocess\n",
                "\n",
                "REPO_URL = \"https://github.com/KasimM05/Kasim-5CCSAMLF-CW2\"\n",
                "REPO_DIR = Path(\"/kaggle/working/Kasim-5CCSAMLF-CW2\")\n",
                "CACHE_DIR = Path(\"/kaggle/working/tpcrp_cache\")\n",
                "RUNS_DIR = Path(\"/kaggle/working/tpcrp_runs\")\n",
                "\n",
                "if REPO_DIR.exists():\n",
                "    shutil.rmtree(REPO_DIR)\n",
                "subprocess.run([\"git\", \"clone\", REPO_URL, str(REPO_DIR)], check=True)\n",
                "CACHE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "RUNS_DIR.mkdir(parents=True, exist_ok=True)\n",
                "print(f\"Repo: {REPO_DIR}\")\n",
                "print(f\"Cache: {CACHE_DIR}\")\n",
                "print(f\"Runs: {RUNS_DIR}\")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Install dependencies\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /kaggle/working/Kasim-5CCSAMLF-CW2\n",
                "%pip install -r requirements.txt\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Check the Kaggle GPU\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "\n",
                "print(\"CUDA available:\", torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print(\"GPU:\", torch.cuda.get_device_name(0))\n",
                "else:\n",
                "    print(\"GPU not detected. Stop here and fix the notebook settings.\")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Fast smoke test\n",
                "\n",
                "Run this first after setup. It should finish quickly and confirm that the end-to-end pipeline works before you spend GPU time on the long runs.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python experiments/run_fully_supervised.py \\\n",
                "    --selector typiclust \\\n",
                "    --debug-subset 256 \\\n",
                "    --query-size 10 \\\n",
                "    --rounds 1 \\\n",
                "    --simclr-epochs 1 \\\n",
                "    --classifier-epochs 1 \\\n",
                "    --simclr-batch-size 128 \\\n",
                "    --classifier-batch-size 128 \\\n",
                "    --num-workers 2 \\\n",
                "    --cache-dir \"$CACHE_DIR\" \\\n",
                "    --output-dir \"$RUNS_DIR/smoke/fully_supervised_typiclust\"\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. One-time SimCLR representation cache\n",
                "\n",
                "Run this once for the practical final configuration. Later framework runs reuse the same representation with `--reuse-representation --reuse-embeddings`. If Kaggle disconnects partway through, rerun this same cell and it will resume from the last saved SimCLR checkpoint.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python experiments/run_fully_supervised.py \\\n",
                "    --selector typiclust \\\n",
                "    --rounds 0 \\\n",
                "    --simclr-epochs 200 \\\n",
                "    --simclr-batch-size 128 \\\n",
                "    --classifier-batch-size 128 \\\n",
                "    --num-workers 2 \\\n",
                "    --simclr-checkpoint-interval 5 \\\n",
                "    --no-show-progress \\\n",
                "    --cache-dir \"$CACHE_DIR\" \\\n",
                "    --output-dir \"$RUNS_DIR/precompute/representation_only\"\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Fully supervised baseline matrix\n",
                "\n",
                "This is the first full coursework run to execute after the representation cache is ready. It keeps the full baseline set but uses reduced epochs and rounds so the coursework can be completed within a practical GPU budget.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python tools/run_experiment_matrix.py \\\n",
                "    --frameworks fully_supervised \\\n",
                "    --cache-dir \"$CACHE_DIR\" \\\n",
                "    --output-root \"$RUNS_DIR\" \\\n",
                "    --query-size 10 \\\n",
                "    --rounds 3 \\\n",
                "    --simclr-epochs 200 \\\n",
                "    --classifier-epochs 10 \\\n",
                "    --simclr-batch-size 128 \\\n",
                "    --classifier-batch-size 128 \\\n",
                "    --num-workers 2 \\\n",
                "    --continue-on-error \\\n",
                "    --no-show-progress \\\n",
                "    --reuse-representation \\\n",
                "    --reuse-embeddings\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Embedding baseline matrix\n",
                "\n",
                "Run this after the fully supervised matrix completes successfully.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python tools/run_experiment_matrix.py \\\n",
                "    --frameworks embedding \\\n",
                "    --cache-dir \"$CACHE_DIR\" \\\n",
                "    --output-root \"$RUNS_DIR\" \\\n",
                "    --query-size 10 \\\n",
                "    --rounds 3 \\\n",
                "    --simclr-epochs 200 \\\n",
                "    --classifier-epochs 10 \\\n",
                "    --simclr-batch-size 128 \\\n",
                "    --classifier-batch-size 128 \\\n",
                "    --num-workers 2 \\\n",
                "    --continue-on-error \\\n",
                "    --no-show-progress \\\n",
                "    --reuse-representation \\\n",
                "    --reuse-embeddings\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Semi-supervised baseline matrix\n",
                "\n",
                "Run this after the embedding matrix. This is usually the slowest framework.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python tools/run_experiment_matrix.py \\\n",
                "    --frameworks semi_supervised \\\n",
                "    --cache-dir \"$CACHE_DIR\" \\\n",
                "    --output-root \"$RUNS_DIR\" \\\n",
                "    --query-size 10 \\\n",
                "    --rounds 3 \\\n",
                "    --simclr-epochs 200 \\\n",
                "    --semi-supervised-epochs 5 \\\n",
                "    --simclr-batch-size 128 \\\n",
                "    --classifier-batch-size 128 \\\n",
                "    --num-workers 2 \\\n",
                "    --continue-on-error \\\n",
                "    --no-show-progress \\\n",
                "    --reuse-representation \\\n",
                "    --reuse-embeddings\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Improvement run\n",
                "\n",
                "Only start this after the baseline matrices are complete enough for comparison in the report.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python experiments/run_improvement.py \\\n",
                "    --selector diversified \\\n",
                "    --query-size 10 \\\n",
                "    --rounds 3 \\\n",
                "    --simclr-epochs 200 \\\n",
                "    --classifier-epochs 10 \\\n",
                "    --simclr-batch-size 128 \\\n",
                "    --classifier-batch-size 128 \\\n",
                "    --num-workers 2 \\\n",
                "    --no-show-progress \\\n",
                "    --cache-dir \"$CACHE_DIR\" \\\n",
                "    --reuse-representation \\\n",
                "    --reuse-embeddings \\\n",
                "    --output-dir \"$RUNS_DIR/improvements/diversified_full\"\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Inspect a run\n",
                "\n",
                "Point this at any completed run folder to inspect the summary and per-round metrics.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import pandas as pd\n",
                "\n",
                "output_dir = RUNS_DIR / \"fully_supervised\" / \"typiclust\"\n",
                "summary_path = output_dir / \"run_summary.json\"\n",
                "\n",
                "if summary_path.exists():\n",
                "    with summary_path.open(\"r\", encoding=\"utf-8\") as handle:\n",
                "        summary = json.load(handle)\n",
                "    print(json.dumps(summary, indent=2))\n",
                "else:\n",
                "    print(\"run_summary.json not found\")\n",
                "\n",
                "for csv_path in sorted(output_dir.glob(\"round_*_metrics.csv\")):\n",
                "    print(f\"\\n=== {csv_path.name} ===\")\n",
                "    display(pd.read_csv(csv_path))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Zip a result folder\n",
                "\n",
                "Use this before saving a Kaggle notebook version if you want a single archive under `/kaggle/working/`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "from IPython.display import FileLink\n",
                "\n",
                "target_dir = RUNS_DIR / \"fully_supervised\" / \"typiclust\"\n",
                "archive_path = shutil.make_archive(str(target_dir), \"zip\", root_dir=str(target_dir))\n",
                "print(archive_path)\n",
                "FileLink(archive_path)\n",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
