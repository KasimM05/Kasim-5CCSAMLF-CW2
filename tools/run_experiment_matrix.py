"""Run a baseline matrix across frameworks with shared cache settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAMEWORK_TO_ENTRYPOINT = {
    "fully_supervised": PROJECT_ROOT / "experiments" / "run_fully_supervised.py",
    "embedding": PROJECT_ROOT / "experiments" / "run_embedding_framework.py",
    "semi_supervised": PROJECT_ROOT / "experiments" / "run_semi_supervised.py",
}

BASELINE_SELECTORS = [
    "random",
    "uncertainty",
    "margin",
    "entropy",
    "dbal",
    "coreset",
    "bald",
    "badge",
    "typiclust",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the coursework baseline matrix.")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=list(FRAMEWORK_TO_ENTRYPOINT),
        default=["fully_supervised"],
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=BASELINE_SELECTORS,
        help="Acquisition strategies to run. Defaults to the coursework baseline set plus typiclust.",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "baseline")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "results" / "cache")
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--initial-labeled-size", type=int, default=0)
    parser.add_argument("--simclr-epochs", type=int, default=500)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--semi-supervised-epochs", type=int, default=20)
    parser.add_argument("--simclr-batch-size", type=int, default=256)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--simclr-lr", type=float, default=0.4)
    parser.add_argument("--classifier-lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-clusters", type=int, default=500)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--diversity-weight", type=float, default=0.35)
    parser.add_argument("--semi-supervised-weight", type=float, default=0.5)
    parser.add_argument("--semi-supervised-threshold", type=float, default=0.95)
    parser.add_argument("--mc-dropout-passes", type=int, default=10)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--debug-subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")
    parser.add_argument("--reuse-representation", dest="reuse_representation", action="store_true")
    parser.add_argument("--force-retrain-representation", dest="reuse_representation", action="store_false")
    parser.add_argument("--reuse-embeddings", dest="reuse_embeddings", action="store_true")
    parser.add_argument("--force-recompute-embeddings", dest="reuse_embeddings", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(show_progress=True, reuse_representation=True, reuse_embeddings=True)
    return parser.parse_args()


def build_base_command(args: argparse.Namespace, framework: str, selector: str) -> list[str]:
    entrypoint = FRAMEWORK_TO_ENTRYPOINT[framework]
    output_dir = args.output_root / framework / selector
    command = [
        sys.executable,
        str(entrypoint),
        "--selector",
        selector,
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(output_dir),
        "--query-size",
        str(args.query_size),
        "--rounds",
        str(args.rounds),
        "--initial-labeled-size",
        str(args.initial_labeled_size),
        "--simclr-epochs",
        str(args.simclr_epochs),
        "--classifier-epochs",
        str(args.classifier_epochs),
        "--semi-supervised-epochs",
        str(args.semi_supervised_epochs),
        "--simclr-batch-size",
        str(args.simclr_batch_size),
        "--classifier-batch-size",
        str(args.classifier_batch_size),
        "--simclr-lr",
        str(args.simclr_lr),
        "--classifier-lr",
        str(args.classifier_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--temperature",
        str(args.temperature),
        "--max-clusters",
        str(args.max_clusters),
        "--min-cluster-size",
        str(args.min_cluster_size),
        "--knn-k",
        str(args.knn_k),
        "--diversity-weight",
        str(args.diversity_weight),
        "--semi-supervised-weight",
        str(args.semi_supervised_weight),
        "--semi-supervised-threshold",
        str(args.semi_supervised_threshold),
        "--mc-dropout-passes",
        str(args.mc_dropout_passes),
        "--dropout-p",
        str(args.dropout_p),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.debug_subset is not None:
        command.extend(["--debug-subset", str(args.debug_subset)])
    command.append("--show-progress" if args.show_progress else "--no-show-progress")
    command.append("--reuse-representation" if args.reuse_representation else "--force-retrain-representation")
    command.append("--reuse-embeddings" if args.reuse_embeddings else "--force-recompute-embeddings")
    return command


def main() -> int:
    args = parse_args()
    failures: list[tuple[str, str, int]] = []

    for framework in args.frameworks:
        for selector in args.selectors:
            command = build_base_command(args, framework=framework, selector=selector)
            print(f"\n=== Running {framework} / {selector} ===")
            print(" ".join(command))
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                failures.append((framework, selector, result.returncode))
                if not args.continue_on_error:
                    break
        if failures and not args.continue_on_error:
            break

    if failures:
        print("\nFailures:")
        for framework, selector, code in failures:
            print(f"- {framework} / {selector}: exit code {code}")
        return 1

    print("\nMatrix run completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
