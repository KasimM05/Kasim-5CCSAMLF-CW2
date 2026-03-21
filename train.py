"""Train and evaluate the TPCRP active learning pipeline on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets

from models import ResNet18Classifier, SimCLRModel, reset_parameters
from tpcrp import TPCRPSelector
from utils import (
    build_eval_transform,
    build_supervised_subset,
    create_loader,
    evaluate_classifier,
    extract_embeddings,
    format_metrics,
    get_device,
    load_cifar10_datasets,
    set_seed,
    split_labeled_unlabeled,
    train_classifier_epoch,
    train_simclr_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPCRP on CIFAR-10.")
    parser.add_argument("--initial-labeled-size", type=int, default=0)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--simclr-epochs", type=int, default=500)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--simclr-batch-size", type=int, default=512)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--simclr-lr", type=float, default=0.4)
    parser.add_argument("--classifier-lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-clusters", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--debug-subset", type=int, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()



def build_active_pool_indices(dataset_size: int, debug_subset: int | None, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)
    if debug_subset is not None:
        indices = indices[: min(debug_subset, dataset_size)]
    return indices.tolist()



def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"Using device: cuda ({torch.cuda.get_device_name(0)})"
    return f"Using device: {device.type}"



def train_representation(
    dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    device: torch.device,
    num_workers: int,
    show_progress: bool,
    round_id: int,
) -> SimCLRModel:
    model = SimCLRModel().to(device)
    loader = create_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    print(f"SimCLR dataset size: {len(dataset)}")
    for epoch in range(1, epochs + 1):
        loss = train_simclr_epoch(
            model,
            loader,
            optimizer,
            device,
            temperature=temperature,
            show_progress=show_progress,
            progress_desc=f"SimCLR r{round_id:02d} e{epoch:03d}",
        )
        scheduler.step()
        print(f"SimCLR epoch {epoch:03d} | loss={loss:.4f}")

    return model



def train_supervised_classifier(
    train_dataset,
    test_dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
    show_progress: bool,
    round_id: int,
) -> tuple[ResNet18Classifier, list[dict[str, float]]]:
    model = ResNet18Classifier().to(device)
    reset_parameters(model)
    train_loader = create_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = create_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[dict[str, float]] = []

    print(f"Classifier train set size: {len(train_dataset)} | test set size: {len(test_dataset)}")
    for epoch in range(1, epochs + 1):
        train_metrics = train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            show_progress=show_progress,
            progress_desc=f"Train r{round_id:02d} e{epoch:03d}",
        )
        test_metrics = evaluate_classifier(
            model,
            test_loader,
            criterion,
            device,
            show_progress=show_progress,
            progress_desc=f"Eval  r{round_id:02d} e{epoch:03d}",
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "test_loss": test_metrics.loss,
                "test_accuracy": test_metrics.accuracy,
            }
        )
        print(format_metrics(epoch, train_metrics, test_metrics))

    return model, history



def write_round_history(output_dir: Path, round_id: int, history: list[dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"round_{round_id:02d}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)



def write_round_summary(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(describe_device(device))
    if args.debug_subset is not None:
        print(f"Debug subset enabled | active training pool={args.debug_subset}")

    base_train, simclr_train, test_dataset = load_cifar10_datasets()
    active_pool_indices = build_active_pool_indices(
        dataset_size=len(base_train),
        debug_subset=args.debug_subset,
        seed=args.seed,
    )
    active_pool_size = len(active_pool_indices)
    labeled_indices, unlabeled_indices = split_labeled_unlabeled(
        dataset_size=active_pool_size,
        initial_labeled_size=min(args.initial_labeled_size, active_pool_size),
        seed=args.seed,
    )

    simclr_pool = Subset(simclr_train, active_pool_indices)
    embedding_pool = Subset(
        datasets.CIFAR10(
            root=getattr(base_train, "root", "data"),
            train=True,
            download=True,
            transform=build_eval_transform(),
        ),
        active_pool_indices,
    )

    selector = TPCRPSelector(max_clusters=args.max_clusters, seed=args.seed)
    run_summary = {
        "seed": args.seed,
        "initial_labeled_size": args.initial_labeled_size,
        "query_size": args.query_size,
        "active_pool_size": active_pool_size,
        "rounds": [],
    }

    for round_id in range(1, args.rounds + 1):
        print(
            f"Round {round_id:02d} | labeled={len(labeled_indices)} | "
            f"unlabeled={len(unlabeled_indices)} | query={args.query_size}"
        )
        simclr_model = train_representation(
            dataset=simclr_pool,
            epochs=args.simclr_epochs,
            batch_size=args.simclr_batch_size,
            lr=args.simclr_lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            device=device,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            round_id=round_id,
        )
        embeddings = extract_embeddings(
            simclr_model,
            embedding_pool,
            device=device,
            batch_size=args.simclr_batch_size,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            progress_desc=f"Embed  r{round_id:02d}",
        )
        query_indices = selector.select(
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            budget=min(args.query_size, len(unlabeled_indices)),
        )

        if not query_indices:
            print("No additional samples were selected.")
            break

        selected_dataset_indices = [active_pool_indices[index] for index in query_indices]
        print(f"Selected pool indices: {query_indices}")
        print(f"Selected dataset indices: {selected_dataset_indices}")
        labeled_indices.extend(query_indices)
        selected_set = set(query_indices)
        unlabeled_indices = [index for index in unlabeled_indices if index not in selected_set]

        labeled_dataset_indices = [active_pool_indices[index] for index in labeled_indices]
        train_subset = build_supervised_subset(base_train, labeled_dataset_indices, train=True)
        _, history = train_supervised_classifier(
            train_dataset=train_subset,
            test_dataset=test_dataset,
            epochs=args.classifier_epochs,
            batch_size=args.classifier_batch_size,
            lr=args.classifier_lr,
            weight_decay=args.weight_decay,
            device=device,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            round_id=round_id,
        )
        write_round_history(args.output_dir, round_id, history)
        best_epoch = max(history, key=lambda item: item["test_accuracy"])
        run_summary["rounds"].append(
            {
                "round": round_id,
                "selected_pool_indices": query_indices,
                "selected_dataset_indices": selected_dataset_indices,
                "labeled_size_after_query": len(labeled_indices),
                "best_epoch": best_epoch,
                "final_epoch": history[-1],
            }
        )

    write_round_summary(args.output_dir, run_summary)


if __name__ == "__main__":
    main()
