"""Train and evaluate TPCRP active-learning experiments on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from .acquisition import (
    SELECTOR_NAMES,
    AcquisitionState,
    QuerySelector,
    build_query_selector,
)
from .models import LinearClassifier, ResNet18Classifier, SimCLRModel, reset_parameters
from .utils import (
    build_embedding_subset,
    build_eval_transform,
    build_supervised_subset,
    create_loader,
    evaluate_classifier,
    evaluate_embedding_classifier,
    extract_embeddings,
    extract_model_features,
    format_metrics,
    get_device,
    load_cifar10_datasets,
    predict_mc_probabilities,
    predict_probabilities,
    set_seed,
    split_labeled_unlabeled,
    train_classifier_epoch,
    train_embedding_classifier_epoch,
    train_semi_supervised_epoch,
    train_simclr_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
FRAMEWORK_NAMES = ["fully_supervised", "embedding", "semi_supervised"]


@dataclass
class QueryArtifacts:
    features: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    mc_probabilities: Optional[np.ndarray] = None


def parse_args(default_selector: str = "typiclust", default_framework: str = "fully_supervised") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TPCRP active-learning experiments on CIFAR-10.")
    parser.add_argument("--framework", choices=FRAMEWORK_NAMES, default=default_framework)
    parser.add_argument("--selector", choices=SELECTOR_NAMES, default=default_selector)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "results" / "cache")
    parser.add_argument("--initial-labeled-size", type=int, default=0)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--simclr-epochs", type=int, default=500)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--semi-supervised-epochs", type=int, default=20)
    parser.add_argument("--simclr-batch-size", type=int, default=512)
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
    parser.add_argument("--simclr-checkpoint-interval", type=int, default=5)
    parser.add_argument("--debug-subset", type=int, default=None)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-representation", dest="reuse_representation", action="store_true")
    parser.add_argument("--force-retrain-representation", dest="reuse_representation", action="store_false")
    parser.add_argument("--reuse-embeddings", dest="reuse_embeddings", action="store_true")
    parser.add_argument("--force-recompute-embeddings", dest="reuse_embeddings", action="store_false")
    parser.set_defaults(show_progress=True, reuse_representation=True, reuse_embeddings=True)
    return parser.parse_args()


def _format_float(value: float) -> str:
    return str(value).replace(".", "p")


def resolve_default_output_dir(framework_name: str, selector_name: str) -> Path:
    if selector_name in {"typiclust", "diversified"}:
        root = "baseline" if selector_name == "typiclust" else "improvements"
    else:
        root = "baseline"
    return PROJECT_ROOT / "results" / root / framework_name / selector_name


def build_active_pool_indices(dataset_size: int, debug_subset: Optional[int], seed: int) -> list[int]:
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


def build_query_selector_from_args(args: argparse.Namespace) -> QuerySelector:
    return build_query_selector(
        selector_name=args.selector,
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
        knn_k=args.knn_k,
        seed=args.seed,
        diversity_weight=args.diversity_weight,
    )


def representation_cache_dir(args: argparse.Namespace) -> Path:
    subset_tag = f"subset{args.debug_subset}" if args.debug_subset is not None else "full"
    cache_name = (
        f"simclr_{subset_tag}_seed{args.seed}_e{args.simclr_epochs}_bs{args.simclr_batch_size}"
        f"_lr{_format_float(args.simclr_lr)}_temp{_format_float(args.temperature)}"
    )
    return args.cache_dir / cache_name


def _simclr_metadata(args: argparse.Namespace) -> dict[str, float | int | None]:
    return {
        "seed": args.seed,
        "simclr_epochs": args.simclr_epochs,
        "simclr_batch_size": args.simclr_batch_size,
        "simclr_lr": args.simclr_lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "debug_subset": args.debug_subset,
    }


def save_simclr_resume_checkpoint(
    resume_path: Path,
    model: SimCLRModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> Path:
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_dim": model.feature_dim,
        "projection_dim": model.projector.layers[-1].out_features,
        "epoch": epoch,
        "metadata": _simclr_metadata(args),
    }
    torch.save(payload, resume_path)
    return resume_path


def load_simclr_resume_checkpoint(
    resume_path: Path,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> tuple[SimCLRModel, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]:
    payload = torch.load(resume_path, map_location=device, weights_only=False)
    projection_dim = int(payload.get("projection_dim", 128))
    model = SimCLRModel(projection_dim=projection_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    start_epoch = int(payload.get("epoch", 0))
    return model, optimizer, scheduler, start_epoch


def train_representation(
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    device: torch.device,
    num_workers: int,
    show_progress: bool,
    progress_prefix: str,
    resume_path: Path,
    checkpoint_interval: int,
    args: argparse.Namespace,
    reuse_existing: bool,
) -> SimCLRModel:
    loader = create_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = SimCLRModel().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    start_epoch = 0

    if resume_path.exists() and reuse_existing:
        model, optimizer, scheduler, start_epoch = load_simclr_resume_checkpoint(
            resume_path=resume_path,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
        )
        print(f"Resuming SimCLR from epoch {start_epoch:03d} using {resume_path}")

    print(f"SimCLR dataset size: {len(dataset)}")
    for epoch in range(start_epoch + 1, epochs + 1):
        loss = train_simclr_epoch(
            model,
            loader,
            optimizer,
            device,
            temperature=temperature,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix} e{epoch:03d}",
        )
        scheduler.step()
        print(f"{progress_prefix} epoch {epoch:03d} | loss={loss:.4f}")
        if checkpoint_interval > 0 and (epoch % checkpoint_interval == 0 or epoch == epochs):
            save_simclr_resume_checkpoint(
                resume_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
            )

    return model


def save_simclr_checkpoint(cache_dir: Path, model: SimCLRModel, args: argparse.Namespace) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "simclr_checkpoint.pt"
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": model.feature_dim,
        "projection_dim": model.projector.layers[-1].out_features,
        "metadata": _simclr_metadata(args),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_simclr_checkpoint(checkpoint_path: Path, device: torch.device) -> SimCLRModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    projection_dim = int(payload.get("projection_dim", 128))
    model = SimCLRModel(projection_dim=projection_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def prepare_representation_artifacts(
    args: argparse.Namespace,
    device: torch.device,
    simclr_pool: Dataset,
    active_eval_pool: Dataset,
    test_dataset: Dataset,
    active_pool_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, Path]:
    cache_dir = representation_cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "simclr_checkpoint.pt"
    resume_path = cache_dir / "simclr_resume.pt"
    train_embeddings_path = cache_dir / "train_embeddings.npy"
    test_embeddings_path = cache_dir / "test_embeddings.npy"
    pool_indices_path = cache_dir / "active_pool_indices.npy"
    metadata_path = cache_dir / "metadata.json"

    simclr_model = None
    if checkpoint_path.exists() and args.reuse_representation:
        print(f"Loading cached SimCLR checkpoint from {checkpoint_path}")
        simclr_model = load_simclr_checkpoint(checkpoint_path, device)
    else:
        print(f"Training SimCLR representation and caching into {cache_dir}")
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
            progress_prefix="SimCLR",
            resume_path=resume_path,
            checkpoint_interval=args.simclr_checkpoint_interval,
            args=args,
            reuse_existing=args.reuse_representation,
        )
        save_simclr_checkpoint(cache_dir, simclr_model, args)
        if resume_path.exists():
            resume_path.unlink()

    need_train_embeddings = not (train_embeddings_path.exists() and args.reuse_embeddings)
    need_test_embeddings = not (test_embeddings_path.exists() and args.reuse_embeddings)
    if need_train_embeddings or need_test_embeddings:
        if simclr_model is None:
            simclr_model = load_simclr_checkpoint(checkpoint_path, device)
        if need_train_embeddings:
            print(f"Extracting cached pool embeddings into {train_embeddings_path}")
            train_embeddings = extract_embeddings(
                simclr_model,
                active_eval_pool,
                device=device,
                batch_size=args.simclr_batch_size,
                num_workers=args.num_workers,
                show_progress=args.show_progress,
                progress_desc="Pool Embeddings",
            )
            np.save(train_embeddings_path, train_embeddings)
        if need_test_embeddings:
            print(f"Extracting cached test embeddings into {test_embeddings_path}")
            test_embeddings = extract_embeddings(
                simclr_model,
                test_dataset,
                device=device,
                batch_size=args.simclr_batch_size,
                num_workers=args.num_workers,
                show_progress=args.show_progress,
                progress_desc="Test Embeddings",
            )
            np.save(test_embeddings_path, test_embeddings)

    np.save(pool_indices_path, np.asarray(active_pool_indices, dtype=np.int64))
    write_json(
        metadata_path,
        {
            "seed": args.seed,
            "debug_subset": args.debug_subset,
            "simclr_epochs": args.simclr_epochs,
            "simclr_batch_size": args.simclr_batch_size,
            "simclr_lr": args.simclr_lr,
            "temperature": args.temperature,
            "active_pool_size": len(active_pool_indices),
        },
    )

    return np.load(train_embeddings_path), np.load(test_embeddings_path), cache_dir


def train_fully_supervised_classifier(
    args: argparse.Namespace,
    base_train: Dataset,
    labeled_dataset_indices: list[int],
    test_dataset: Dataset,
    device: torch.device,
    round_id: int,
) -> tuple[ResNet18Classifier, list[dict[str, float]]]:
    model = ResNet18Classifier(dropout_p=args.dropout_p).to(device)
    reset_parameters(model)
    train_subset = build_supervised_subset(base_train, labeled_dataset_indices, train=True)
    train_loader = create_loader(
        train_subset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.classifier_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.classifier_epochs, 1))
    history: list[dict[str, float]] = []

    print(f"Fully supervised train set size: {len(train_subset)} | test set size: {len(test_dataset)}")
    for epoch in range(1, args.classifier_epochs + 1):
        train_metrics = train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            show_progress=args.show_progress,
            progress_desc=f"Sup Train r{round_id:02d} e{epoch:03d}",
        )
        test_metrics = evaluate_classifier(
            model,
            test_loader,
            criterion,
            device,
            show_progress=args.show_progress,
            progress_desc=f"Sup Eval  r{round_id:02d} e{epoch:03d}",
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


def train_embedding_classifier(
    args: argparse.Namespace,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    labeled_indices: list[int],
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    round_id: int,
) -> tuple[LinearClassifier, list[dict[str, float]]]:
    model = LinearClassifier(
        in_dim=train_embeddings.shape[1],
        num_classes=int(test_labels.max()) + 1,
        dropout_p=args.dropout_p,
    ).to(device)
    reset_parameters(model)
    train_subset = build_embedding_subset(train_embeddings, train_labels, labeled_indices)
    test_subset = build_embedding_subset(test_embeddings, test_labels, list(range(len(test_labels))))
    train_loader = create_loader(
        train_subset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = create_loader(
        test_subset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.classifier_lr * 10.0,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.classifier_epochs, 1))
    history: list[dict[str, float]] = []

    print(f"Embedding framework train set size: {len(train_subset)} | test set size: {len(test_subset)}")
    for epoch in range(1, args.classifier_epochs + 1):
        train_metrics = train_embedding_classifier_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            show_progress=args.show_progress,
            progress_desc=f"Emb Train r{round_id:02d} e{epoch:03d}",
        )
        test_metrics = evaluate_embedding_classifier(
            model,
            test_loader,
            criterion,
            device,
            show_progress=args.show_progress,
            progress_desc=f"Emb Eval  r{round_id:02d} e{epoch:03d}",
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


def train_semi_supervised_classifier(
    args: argparse.Namespace,
    base_train: Dataset,
    labeled_dataset_indices: list[int],
    unlabeled_dataset_indices: list[int],
    test_dataset: Dataset,
    device: torch.device,
    round_id: int,
) -> tuple[ResNet18Classifier, list[dict[str, float]]]:
    model = ResNet18Classifier(dropout_p=args.dropout_p).to(device)
    reset_parameters(model)
    labeled_subset = build_supervised_subset(base_train, labeled_dataset_indices, train=True)
    unlabeled_subset = build_supervised_subset(base_train, unlabeled_dataset_indices, train=True)
    labeled_loader = create_loader(
        labeled_subset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    unlabeled_loader = create_loader(
        unlabeled_subset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.classifier_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.semi_supervised_epochs, 1))
    history: list[dict[str, float]] = []

    print(
        f"Semi-supervised labeled size: {len(labeled_subset)} | "
        f"unlabeled size: {len(unlabeled_subset)} | test set size: {len(test_dataset)}"
    )
    for epoch in range(1, args.semi_supervised_epochs + 1):
        if len(unlabeled_subset) > 0:
            train_metrics = train_semi_supervised_epoch(
                model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                criterion,
                device,
                unsupervised_weight=args.semi_supervised_weight,
                confidence_threshold=args.semi_supervised_threshold,
                show_progress=args.show_progress,
                progress_desc=f"Semi Train r{round_id:02d} e{epoch:03d}",
            )
        else:
            train_metrics = train_classifier_epoch(
                model,
                labeled_loader,
                optimizer,
                criterion,
                device,
                show_progress=args.show_progress,
                progress_desc=f"Semi Train r{round_id:02d} e{epoch:03d}",
            )
        test_metrics = evaluate_classifier(
            model,
            test_loader,
            criterion,
            device,
            show_progress=args.show_progress,
            progress_desc=f"Semi Eval  r{round_id:02d} e{epoch:03d}",
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


def collect_image_model_query_artifacts(
    args: argparse.Namespace,
    model: ResNet18Classifier,
    active_eval_pool: Dataset,
    selector: QuerySelector,
    device: torch.device,
) -> QueryArtifacts:
    features = None
    if selector.requires_features:
        features = extract_model_features(
            model,
            active_eval_pool,
            device=device,
            batch_size=args.classifier_batch_size,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            progress_desc="Query Features",
        )

    probabilities = None
    if selector.requires_probabilities:
        probabilities = predict_probabilities(
            model,
            active_eval_pool,
            device=device,
            batch_size=args.classifier_batch_size,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            progress_desc="Query Probs",
        )

    mc_probabilities = None
    if selector.requires_mc_probabilities:
        mc_probabilities = predict_mc_probabilities(
            model,
            active_eval_pool,
            device=device,
            passes=args.mc_dropout_passes,
            batch_size=args.classifier_batch_size,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            progress_desc="Query MC",
        )

    return QueryArtifacts(features=features, probabilities=probabilities, mc_probabilities=mc_probabilities)


def collect_embedding_query_artifacts(
    args: argparse.Namespace,
    model: LinearClassifier,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    selector: QuerySelector,
    device: torch.device,
) -> QueryArtifacts:
    pool_subset = build_embedding_subset(train_embeddings, train_labels, list(range(len(train_labels))))
    probabilities = None
    if selector.requires_probabilities:
        probabilities = predict_probabilities(
            model,
            pool_subset,
            device=device,
            batch_size=args.classifier_batch_size,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            progress_desc="Emb Query Probs",
        )

    mc_probabilities = None
    if selector.requires_mc_probabilities:
        mc_probabilities = predict_mc_probabilities(
            model,
            pool_subset,
            device=device,
            passes=args.mc_dropout_passes,
            batch_size=args.classifier_batch_size,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            progress_desc="Emb Query MC",
        )

    features = train_embeddings if selector.requires_features else None
    return QueryArtifacts(features=features, probabilities=probabilities, mc_probabilities=mc_probabilities)


def bootstrap_query_artifacts(
    args: argparse.Namespace,
    selector: QuerySelector,
    base_train: Dataset,
    active_eval_pool: Dataset,
    test_dataset: Dataset,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    labeled_indices: list[int],
    unlabeled_indices: list[int],
    device: torch.device,
) -> QueryArtifacts:
    default_features = train_embeddings if selector.requires_features else None
    if not labeled_indices:
        return QueryArtifacts(features=default_features)

    if args.framework == "embedding":
        model, _ = train_embedding_classifier(
            args=args,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            labeled_indices=labeled_indices,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            device=device,
            round_id=0,
        )
        return collect_embedding_query_artifacts(args, model, train_embeddings, train_labels, selector, device)

    if args.framework == "semi_supervised":
        model, _ = train_semi_supervised_classifier(
            args=args,
            base_train=base_train,
            labeled_dataset_indices=[int(index) for index in labeled_indices],
            unlabeled_dataset_indices=[int(index) for index in unlabeled_indices],
            test_dataset=test_dataset,
            device=device,
            round_id=0,
        )
        return collect_image_model_query_artifacts(args, model, active_eval_pool, selector, device)

    model, _ = train_fully_supervised_classifier(
        args=args,
        base_train=base_train,
        labeled_dataset_indices=[int(index) for index in labeled_indices],
        test_dataset=test_dataset,
        device=device,
        round_id=0,
    )
    return collect_image_model_query_artifacts(args, model, active_eval_pool, selector, device)


def main(default_selector: str = "typiclust", default_framework: str = "fully_supervised") -> None:
    args = parse_args(default_selector=default_selector, default_framework=default_framework)
    if args.output_dir is None:
        args.output_dir = resolve_default_output_dir(args.framework, args.selector)
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(describe_device(device))
    if args.debug_subset is not None:
        print(f"Debug subset enabled | active training pool={args.debug_subset}")

    base_train, simclr_train, test_dataset = load_cifar10_datasets(data_dir=args.data_dir)
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
    active_eval_pool = Subset(
        datasets.CIFAR10(
            root=str(args.data_dir),
            train=True,
            download=True,
            transform=build_eval_transform(),
        ),
        active_pool_indices,
    )

    train_embeddings, test_embeddings, cache_dir = prepare_representation_artifacts(
        args=args,
        device=device,
        simclr_pool=simclr_pool,
        active_eval_pool=active_eval_pool,
        test_dataset=test_dataset,
        active_pool_indices=active_pool_indices,
    )
    selector = build_query_selector_from_args(args)
    base_train_targets = np.asarray(getattr(base_train, "targets"), dtype=np.int64)
    active_pool_targets = base_train_targets[np.asarray(active_pool_indices)]
    test_targets = np.asarray(getattr(test_dataset, "targets"), dtype=np.int64)

    current_query_artifacts = bootstrap_query_artifacts(
        args=args,
        selector=selector,
        base_train=base_train,
        active_eval_pool=active_eval_pool,
        test_dataset=test_dataset,
        train_embeddings=train_embeddings,
        train_labels=active_pool_targets,
        test_embeddings=test_embeddings,
        test_labels=test_targets,
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        device=device,
    )
    if current_query_artifacts.features is None and selector.requires_features:
        current_query_artifacts.features = train_embeddings

    run_summary = {
        "framework": args.framework,
        "selector": args.selector,
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "cache_dir": str(cache_dir),
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
        acquisition_state = AcquisitionState(
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            embeddings=train_embeddings,
            features=current_query_artifacts.features,
            probabilities=current_query_artifacts.probabilities,
            mc_probabilities=current_query_artifacts.mc_probabilities,
            seed=args.seed + round_id - 1,
        )
        query_indices = selector.select(
            acquisition_state,
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

        if args.framework == "embedding":
            model, history = train_embedding_classifier(
                args=args,
                train_embeddings=train_embeddings,
                train_labels=active_pool_targets,
                labeled_indices=labeled_indices,
                test_embeddings=test_embeddings,
                test_labels=test_targets,
                device=device,
                round_id=round_id,
            )
            current_query_artifacts = collect_embedding_query_artifacts(
                args,
                model,
                train_embeddings,
                active_pool_targets,
                selector,
                device,
            )
        elif args.framework == "semi_supervised":
            model, history = train_semi_supervised_classifier(
                args=args,
                base_train=base_train,
                labeled_dataset_indices=labeled_dataset_indices,
                unlabeled_dataset_indices=[active_pool_indices[index] for index in unlabeled_indices],
                test_dataset=test_dataset,
                device=device,
                round_id=round_id,
            )
            current_query_artifacts = collect_image_model_query_artifacts(
                args,
                model,
                active_eval_pool,
                selector,
                device,
            )
        else:
            model, history = train_fully_supervised_classifier(
                args=args,
                base_train=base_train,
                labeled_dataset_indices=labeled_dataset_indices,
                test_dataset=test_dataset,
                device=device,
                round_id=round_id,
            )
            current_query_artifacts = collect_image_model_query_artifacts(
                args,
                model,
                active_eval_pool,
                selector,
                device,
            )

        if current_query_artifacts.features is None and selector.requires_features:
            current_query_artifacts.features = train_embeddings

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

    write_json(args.output_dir / "run_summary.json", run_summary)


if __name__ == "__main__":
    main()
