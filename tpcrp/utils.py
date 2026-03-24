"""Utilities for CIFAR-10 TPCRP experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class IndexedDataset(Dataset):
    """Wrap a dataset so each sample also returns its original index."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if len(item) == 2:
            inputs, target = item
            return inputs, target, index
        return (*item, index)


class ArrayDataset(Dataset):
    """Dataset backed by numpy arrays for frozen embedding experiments."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.int64))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


class SimCLRViewTransform:
    """Produce two stochastic views of the same image."""

    def __init__(self) -> None:
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def build_supervised_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_cifar10_datasets(data_dir: str | Path = "data") -> tuple[Dataset, Dataset, Dataset]:
    base_train = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=None)
    simclr_train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=SimCLRViewTransform(),
    )
    eval_test = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=build_eval_transform(),
    )
    return base_train, simclr_train, eval_test



def build_supervised_subset(base_dataset: Dataset, indices: Sequence[int], train: bool) -> Dataset:
    dataset = datasets.CIFAR10(
        root=getattr(base_dataset, "root", "data"),
        train=getattr(base_dataset, "train", True),
        download=True,
        transform=build_supervised_train_transform() if train else build_eval_transform(),
    )
    return Subset(dataset, list(indices))



def create_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_embedding_subset(
    embeddings: np.ndarray,
    labels: Sequence[int],
    indices: Sequence[int],
) -> Dataset:
    subset_embeddings = embeddings[np.asarray(indices)]
    subset_labels = np.asarray(labels, dtype=np.int64)[np.asarray(indices)]
    return ArrayDataset(subset_embeddings, subset_labels)



def split_labeled_unlabeled(
    dataset_size: int,
    initial_labeled_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(dataset_size)
    rng.shuffle(all_indices)
    labeled = all_indices[:initial_labeled_size].tolist()
    unlabeled = all_indices[initial_labeled_size:].tolist()
    return labeled, unlabeled



def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()



def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity = representations @ representations.T
    similarity = similarity / temperature

    batch_size = z1.size(0)
    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    targets = torch.arange(batch_size, device=similarity.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(similarity, targets)



def train_simclr_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.5,
    show_progress: bool = True,
    progress_desc: str = "SimCLR",
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=progress_desc, leave=False, disable=not show_progress)

    for step, batch in enumerate(progress, start=1):
        if len(batch) == 3:
            (view1, view2), _, _ = batch
        elif len(batch) == 2:
            (view1, view2), _ = batch
        else:
            raise ValueError(f"Unexpected SimCLR batch format with {len(batch)} items.")

        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, proj1 = model(view1)
        _, proj2 = model(view2)
        loss = nt_xent_loss(proj1, proj2, temperature=temperature)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{running_loss / step:.4f}")

    return running_loss / max(len(loader), 1)



def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str = "Classifier Train",
) -> EpochMetrics:
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    progress = tqdm(loader, desc=progress_desc, leave=False, disable=not show_progress)

    for step, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += _accuracy(logits, targets)
        progress.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{running_accuracy / step:.4f}",
        )

    steps = max(len(loader), 1)
    return EpochMetrics(running_loss / steps, running_accuracy / steps)


def train_embedding_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str = "Embedding Train",
) -> EpochMetrics:
    return train_classifier_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str = "Classifier Eval",
) -> EpochMetrics:
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    progress = tqdm(loader, desc=progress_desc, leave=False, disable=not show_progress)

    for step, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        running_loss += loss.item()
        running_accuracy += _accuracy(logits, targets)
        progress.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{running_accuracy / step:.4f}",
        )

    steps = max(len(loader), 1)
    return EpochMetrics(running_loss / steps, running_accuracy / steps)


def evaluate_embedding_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str = "Embedding Eval",
) -> EpochMetrics:
    return evaluate_classifier(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
    show_progress: bool = True,
    progress_desc: str = "Embeddings",
) -> np.ndarray:
    model.eval()
    indexed_dataset = IndexedDataset(dataset)
    loader = create_loader(indexed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embedding_dim = getattr(model, "feature_dim", 512)
    embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)
    progress = tqdm(loader, desc=progress_desc, leave=False, disable=not show_progress)

    for inputs, _, indices in progress:
        inputs = inputs.to(device, non_blocking=True)
        features = model.encode(inputs)
        features = F.normalize(features, dim=1)
        embeddings[indices.numpy()] = features.cpu().numpy()

    return embeddings


@torch.no_grad()
def extract_model_features(
    model: Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
    show_progress: bool = True,
    progress_desc: str = "Features",
) -> np.ndarray:
    model.eval()
    indexed_dataset = IndexedDataset(dataset)
    loader = create_loader(indexed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feature_dim = getattr(model, "feature_dim", 512)
    features = np.zeros((len(dataset), feature_dim), dtype=np.float32)
    progress = tqdm(loader, desc=progress_desc, leave=False, disable=not show_progress)

    for inputs, _, indices in progress:
        inputs = inputs.to(device, non_blocking=True)
        batch_features = model.encode(inputs)
        features[indices.numpy()] = batch_features.cpu().numpy()

    return features


@torch.no_grad()
def predict_probabilities(
    model: Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
    show_progress: bool = True,
    progress_desc: str = "Predict",
) -> np.ndarray:
    model.eval()
    indexed_dataset = IndexedDataset(dataset)
    loader = create_loader(indexed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities = None
    progress = tqdm(loader, desc=progress_desc, leave=False, disable=not show_progress)

    for inputs, _, indices in progress:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        if probabilities is None:
            probabilities = np.zeros((len(dataset), batch_probabilities.shape[1]), dtype=np.float32)
        probabilities[indices.numpy()] = batch_probabilities

    if probabilities is None:
        return np.zeros((0, 0), dtype=np.float32)
    return probabilities


@torch.no_grad()
def predict_mc_probabilities(
    model: Module,
    dataset: Dataset,
    device: torch.device,
    passes: int,
    batch_size: int = 256,
    num_workers: int = 2,
    show_progress: bool = True,
    progress_desc: str = "MC Predict",
) -> np.ndarray:
    if passes <= 0:
        return np.zeros((0, len(dataset), 0), dtype=np.float32)

    indexed_dataset = IndexedDataset(dataset)
    loader = create_loader(indexed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_passes = []

    for pass_id in range(1, passes + 1):
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        probabilities = None
        progress = tqdm(
            loader,
            desc=f"{progress_desc} p{pass_id:02d}",
            leave=False,
            disable=not show_progress,
        )
        for inputs, _, indices in progress:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            if probabilities is None:
                probabilities = np.zeros((len(dataset), batch_probabilities.shape[1]), dtype=np.float32)
            probabilities[indices.numpy()] = batch_probabilities
        if probabilities is None:
            probabilities = np.zeros((len(dataset), 0), dtype=np.float32)
        all_passes.append(probabilities)

    model.eval()
    return np.stack(all_passes, axis=0)


def train_semi_supervised_epoch(
    model: Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    unsupervised_weight: float = 0.5,
    confidence_threshold: float = 0.95,
    show_progress: bool = True,
    progress_desc: str = "Semi-Supervised Train",
) -> EpochMetrics:
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    step_count = 0
    unlabeled_iter = iter(unlabeled_loader)
    progress = tqdm(labeled_loader, desc=progress_desc, leave=False, disable=not show_progress)

    for labeled_inputs, labeled_targets in progress:
        try:
            unlabeled_inputs, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_inputs, _ = next(unlabeled_iter)

        labeled_inputs = labeled_inputs.to(device, non_blocking=True)
        labeled_targets = labeled_targets.to(device, non_blocking=True)
        unlabeled_inputs = unlabeled_inputs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        labeled_logits = model(labeled_inputs)
        supervised_loss = criterion(labeled_logits, labeled_targets)

        unlabeled_logits = model(unlabeled_inputs)
        unlabeled_probabilities = torch.softmax(unlabeled_logits.detach(), dim=1)
        confidence, pseudo_targets = unlabeled_probabilities.max(dim=1)
        mask = confidence >= confidence_threshold
        if mask.any():
            unsupervised_loss = criterion(unlabeled_logits[mask], pseudo_targets[mask])
        else:
            unsupervised_loss = torch.tensor(0.0, device=device)

        loss = supervised_loss + unsupervised_weight * unsupervised_loss
        loss.backward()
        optimizer.step()

        step_count += 1
        running_loss += loss.item()
        running_accuracy += _accuracy(labeled_logits, labeled_targets)
        progress.set_postfix(
            loss=f"{running_loss / step_count:.4f}",
            acc=f"{running_accuracy / step_count:.4f}",
        )

    steps = max(step_count, 1)
    return EpochMetrics(running_loss / steps, running_accuracy / steps)



def format_metrics(epoch: int, train_metrics: EpochMetrics, test_metrics: EpochMetrics) -> str:
    return (
        f"Epoch {epoch:03d} | "
        f"train loss={train_metrics.loss:.4f}, train acc={train_metrics.accuracy:.4f} | "
        f"test loss={test_metrics.loss:.4f}, test acc={test_metrics.accuracy:.4f}"
    )
