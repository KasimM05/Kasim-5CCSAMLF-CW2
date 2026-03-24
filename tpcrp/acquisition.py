"""Acquisition strategies for active learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .improvements import DiversifiedTPCRPSelector
from .selector import TPCRPSelector


def _ensure_array(indices) -> np.ndarray:
    return np.asarray(indices, dtype=np.int64)


def _random_choice(unlabeled_indices, budget: int, seed: int) -> list[int]:
    unlabeled = _ensure_array(unlabeled_indices)
    if budget <= 0 or len(unlabeled) == 0:
        return []
    rng = np.random.default_rng(seed)
    shuffled = unlabeled.copy()
    rng.shuffle(shuffled)
    return shuffled[: min(budget, len(shuffled))].tolist()


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _select_top(scores: np.ndarray, candidate_indices: np.ndarray, budget: int) -> list[int]:
    if budget <= 0 or len(candidate_indices) == 0:
        return []
    order = np.argsort(scores)[::-1]
    top = candidate_indices[order[: min(budget, len(candidate_indices))]]
    return top.tolist()


def _kmeans_plus_plus(features: np.ndarray, budget: int, seed: int) -> list[int]:
    if budget <= 0 or len(features) == 0:
        return []

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, len(features)))]
    min_distances = np.sum((features - features[selected[0]]) ** 2, axis=1)

    while len(selected) < min(budget, len(features)):
        probs = min_distances / np.maximum(min_distances.sum(), 1e-12)
        next_index = int(rng.choice(len(features), p=probs))
        if next_index in selected:
            next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        distances = np.sum((features - features[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)

    return selected


@dataclass
class AcquisitionState:
    labeled_indices: list[int]
    unlabeled_indices: list[int]
    embeddings: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    mc_probabilities: Optional[np.ndarray] = None
    seed: int = 42


class QuerySelector:
    """Base class for all query strategies."""

    requires_embeddings: bool = False
    requires_features: bool = False
    requires_probabilities: bool = False
    requires_mc_probabilities: bool = False
    uses_random_start: bool = False

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        raise NotImplementedError


class RandomSelector(QuerySelector):
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)


class LeastConfidenceSelector(QuerySelector):
    requires_probabilities = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.probabilities is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)
        candidate_indices = _ensure_array(state.unlabeled_indices)
        confidence = state.probabilities[candidate_indices].max(axis=1)
        scores = 1.0 - confidence
        return _select_top(scores, candidate_indices, budget)


class MarginSelector(QuerySelector):
    requires_probabilities = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.probabilities is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)
        candidate_indices = _ensure_array(state.unlabeled_indices)
        probabilities = state.probabilities[candidate_indices]
        top2 = np.sort(probabilities, axis=1)[:, -2:]
        margin = np.abs(top2[:, 1] - top2[:, 0])
        scores = -margin
        return _select_top(scores, candidate_indices, budget)


class EntropySelector(QuerySelector):
    requires_probabilities = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.probabilities is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)
        candidate_indices = _ensure_array(state.unlabeled_indices)
        scores = _entropy(state.probabilities[candidate_indices])
        return _select_top(scores, candidate_indices, budget)


class DBALSelector(QuerySelector):
    """MC-dropout predictive entropy."""

    requires_mc_probabilities = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.mc_probabilities is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)
        candidate_indices = _ensure_array(state.unlabeled_indices)
        mean_probabilities = state.mc_probabilities[:, candidate_indices, :].mean(axis=0)
        scores = _entropy(mean_probabilities)
        return _select_top(scores, candidate_indices, budget)


class BALDSelector(QuerySelector):
    """MC-dropout mutual information."""

    requires_mc_probabilities = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.mc_probabilities is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)
        candidate_indices = _ensure_array(state.unlabeled_indices)
        mc_probabilities = state.mc_probabilities[:, candidate_indices, :]
        predictive_entropy = _entropy(mc_probabilities.mean(axis=0))
        expected_entropy = _entropy(mc_probabilities.reshape(-1, mc_probabilities.shape[-1])).reshape(
            mc_probabilities.shape[0],
            mc_probabilities.shape[1],
        ).mean(axis=0)
        scores = predictive_entropy - expected_entropy
        return _select_top(scores, candidate_indices, budget)


class CoreSetSelector(QuerySelector):
    requires_features = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.features is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)

        candidate_indices = _ensure_array(state.unlabeled_indices)
        candidate_features = state.features[candidate_indices]
        labeled_indices = _ensure_array(state.labeled_indices)

        selected_local = []
        if len(labeled_indices) > 0:
            anchor_features = state.features[labeled_indices]
            min_distances = np.linalg.norm(
                candidate_features[:, None, :] - anchor_features[None, :, :],
                axis=2,
            ).min(axis=1)
        else:
            min_distances = np.full(len(candidate_indices), np.inf, dtype=np.float32)

        while len(selected_local) < min(budget, len(candidate_indices)):
            next_local_index = int(np.argmax(min_distances))
            selected_local.append(next_local_index)
            new_distances = np.linalg.norm(
                candidate_features - candidate_features[next_local_index],
                axis=1,
            )
            min_distances = np.minimum(min_distances, new_distances)
            min_distances[selected_local] = -np.inf

        return candidate_indices[np.asarray(selected_local, dtype=np.int64)].tolist()


class BADGESelector(QuerySelector):
    requires_features = True
    requires_probabilities = True
    uses_random_start = True

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.features is None or state.probabilities is None:
            return _random_choice(state.unlabeled_indices, budget=budget, seed=state.seed)

        candidate_indices = _ensure_array(state.unlabeled_indices)
        candidate_features = state.features[candidate_indices]
        candidate_probabilities = state.probabilities[candidate_indices]
        predicted_labels = candidate_probabilities.argmax(axis=1)
        num_classes = candidate_probabilities.shape[1]
        one_hot = np.eye(num_classes, dtype=np.float32)[predicted_labels]
        gradient_embeddings = []
        for feature, probability, target in zip(candidate_features, candidate_probabilities, one_hot):
            gradient_embeddings.append(np.outer(probability - target, feature).reshape(-1))
        gradient_embeddings = np.asarray(gradient_embeddings, dtype=np.float32)
        selected_local = _kmeans_plus_plus(gradient_embeddings, budget=budget, seed=state.seed)
        return candidate_indices[np.asarray(selected_local, dtype=np.int64)].tolist()


class TypiClustSelector(QuerySelector):
    requires_embeddings = True

    def __init__(self, max_clusters: int, min_cluster_size: int, knn_k: int, seed: int) -> None:
        self.inner = TPCRPSelector(
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            knn_k=knn_k,
            seed=seed,
        )

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.embeddings is None:
            return []
        return self.inner.select(
            embeddings=state.embeddings,
            labeled_indices=state.labeled_indices,
            unlabeled_indices=state.unlabeled_indices,
            budget=budget,
        )


class DiversifiedTypiClustSelector(QuerySelector):
    requires_embeddings = True

    def __init__(
        self,
        max_clusters: int,
        min_cluster_size: int,
        knn_k: int,
        seed: int,
        diversity_weight: float,
    ) -> None:
        self.inner = DiversifiedTPCRPSelector(
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            knn_k=knn_k,
            seed=seed,
            diversity_weight=diversity_weight,
        )

    def select(self, state: AcquisitionState, budget: int) -> list[int]:
        if state.embeddings is None:
            return []
        return self.inner.select(
            embeddings=state.embeddings,
            labeled_indices=state.labeled_indices,
            unlabeled_indices=state.unlabeled_indices,
            budget=budget,
        )


SELECTOR_NAMES = [
    "random",
    "uncertainty",
    "margin",
    "entropy",
    "dbal",
    "coreset",
    "bald",
    "badge",
    "typiclust",
    "diversified",
]


def build_query_selector(
    selector_name: str,
    max_clusters: int,
    min_cluster_size: int,
    knn_k: int,
    seed: int,
    diversity_weight: float,
) -> QuerySelector:
    if selector_name == "random":
        return RandomSelector()
    if selector_name == "uncertainty":
        return LeastConfidenceSelector()
    if selector_name == "margin":
        return MarginSelector()
    if selector_name == "entropy":
        return EntropySelector()
    if selector_name == "dbal":
        return DBALSelector()
    if selector_name == "coreset":
        return CoreSetSelector()
    if selector_name == "bald":
        return BALDSelector()
    if selector_name == "badge":
        return BADGESelector()
    if selector_name == "typiclust":
        return TypiClustSelector(
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            knn_k=knn_k,
            seed=seed,
        )
    if selector_name == "diversified":
        return DiversifiedTypiClustSelector(
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            knn_k=knn_k,
            seed=seed,
            diversity_weight=diversity_weight,
        )
    raise ValueError(f"Unknown selector: {selector_name}")
