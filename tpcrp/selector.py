"""TPCRP query selection for low-budget active learning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
except ImportError:  # pragma: no cover - optional dependency
    KMeans = None
    MiniBatchKMeans = None


def _run_numpy_kmeans(embeddings: np.ndarray, n_clusters: int, seed: int, iterations: int = 25) -> np.ndarray:
    """Fallback K-means used when scikit-learn is unavailable."""
    rng = np.random.default_rng(seed)
    n_samples = embeddings.shape[0]
    centers = embeddings[rng.choice(n_samples, size=n_clusters, replace=False)].copy()

    for _ in range(iterations):
        distances = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = distances.argmin(axis=1)

        new_centers = centers.copy()
        for cluster_id in range(n_clusters):
            members = embeddings[assignments == cluster_id]
            if len(members) == 0:
                new_centers[cluster_id] = embeddings[rng.integers(0, n_samples)]
            else:
                new_centers[cluster_id] = members.mean(axis=0)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    distances = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return distances.argmin(axis=1)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if n_clusters <= 1:
        return np.zeros(len(embeddings), dtype=np.int64)

    if KMeans is not None and MiniBatchKMeans is not None:
        if n_clusters <= 50:
            estimator = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        else:
            estimator = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024, n_init=10)
        return estimator.fit_predict(embeddings)

    return _run_numpy_kmeans(embeddings, n_clusters=n_clusters, seed=seed)


def compute_typicality(cluster_embeddings: np.ndarray, neighbor_count: int) -> np.ndarray:
    if len(cluster_embeddings) == 0:
        return np.array([], dtype=np.float32)

    if len(cluster_embeddings) == 1:
        return np.array([float("inf")], dtype=np.float32)

    distances = np.linalg.norm(
        cluster_embeddings[:, None, :] - cluster_embeddings[None, :, :],
        axis=2,
    )
    np.fill_diagonal(distances, np.inf)
    k = max(1, min(neighbor_count, len(cluster_embeddings) - 1))
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    mean_distance = nearest.mean(axis=1)
    return 1.0 / np.maximum(mean_distance, 1e-12)


@dataclass
class TPCRPSelector:
    max_clusters: int = 500
    min_cluster_size: int = 5
    knn_k: int = 20
    seed: int = 42

    def select(
        self,
        embeddings: np.ndarray,
        labeled_indices: Sequence[int],
        unlabeled_indices: Sequence[int],
        budget: int,
    ) -> list[int]:
        if budget <= 0 or len(unlabeled_indices) == 0:
            return []

        total_clusters = min(len(labeled_indices) + budget, self.max_clusters, len(embeddings))
        assignments = cluster_embeddings(embeddings, n_clusters=max(total_clusters, 1), seed=self.seed)

        cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        cluster_labeled_counts: dict[int, int] = defaultdict(int)
        cluster_unlabeled_indices: dict[int, list[int]] = defaultdict(list)

        labeled_set = set(labeled_indices)
        unlabeled_set = set(unlabeled_indices)

        for index, cluster_id in enumerate(assignments.tolist()):
            cluster_to_indices[cluster_id].append(index)
            if index in labeled_set:
                cluster_labeled_counts[cluster_id] += 1
            if index in unlabeled_set:
                cluster_unlabeled_indices[cluster_id].append(index)

        selected: list[int] = []
        selected_set: set[int] = set()
        added_labeled_counts: dict[int, int] = defaultdict(int)

        while len(selected) < budget:
            candidates: list[tuple[int, int]] = []
            for cluster_id, members in cluster_to_indices.items():
                if len(members) < self.min_cluster_size:
                    continue

                available_unlabeled = [
                    idx for idx in cluster_unlabeled_indices[cluster_id] if idx not in selected_set
                ]
                if not available_unlabeled:
                    continue

                labeled_count = cluster_labeled_counts[cluster_id] + added_labeled_counts[cluster_id]
                candidates.append((cluster_id, labeled_count))

            if not candidates:
                break

            min_labeled = min(count for _, count in candidates)
            best_clusters = [
                cluster_id
                for cluster_id, count in candidates
                if count == min_labeled
            ]
            selected_cluster = max(best_clusters, key=lambda cluster_id: len(cluster_to_indices[cluster_id]))
            full_cluster_indices = cluster_to_indices[selected_cluster]
            available_cluster_indices = [
                idx for idx in cluster_unlabeled_indices[selected_cluster] if idx not in selected_set
            ]
            cluster_features = embeddings[np.asarray(full_cluster_indices)]
            typicality = compute_typicality(
                cluster_features,
                neighbor_count=min(self.knn_k, len(full_cluster_indices)),
            )
            typicality_by_index = {
                index: score for index, score in zip(full_cluster_indices, typicality.tolist())
            }
            best_global_index = max(available_cluster_indices, key=lambda index: typicality_by_index[index])

            selected.append(best_global_index)
            selected_set.add(best_global_index)
            added_labeled_counts[selected_cluster] += 1

        return selected
