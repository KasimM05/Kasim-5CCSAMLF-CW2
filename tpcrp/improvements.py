"""Improvement variants for TPCRP coursework experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .selector import TPCRPSelector, cluster_embeddings, compute_typicality


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    value_min = float(values.min())
    value_max = float(values.max())
    if np.isclose(value_min, value_max):
        return np.ones_like(values, dtype=np.float32)
    scaled = (values - value_min) / (value_max - value_min)
    return scaled.astype(np.float32)


@dataclass
class DiversifiedTPCRPSelector(TPCRPSelector):
    """TPCRP variant that discourages redundant picks inside a chosen cluster."""

    diversity_weight: float = 0.35

    def select(
        self,
        embeddings: np.ndarray,
        labeled_indices: list[int],
        unlabeled_indices: list[int],
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
            best_clusters = [cluster_id for cluster_id, count in candidates if count == min_labeled]
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

            reference_indices = [
                idx for idx in full_cluster_indices if idx in labeled_set or idx in selected_set
            ]
            if reference_indices:
                candidate_scores = self._score_candidates(
                    embeddings=embeddings,
                    available_cluster_indices=available_cluster_indices,
                    reference_indices=reference_indices,
                    typicality_by_index=typicality_by_index,
                )
                best_global_index = max(candidate_scores, key=candidate_scores.get)
            else:
                best_global_index = max(available_cluster_indices, key=lambda index: typicality_by_index[index])

            selected.append(best_global_index)
            selected_set.add(best_global_index)
            added_labeled_counts[selected_cluster] += 1

        return selected

    def _score_candidates(
        self,
        embeddings: np.ndarray,
        available_cluster_indices: list[int],
        reference_indices: list[int],
        typicality_by_index: dict[int, float],
    ) -> dict[int, float]:
        candidate_features = embeddings[np.asarray(available_cluster_indices)]
        reference_features = embeddings[np.asarray(reference_indices)]
        diversity = np.linalg.norm(
            candidate_features[:, None, :] - reference_features[None, :, :],
            axis=2,
        ).mean(axis=1)
        typicality = np.asarray(
            [typicality_by_index[index] for index in available_cluster_indices],
            dtype=np.float32,
        )
        combined = _minmax_scale(typicality) + self.diversity_weight * _minmax_scale(diversity)
        return {
            index: float(score)
            for index, score in zip(available_cluster_indices, combined.tolist())
        }
