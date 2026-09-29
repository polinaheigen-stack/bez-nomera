"""Single-query, top-K-only k-reciprocal reranking using NumPy.

Clean implementation of reciprocal encoding, set expansion and weighted Jaccard
distance from Zhong et al., CVPR 2017, https://arxiv.org/abs/1701.08398.
This deliberately omits the paper's local query expansion (Eq. 11 / k2).
Neither input embeddings nor reciprocal encoding rows are averaged with neighbors.

Adaptations fixed for this experiment:
* One query and its cosine top-K gallery entries form the entire local graph.
* Baseline score = clip((cosine + 1) / 2, 0, 1), using baseline float32 math.
  d = 1 - score, i.e. bounded cosine distance, without per-query maximum scaling.
* Each row contains self plus up to k1 other nearest neighbors. Self wins ties.
* Reciprocal sets may expand by the half-k1 overlap rule (Eq. 4). This is set
  membership expansion, not query expansion or embedding/encoding averaging.
* exp(-d) weights are L1-normalized before weighted Jaccard comparison.
* Final distance = (1-lambda) * Jaccard + lambda * d.

Only ordering changes. Returned rerank scores are similarities, not calibrated
probabilities; use the original cosine scores for existing calibration/decisions.
"""

from __future__ import annotations

import numbers

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_embeddings(query: np.ndarray, gallery: np.ndarray) -> None:
    for name, value, ndim in (("query", query, 1), ("gallery", gallery, 2)):
        if not isinstance(value, np.ndarray) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D NumPy array")
        if value.dtype != np.float32:
            raise ValueError(f"{name} must have dtype float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if query.size == 0 or gallery.shape[1] != query.size:
        raise ValueError("query and gallery must have the same nonzero embedding dimension")
    # Validation never renormalizes or mutates frozen baseline embeddings.
    for name, value in (("query", query), ("gallery", gallery)):
        norm = np.linalg.norm(value.astype(np.float64), axis=-1)
        if not np.allclose(norm, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError(f"{name} embeddings must be L2-normalized")


def _neighbor_order(distances: np.ndarray) -> np.ndarray:
    """Self first, then increasing distance with stable local-index ties."""
    sorting_distances = distances.copy()
    np.fill_diagonal(sorting_distances, -1.0)
    return np.argsort(sorting_distances, axis=1, kind="stable")


def _reciprocal_set(neighbor_order: np.ndarray, item: int, k: int) -> np.ndarray:
    """Mutual neighbors among self plus k other items (self always included)."""
    width = min(k + 1, len(neighbor_order))
    forward = neighbor_order[item, :width]
    is_mutual = np.any(neighbor_order[forward, :width] == item, axis=1)
    return forward[is_mutual]


def _reciprocal_encoding(distances: np.ndarray, k1: int) -> np.ndarray:
    """Encode expanded reciprocal sets; never average neighboring rows."""
    neighbor_order = _neighbor_order(distances)
    effective_k = min(k1, len(distances) - 1)
    half_k = max(1, effective_k // 2)
    encoding = np.zeros(distances.shape, dtype=np.float64)
    for item in range(len(distances)):
        reciprocal = _reciprocal_set(neighbor_order, item, effective_k)
        expanded = set(reciprocal.tolist())
        for candidate in reciprocal:
            half_reciprocal = _reciprocal_set(neighbor_order, int(candidate), half_k)
            overlap = np.intersect1d(reciprocal, half_reciprocal, assume_unique=True).size
            # Strict > 2/3 is the published set-expansion condition (Eq. 4).
            if 3 * overlap > 2 * half_reciprocal.size:
                expanded.update(half_reciprocal.tolist())
        indices = np.asarray(sorted(expanded), dtype=np.int64)
        weights = np.exp(-distances[item, indices].astype(np.float64))
        encoding[item, indices] = weights / weights.sum()
    return encoding


def rank_one(
    query: np.ndarray,
    gallery: np.ndarray,
    *,
    top_k: int = 50,
    k1: int = 10,
    lambda_value: float = 0.3,
) -> dict:
    """Rerank cosine top-K candidates for exactly one query.

    Inputs must be finite L2-normalized float32 arrays of shape (D,) and (N,D).
    No IDs, labels, cameras, other queries or persistent mutable state are used.
    An empty gallery is supported. Invalid inputs raise ValueError.

    Returns:
        order: Full gallery-index permutation. Only its initial top-K is reranked;
            the remaining indices retain their original cosine ranking order.
        cosine_scores: Baseline scores clip((cosine+1)/2,0,1), aligned with the
            original gallery rows. Exact float32 baseline arithmetic is retained.
        raw_cosines: Clipped cosines in [-1,1], aligned with original gallery rows.
        selected_indices: Cosine top-K gallery indices before reranking.
        final_distances: Distances in [0,1], aligned with selected_indices.
        rerank_scores: 1-final_distances, aligned with selected_indices; these are
            not probabilities and must not be substituted into old calibration.
        params: Requested parameters plus the effective top-K and k1.
        metadata: Explicit algorithm/protocol choices for provenance.

    Final-distance ties preserve the original cosine order. Initial cosine ties
    preserve the input gallery index. Results and inputs share no writable arrays.
    """
    top_k = _positive_integer(top_k, "top_k")
    k1 = _positive_integer(k1, "k1")
    if (
        isinstance(lambda_value, (bool, np.bool_))
        or not isinstance(lambda_value, numbers.Real)
        or not np.isfinite(lambda_value)
        or not 0.0 <= lambda_value <= 1.0
    ):
        raise ValueError("lambda_value must be a finite number in [0,1]")
    lambda_value = float(lambda_value)
    _validate_embeddings(query, gallery)

    dot_products = gallery @ query
    # Keep the exact formula and float32 operation order used by the frozen
    # validation.rank_vectors baseline. Normalizing embeddings or promoting
    # this expression to float64 could change threshold decisions or tied ranks.
    cosine_scores = np.clip((dot_products + 1) / 2, 0, 1)
    raw_cosines = np.clip(dot_products, -1, 1)
    baseline_order = np.argsort(-cosine_scores, kind="stable")
    count = min(top_k, len(gallery))
    selected_indices = baseline_order[:count].copy()
    params = {
        "top_k": top_k,
        "k1": k1,
        "lambda_value": lambda_value,
        "effective_top_k": count,
        "effective_k1": min(k1, count),
    }
    metadata = {
        "algorithm": "single_query_top_k_k_reciprocal",
        "algorithm_version": 1,
        "distance": "1-clip((cosine+1)/2,0,1), baseline float32 score arithmetic",
        "cosine_score": "clip((gallery@query+1)/2,0,1)",
        "weighting": "exp(-distance), L1 normalized",
        "reciprocal_set_expansion": "half effective k1, strict overlap > 2/3",
        "self_neighbor": True,
        "local_query_expansion": False,
        "k2": None,
        "aqe": False,
        "neighbor_feature_averaging": False,
        "query_count_in_graph": 1,
        "gallery_count_in_graph": count,
        "score_is_probability": False,
        "initial_tie_break": "input gallery index",
        "final_tie_break": "initial cosine ranking",
    }
    if count == 0:
        final_distances = np.empty(0, dtype=np.float64)
        order = baseline_order.copy()
    else:
        local_features = np.concatenate((query[None, :], gallery[selected_indices]), axis=0)
        local_scores = np.clip((local_features @ local_features.T + 1) / 2, 0, 1)
        distances = 1.0 - local_scores.astype(np.float64)
        np.fill_diagonal(distances, 0.0)
        # Reuse the very same baseline query cosines used for top-K selection;
        # matrix-matrix and matrix-vector BLAS may otherwise differ by a few ulps.
        query_distances = 1.0 - cosine_scores[selected_indices].astype(np.float64)
        distances[0, 1:] = query_distances
        distances[1:, 0] = query_distances
        encoding = _reciprocal_encoding(distances, k1)
        intersection = np.minimum(encoding[0], encoding[1:]).sum(axis=1)
        union = np.maximum(encoding[0], encoding[1:]).sum(axis=1)
        jaccard_distances = np.clip(1.0 - intersection / union, 0.0, 1.0)
        final_distances = np.clip(
            (1.0 - lambda_value) * jaccard_distances + lambda_value * query_distances,
            0.0,
            1.0,
        )
        reranked = selected_indices[np.argsort(final_distances, kind="stable")]
        order = np.concatenate((reranked, baseline_order[count:]))
    return {
        "order": order,
        "cosine_scores": cosine_scores,
        "raw_cosines": raw_cosines,
        "selected_indices": selected_indices,
        "final_distances": final_distances,
        "rerank_scores": 1.0 - final_distances,
        "params": params,
        "metadata": metadata,
    }
