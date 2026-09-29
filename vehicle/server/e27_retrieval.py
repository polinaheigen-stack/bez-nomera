"""Frozen stand-5 postprocessing, independent of all other query images.

Gallery mixing follows rerank_trial.gallery_features exactly. Top-1 support
follows the owned neighborhood_confidence.GallerySupport implementation.
Raw descriptors are never changed; caches are keyed by their complete bytes.
"""
from collections import OrderedDict
import hashlib
import threading

import numpy as np

from .e25_rerank import rank_one

POLICY = {'type': 'e27_gallery_support', 'top_k': 50, 'k1': 5,
          'lambda_value': 0.6, 'gallery_neighbors': 4, 'mix': 0.25,
          'support_neighbors': 3, 'alpha': 0.5, 'reject_all': False,
          'score_domain': 'kreciprocal_top1_times_gallery_support',
          'query_expansion': False}
THRESHOLD = 0.713592811641558
SCORE_DEFINITION = ('top1: k-reciprocal score * (0.5 + 0.5 * gallery support); '
                    'other ranks: base k-reciprocal score; similarity, not probability')
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def _matrix(gallery):
    if (not isinstance(gallery, np.ndarray) or gallery.dtype != np.float32
            or gallery.ndim != 2 or gallery.shape[1] != 384 or len(gallery) < 4
            or not np.isfinite(gallery).all()
            or not np.allclose(np.linalg.norm(gallery, axis=1), 1, atol=1e-4)):
        raise ValueError('E27 requires a finite normalized float32 gallery [N,384], N>=4')


class GalleryContext:
    def __init__(self, gallery):
        _matrix(gallery)
        self.raw = gallery.copy()
        self.raw.setflags(write=False)
        similarity = self.raw @ self.raw.T
        np.fill_diagonal(similarity, -np.inf)
        order = np.argsort(-similarity, axis=1, kind='stable')
        neighbours = order[:, :min(4, len(self.raw) - 1)]
        mixed = .75 * self.raw + .25 * self.raw[neighbours].mean(axis=1)
        self.mixed = (mixed / np.linalg.norm(mixed, axis=1, keepdims=True)).astype(np.float32)
        if not np.isfinite(self.mixed).all():
            raise ValueError('Gallery mixing produced invalid vectors')
        self.mixed.setflags(write=False)
        self.neighbours = order[:, :3].copy()
        self.neighbours.setflags(write=False)
        self.similarity = similarity
        self.similarity.setflags(write=False)

    def support(self, query, top):
        neighbours = self.neighbours[int(top)]
        anchor_similarity = np.clip(self.similarity[int(top), neighbours].astype(np.float64), 0, 1)
        query_similarity = np.clip((self.raw[neighbours] @ query).astype(np.float64), 0, 1)
        weights = anchor_similarity ** 2
        total = float(weights.sum())
        if total <= 1e-12:
            return 1.0
        relative_support = np.minimum(query_similarity / np.maximum(anchor_similarity, 1e-12), 1)
        return float(np.clip(np.dot(weights, relative_support) / total, 0, 1))


def gallery_context(gallery):
    _matrix(gallery)
    key = (hashlib.sha256(gallery.tobytes()).hexdigest(), gallery.shape)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = GalleryContext(gallery)
            while len(_CACHE) > 2:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(key)
        return _CACHE[key]


def clear_cache():
    with _LOCK:
        _CACHE.clear()


def rank_e27(query, gallery, threshold):
    query = np.asarray(query)
    if (query.dtype != np.float32 or query.shape != (384,) or not np.isfinite(query).all()
            or not np.isclose(np.linalg.norm(query), 1, atol=1e-4)):
        raise ValueError('E27 requires one finite normalized float32 query [384]')
    context = gallery_context(gallery)
    result = rank_one(query, context.mixed, top_k=50, k1=5, lambda_value=.6)
    scores = np.zeros(len(gallery), dtype=np.float64)
    scores[result['selected_indices']] = result['rerank_scores']
    order = result['order']
    top = int(order[0])
    support = context.support(query, top)  # Support uses ORIGINAL gallery descriptors.
    scores[top] = float(np.clip(float(scores[top]) * (1 - .5 + .5 * support), 0, 1))
    accepted = [top] if float(scores[top]) >= threshold else []
    return scores, order, accepted
