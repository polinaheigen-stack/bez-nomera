"""One search policy for HTTP, batch export and export verification.

The E25 kernel is copied byte-for-byte from the frozen experiment. Its original
module header mentions retaining old cosine calibration; E25 instead uses the
separately calibrated final-distance threshold specified here by the caller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .e25_rerank import rank_one

RERANK_SHA256 = "6eef2ebd048e257737a67b0218d7e1e9658269c985125c55b23d9da5eff73620"
COSINE_SCORE = "(cosine + 1) / 2; similarity, not probability"
E25_SCORE = "1-final_distance of k-reciprocal top1; similarity, not probability"


def validate_retrieval(retrieval):
    if retrieval is None:
        return None
    if not isinstance(retrieval, dict):
        raise ValueError("Retrieval policy must be a dictionary.")
    if retrieval.get("type") != "k_reciprocal":
        raise ValueError("Unsupported retrieval policy; cosine fallback is forbidden.")
    for key, expected in (("top_k", 50), ("k1", 5)):
        if type(retrieval.get(key)) is not int or retrieval[key] != expected:
            raise ValueError(f"E25 requires {key}={expected}.")
    if type(retrieval.get("lambda_value")) not in (int, float) or retrieval["lambda_value"] != .6:
        raise ValueError("E25 requires lambda_value=0.6.")
    optional = {"implementation_sha256": RERANK_SHA256, "score_domain": "1-final_distance", "query_expansion": False}
    if set(retrieval) - {"type", "top_k", "k1", "lambda_value", *optional}:
        raise ValueError("Unknown E25 retrieval fields.")
    for key, expected in optional.items():
        if key in retrieval and (retrieval[key] != expected or type(retrieval[key]) is not type(expected)):
            raise ValueError(f"Invalid E25 {key}.")
    if hashlib.sha256(Path(__file__).with_name("e25_rerank.py").read_bytes()).hexdigest() != RERANK_SHA256:
        raise ValueError("E25 reranker differs from the frozen implementation.")
    return deepcopy(retrieval)


def score_definition(retrieval):
    return E25_SCORE if validate_retrieval(retrieval) else COSINE_SCORE


def rank_vectors(query, gallery, threshold, retrieval=None):
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0,1].")
    policy = validate_retrieval(retrieval)
    if policy is None:
        scores = np.clip((gallery @ query + 1) / 2, 0, 1)
        order = np.argsort(-scores, kind="stable")
        return scores, order, [int(i) for i in order if float(scores[i]) >= threshold]
    result = rank_one(query, gallery, top_k=policy["top_k"], k1=policy["k1"], lambda_value=policy["lambda_value"])
    # Scores are only defined for the reranked top-50. The unchanged cosine tail
    # never supplies display results (top-10) or accepted candidates (top-1).
    scores = np.zeros(len(gallery), dtype=np.float64)
    scores[result["selected_indices"]] = result["rerank_scores"]
    order = result["order"]
    accepted = [int(order[0])] if len(order) and float(scores[order[0]]) >= threshold else []
    return scores, order, accepted


def retrieval_record(retrieval, threshold):
    policy = validate_retrieval(retrieval)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid export threshold.")
    return {"schema_version": 1, "retrieval": policy, "threshold": threshold,
            "score_definition": score_definition(policy),
            "candidate_policy": "accepted_top1_only" if policy else "all_above_threshold"}


def resolve_export_policy(directory, retrieval=None, threshold=None):
    """Use a saved recipe when checking a detached export; never infer cosine for E25."""
    path = Path(directory) / "retrieval.json"
    policy = validate_retrieval(retrieval)
    proof_path = Path(directory) / "provenance.json"
    if not path.is_file() and proof_path.is_file():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if proof.get("retrieval") is not None:
            raise ValueError("E25 export is missing retrieval.json; cosine fallback is forbidden.")
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        saved = validate_retrieval(record.get("retrieval"))
        saved_threshold = record.get("threshold")
        if record != retrieval_record(saved, saved_threshold):
            raise ValueError("Invalid export retrieval metadata.")
        if policy is not None and policy != saved:
            raise ValueError("Export retrieval policy differs from requested policy.")
        if threshold is not None and threshold != saved_threshold:
            raise ValueError("Export threshold differs from requested threshold.")
        policy, threshold = saved, saved_threshold
    if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid export threshold.")
    return policy, threshold
