"""Metric-learning classifier heads used only during supervised training."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CosineMarginHead(nn.Module):
    """CosFace/ArcFace-style normalized classifier with configurable margin."""

    def __init__(self, in_features: int, num_classes: int, scale: float, margin: float, mode: str) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("metric head requires at least one class")
        if scale <= 0:
            raise ValueError("metric head scale must be positive")
        if margin < 0:
            raise ValueError("metric head margin must be non-negative")
        if mode not in {"arcface", "cosface"}:
            raise ValueError(f"unsupported metric head: {mode}")
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale = float(scale)
        self.margin = float(margin)
        self.mode = mode
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = F.linear(F.normalize(features, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if labels is None:
            return cosine * self.scale
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("metric head labels must be a one-dimensional batch-aligned tensor")
        if labels.min().item() < 0 or labels.max().item() >= self.weight.shape[0]:
            raise ValueError("metric head labels contain an out-of-range class")
        target = cosine.gather(1, labels[:, None]).squeeze(1)
        if self.mode == "arcface":
            sine = torch.sqrt((1.0 - target.square()).clamp_min(1e-7))
            target = target * math.cos(self.margin)
            target = target - sine * math.sin(self.margin)
        else:
            target = target - self.margin
        logits = cosine.clone()
        logits.scatter_(1, labels[:, None], target.to(dtype=logits.dtype)[:, None])
        return logits * self.scale


def build_classifier(
    embedding_dim: int,
    num_classes: int,
    metric_head: str = "linear",
    metric_scale: float = 30.0,
    metric_margin: float = 0.3,
) -> nn.Module | None:
    if num_classes <= 0:
        return None
    if metric_head == "linear":
        return nn.Linear(embedding_dim, num_classes)
    if metric_head in {"arcface", "cosface"}:
        return CosineMarginHead(
            embedding_dim,
            num_classes,
            scale=metric_scale,
            margin=metric_margin,
            mode=metric_head,
        )
    raise ValueError(f"unsupported metric head: {metric_head}")
