"""Dependency-light ConvNeXt-Tiny backbone for Vehicle Re-ID.

The feature extractor follows the torchvision ConvNeXt-Tiny parameter layout so
that the public ImageNet checkpoint can initialize the convolutional backbone.
The ImageNet classifier is replaced by the same embedding/BNNeck contract used
by the other project backbones.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .metric_heads import build_classifier


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels while keeping NCHW tensors."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("stochastic-depth probability must be in [0, 1]")
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_probability
        return x * mask / keep_probability


class CNBlock(nn.Module):
    """ConvNeXt block with torchvision-compatible state-dict names."""

    def __init__(self, dim: int, layer_scale: float, stochastic_depth_probability: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.Identity(),  # torchvision's Permute has no parameters
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Identity(),  # torchvision's second Permute has no parameters
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_probability)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.block[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.block[2](x)
        x = self.block[3](x)
        x = self.block[4](x)
        x = self.block[5](x)
        x = x.permute(0, 3, 1, 2)
        x = self.layer_scale * x
        return residual + self.stochastic_depth(x)


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny with a 512-dimensional Vehicle Re-ID head."""

    architecture = "ConvNeXt-Tiny"

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 0,
        bnneck_mode: str = "shared",
        metric_head: str = "linear",
        metric_scale: float = 30.0,
        metric_margin: float = 0.3,
    ) -> None:
        super().__init__()
        if bnneck_mode not in {"shared", "separate"}:
            raise ValueError(f"Unsupported BNNeck mode: {bnneck_mode}")
        self.bnneck_mode = bnneck_mode
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        total_blocks = sum(depths)
        block_index = 0
        features: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        ]
        for stage_index, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    CNBlock(
                        dim,
                        layer_scale=1e-6,
                        stochastic_depth_probability=0.1 * block_index / max(total_blocks - 1, 1),
                    )
                )
                block_index += 1
            features.append(nn.Sequential(*blocks))
            if stage_index < len(dims) - 1:
                features.append(
                    nn.Sequential(
                        LayerNorm2d(dim, eps=1e-6),
                        nn.Conv2d(dim, dims[stage_index + 1], kernel_size=2, stride=2),
                    )
                )
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(dims[-1], embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)
        self.metric_head = metric_head
        self.metric_scale = float(metric_scale)
        self.metric_margin = float(metric_margin)
        self.classifier = build_classifier(
            embedding_dim,
            num_classes,
            metric_head=metric_head,
            metric_scale=metric_scale,
            metric_margin=metric_margin,
        )
        self.embedding_dim = embedding_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, CNBlock):
                nn.init.constant_(module.layer_scale, 1e-6)

    def forward_pre_bn(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.embedding(x)

    def forward(
        self,
        x: Tensor,
        return_logits: bool = False,
        labels: Tensor | None = None,
    ):
        pre_bn_features = self.forward_pre_bn(x)
        bn_features = self.embedding_bn(pre_bn_features)
        features = pre_bn_features if self.bnneck_mode == "separate" else bn_features
        if return_logits and self.classifier is not None:
            if self.metric_head == "linear":
                logits = self.classifier(bn_features)
            else:
                logits = self.classifier(bn_features, labels)
            return features, logits
        return F.normalize(features, p=2, dim=1)
