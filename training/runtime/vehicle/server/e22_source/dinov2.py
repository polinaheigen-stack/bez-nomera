"""Dependency-light DINOv2 ViT-S/14 with a trainable Re-ID projection head.

The frozen transformer follows the public DINOv2 state-dict layout. The public
ImageNet/self-supervised checkpoint initializes only ``backbone.*``; the local
projection, BNNeck and classifier are trained for Vehicle Re-ID.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .metric_heads import build_classifier


class DropPath(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("DINOv2 embedding dimension must divide attention heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(output)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, init_values: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.drop_path1 = DropPath()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4, dim)
        self.ls2 = LayerScale(dim, init_values)
        self.drop_path2 = DropPath()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


_DINO_VARIANTS = {
    "s14": {"label": "S", "embed_dim": 384, "heads": 6, "depth": 12},
    "b14": {"label": "B", "embed_dim": 768, "heads": 12, "depth": 12},
    "l14": {"label": "L", "embed_dim": 1024, "heads": 16, "depth": 24},
}


class DinoV2ViT(nn.Module):
    """DINOv2 ViT-S/B/L/14 backbone with official state-dict names."""

    def __init__(self, variant: str = "s14") -> None:
        super().__init__()
        if variant not in _DINO_VARIANTS:
            raise ValueError(f"Unsupported DINOv2 variant: {variant}")
        spec = _DINO_VARIANTS[variant]
        self.variant = variant
        self.embed_dim = spec["embed_dim"]
        self.depth = spec["depth"]
        self.patch_size = 14
        self.patch_embed = PatchEmbed(self.patch_size, 3, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # Official dinov2_vits14 is pretrained at 518x518: 37x37 patches + CLS.
        # The runtime crop uses bicubic interpolation to its 256x128 patch grid.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(self.embed_dim, num_heads=spec["heads"], init_values=1.0)
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: Tensor, height: int, width: int) -> Tensor:
        patch_tokens = self.pos_embed[:, 1:]
        source_size = int(patch_tokens.shape[1] ** 0.5)
        patch_tokens = patch_tokens.reshape(1, source_size, source_size, self.embed_dim)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)
        patch_tokens = F.interpolate(
            patch_tokens,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, height * width, self.embed_dim)
        return torch.cat((self.pos_embed[:, :1], patch_tokens), dim=1)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, int, int]:
        patches = self.patch_embed(x)
        height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), patches), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, height, width)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:], height, width

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]


class DinoV2ViTS14(DinoV2ViT):
    """Backward-compatible alias for the original ViT-S/14 backbone."""

    def __init__(self) -> None:
        super().__init__(variant="s14")


class DinoV2ReID(nn.Module):
    """DINOv2 global feature plus a trainable Vehicle Re-ID head."""

    architecture = "DINOv2-ViT-S/14-frozen"

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 0,
        bnneck_mode: str = "shared",
        metric_head: str = "linear",
        metric_scale: float = 30.0,
        metric_margin: float = 0.3,
        unfreeze_blocks: int = 0,
        part_stripes: int = 0,
        variant: str = "s14",
    ) -> None:
        super().__init__()
        if bnneck_mode not in {"shared", "separate"}:
            raise ValueError(f"Unsupported BNNeck mode: {bnneck_mode}")
        if variant not in _DINO_VARIANTS:
            raise ValueError(f"Unsupported DINOv2 variant: {variant}")
        depth = _DINO_VARIANTS[variant]["depth"]
        if not 0 <= unfreeze_blocks <= depth:
            raise ValueError(f"DINOv2 unfreeze_blocks must be between 0 and {depth}")
        if not 0 <= part_stripes <= 18:
            raise ValueError("DINOv2 part_stripes must be between 0 and 18")
        self.variant = variant
        label = _DINO_VARIANTS[variant]["label"]
        self.architecture = f"DINOv2-ViT-{label}/14"
        self.backbone = DinoV2ViT(variant=variant)
        self.backbone.requires_grad_(False)
        if unfreeze_blocks:
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.unfreeze_blocks = int(unfreeze_blocks)
        self.bnneck_mode = bnneck_mode
        self.metric_head = metric_head
        self.metric_scale = float(metric_scale)
        self.metric_margin = float(metric_margin)
        self.part_stripes = int(part_stripes)
        input_dim = self.backbone.embed_dim * (1 + self.part_stripes)
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)
        self.classifier = build_classifier(
            embedding_dim,
            num_classes,
            metric_head=metric_head,
            metric_scale=metric_scale,
            metric_margin=metric_margin,
        )
        self.embedding_dim = embedding_dim
        self._init_head()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _init_head(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.001)
        nn.init.ones_(self.embedding_bn.weight)
        nn.init.zeros_(self.embedding_bn.bias)

    def forward_pre_bn(self, x: Tensor) -> Tensor:
        if self.unfreeze_blocks:
            global_features, patch_features, height, width = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                global_features, patch_features, height, width = self.backbone.forward_features(x)
        if self.part_stripes:
            if self.part_stripes > height:
                raise ValueError(
                    f"DINOv2 part_stripes={self.part_stripes} exceeds patch-grid height={height}"
                )
            patch_grid = patch_features.reshape(x.shape[0], height, width, -1)
            part_features = [part.mean(dim=(1, 2)) for part in torch.tensor_split(
                patch_grid, self.part_stripes, dim=1
            )]
            features = torch.cat((global_features, *part_features), dim=1)
        else:
            features = global_features
        return self.embedding(features)

    def forward(self, x: Tensor, return_logits: bool = False, labels: Tensor | None = None):
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
