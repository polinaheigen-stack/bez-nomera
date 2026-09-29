"""Offline backbones and identity losses for the three independent training runs.

The constructor never downloads weights. Initial HF checkpoints are loaded into
the complete native timm architecture with strict=True before its classifier is
removed. Training checkpoints belong to this ReIDModel, not to the old E22 loader.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import timm
import torch
from safetensors.torch import load_file
from timm.layers import set_reentrant_ckpt
from torch import Tensor, nn
from torch.nn import functional as F


SUPPORTED_MODELS = {
    "vit_large_patch14_dinov2.lvd142m": "dino",
    "convnext_base.fb_in22k_ft_in1k": "convnext",
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k": "swin",
}


def _load_native_weights(backbone: nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Pretrained file must contain a native timm state_dict")
    if not all(isinstance(key, str) and isinstance(value, Tensor)
               for key, value in state.items()):
        raise ValueError("Pretrained state_dict contains non-tensor values")
    # This pinned Swin checkpoint predates timm's non-persistent attention
    # buffers. It includes 35 index/mask tensors that timm 1.0.20 regenerates.
    # Remove only recognized buffers with matching shapes, never learned weights.
    native_keys = set(backbone.state_dict())
    native_buffers = dict(backbone.named_buffers())
    for key in list(state):
        if key not in native_keys and key.rsplit(".", 1)[-1] in {"relative_position_index", "attn_mask"}:
            if key not in native_buffers or native_buffers[key].shape != state[key].shape:
                raise ValueError(f"Unexpected legacy attention buffer: {key}")
            del state[key]
    # No permissive key stripping, silently dropped layers or missing parameters.
    backbone.load_state_dict(state, strict=True)


class ArcFace(nn.Module):
    """One angular-margin classifier; not a stack of competing margin losses."""

    def __init__(self, dimension: int, classes: int, margin: float, scale: float):
        super().__init__()
        if classes < 2 or not 0.0 <= margin < math.pi / 2 or scale <= 0:
            raise ValueError("ArcFace requires >=2 classes, margin in [0,pi/2), scale>0")
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.monotonic_offset = math.sin(math.pi - margin) * margin

    def forward(self, embedding: Tensor, labels: Tensor) -> Tensor:
        if labels.ndim != 1 or labels.shape[0] != embedding.shape[0]:
            raise ValueError("ArcFace labels must be one integer per image")
        if labels.dtype != torch.long:
            raise ValueError("ArcFace labels must have dtype torch.long")
        if bool(((labels < 0) | (labels >= self.weight.shape[0])).any()):
            raise ValueError("ArcFace labels outside the training class range")
        # Float32 here avoids sqrt/acos-boundary instability under mixed precision.
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            cosine = F.linear(F.normalize(embedding.float(), dim=1),
                              F.normalize(self.weight.float(), dim=1))
            cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.threshold, phi,
                              cosine - self.monotonic_offset)
            target_mask = F.one_hot(labels, num_classes=self.weight.shape[0]).bool()
            return torch.where(target_mask, phi, cosine) * self.scale


class ReIDModel(nn.Module):
    """A normalized 512-dimensional vehicle descriptor plus a training classifier.

    Head warmup: entire backbone frozen and kept in eval mode.
    Fine tuning: last four DINO blocks + final norm, or full ConvNeXt/Swin.
    No class, camera or other metadata enters descriptor extraction.
    """

    def __init__(self, config: dict[str, Any], num_classes: int,
                 initial_weights: str | Path | None = None):
        super().__init__()
        self.config = dict(config)
        model_name = config["timm_model"]
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        self.family = SUPPORTED_MODELS[model_name]
        self.image_size = tuple(int(v) for v in config["image_size"])
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive [height,width]")
        if self.family == "dino" and any(v % 14 for v in self.image_size):
            raise ValueError("DINO dimensions must be multiples of patch size 14")
        if self.family == "swin" and self.image_size != (384, 384):
            raise ValueError("This Swin checkpoint requires image_size=[384,384]")
        if int(config["embedding_dim"]) != 512:
            raise ValueError("This kit fixes embedding_dim=512")

        # Construct at the checkpoint's native resolution/class count. In
        # particular DINO uses 518px positional embeddings in its public weights.
        self.backbone = timm.create_model(model_name, pretrained=False)
        if initial_weights is not None:
            _load_native_weights(self.backbone, Path(initial_weights))
        self.backbone.reset_classifier(0)
        if self.family == "dino":
            self.backbone.set_input_size(img_size=self.image_size)

        dimension = int(self.backbone.num_features)
        if self.family == "dino":
            dimension *= 2  # CLS token concatenated with mean normalized patches.
        self.projection = nn.Sequential(
            nn.Linear(dimension, 512, bias=False),
            nn.LayerNorm(512),
        )
        nn.init.normal_(self.projection[0].weight, std=0.01)
        self.classifier = (
            ArcFace(512, num_classes, float(config["arcface_margin"]),
                    float(config["arcface_scale"])) if num_classes > 0 else None
        )
        self._finetune = False
        self._gradient_checkpointing = bool(config.get("gradient_checkpointing", False))
        # Non-reentrant checkpointing supports frozen inputs to trainable blocks.
        set_reentrant_ckpt(False)
        self.set_training_phase(False)

    def set_training_phase(self, finetune: bool) -> None:
        self._finetune = bool(finetune)
        self.backbone.requires_grad_(False)
        if self._finetune:
            if self.family == "dino":
                count = int(self.config.get("dino_unfreeze_blocks", 4))
                if not 1 <= count <= len(self.backbone.blocks):
                    raise ValueError("Invalid number of DINO blocks to unfreeze")
                for block in self.backbone.blocks[-count:]:
                    block.requires_grad_(True)
                self.backbone.norm.requires_grad_(True)
            else:
                self.backbone.requires_grad_(True)
        self.projection.requires_grad_(True)
        if self.classifier is not None:
            self.classifier.requires_grad_(True)
        self.backbone.set_grad_checkpointing(self._gradient_checkpointing and self._finetune)
        self.train(self.training)

    def train(self, mode: bool = True) -> "ReIDModel":
        super().train(mode)
        if not mode or not self._finetune:
            self.backbone.eval()
        elif self.family == "dino":
            self.backbone.eval()
            count = int(self.config.get("dino_unfreeze_blocks", 4))
            for block in self.backbone.blocks[-count:]:
                block.train(True)
            self.backbone.norm.train(True)
        return self

    def _features(self, images: Tensor) -> Tensor:
        features = self.backbone.forward_features(images)
        if self.family == "dino":
            cls_token = features[:, 0]
            patches = features[:, self.backbone.num_prefix_tokens:]
            return torch.cat((cls_token, patches.mean(dim=1)), dim=1)
        return self.backbone.forward_head(features, pre_logits=True)

    def forward(self, images: Tensor, labels: Tensor | None = None) -> dict[str, Tensor | None]:
        if images.ndim != 4 or images.shape[1] != 3 or tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected NCHW RGB images at {self.image_size}; got {tuple(images.shape)}")
        if self._finetune and self.training:
            features = self._features(images)
        else:
            with torch.no_grad():
                features = self._features(images)
        # Return float32 descriptors even when the backbone uses FP16/BF16 AMP.
        embedding = F.normalize(self.projection(features).float(), dim=1, eps=1e-12)
        logits = None
        if labels is not None:
            if self.classifier is None:
                raise ValueError("An inference-only model has no identity classifier")
            logits = self.classifier(embedding, labels)
        return {"embedding": embedding, "logits": logits}


def batch_hard_triplet(embeddings: Tensor, labels: Tensor, margin: float) -> Tensor:
    """Hardest positive and negative per valid anchor, excluding self matches."""
    if embeddings.ndim != 2 or labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("Triplet inputs must be [batch,dimension] and [batch]")
    if margin < 0:
        raise ValueError("Triplet margin must be nonnegative")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        distance = torch.cdist(embeddings.float(), embeddings.float(), p=2)
        same_id = labels[:, None].eq(labels[None, :])
        positive = same_id & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        negative = ~same_id
        valid = positive.any(dim=1) & negative.any(dim=1)
        if not bool(valid.any()):
            return embeddings.sum() * 0.0
        hardest_positive = distance.masked_fill(~positive, -torch.inf).max(dim=1).values
        hardest_negative = distance.masked_fill(~negative, torch.inf).min(dim=1).values
        return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()
