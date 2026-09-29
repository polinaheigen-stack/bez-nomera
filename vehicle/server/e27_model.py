"""Inference architecture copied from the measured stand-5 compact model.

Native forward, preprocessing, precision and batched extraction are retained.
Production loads only the complete pinned checkpoint through e27_adapter.
"""
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model_bundle import require_hash, sha256
from .reidkit_adapter import image_tensor

PRETRAINED_SHA256 = '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615'


CONFIG = {'schema_version': 1, 'timm_model': 'vit_small_plus_patch16_dinov3.lvd1689m',
          'image_size': [256, 256], 'native_dimension': 384, 'embedding_dim': 384,
          'pooling': 'native_pre_logits', 'projection': 'linear384_layernorm_l2',
          'preprocessing': 'owned_reidkit_bicubic_letterbox_imagenet', 'flip_tta': False,
          'training_amp': 'cuda_bfloat16_if_supported_else_float32',
          'inference_precision': 'float32_autocast_off_tf32_off', 'pretrained_sha256': PRETRAINED_SHA256}


PINNED = {'torch': '2.8.0', 'torchvision': '0.23.0', 'timm': '1.0.20',
          'numpy': '2.2.6', 'Pillow': '11.3.0', 'safetensors': '0.6.2'}


def dependencies():
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    for name, expected in PINNED.items():
        actual = versions[name].split('+')[0] if name in ('torch', 'torchvision') else versions[name]
        if actual != expected:
            raise ValueError(f'Pinned dependency required: {name}=={expected}; found {versions[name]}')
    return versions


def verify_pretrained(path, expected_sha):
    require_hash(expected_sha, 'pretrained SHA256')
    if expected_sha != PRETRAINED_SHA256:
        raise ValueError('Pretrained SHA is not the pinned DINOv3-S+ initialization')
    path = Path(path)
    if path.suffix != '.safetensors' or not path.is_file() or sha256(path) != expected_sha:
        raise ValueError('Verified local pretrained safetensors are mandatory')
    return path


class CompactModel(nn.Module):
    """Fresh production construction always requires the pinned local initialization."""
    def __init__(self, pretrained_path=None, pretrained_sha=None, *, backbone=None,
                 test_only=False, _checkpoint_state=None):
        super().__init__()
        self.config = deepcopy(CONFIG)
        self.test_only = bool(test_only)
        if backbone is not None:
            if not test_only or _checkpoint_state is not None or pretrained_path is not None:
                raise ValueError('Injected backbones are reserved for explicit synthetic tests')
        else:
            if test_only:
                raise ValueError('Synthetic tests must inject a tiny backbone')
            dependencies()
            if _checkpoint_state is None:
                if pretrained_path is None or pretrained_sha is None:
                    raise ValueError('Fresh model requires verified pretrained weights; random fallback is forbidden')
                pretrained_path = verify_pretrained(pretrained_path, pretrained_sha)
            import timm
            backbone = timm.create_model(CONFIG['timm_model'], pretrained=False, num_classes=0,
                                         img_size=CONFIG['image_size'][0])
            if _checkpoint_state is None:
                from safetensors.torch import load_file
                state = load_file(str(pretrained_path), device='cpu')
                backbone.load_state_dict(state, strict=True)
                verify_pretrained(pretrained_path, pretrained_sha)
        if int(backbone.num_features) != CONFIG['native_dimension']:
            raise ValueError('Backbone must expose native 384-dimensional features')
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(384, 384, bias=False), nn.LayerNorm(384))
        if _checkpoint_state is not None:
            validate_state(self, _checkpoint_state)
            self.load_state_dict(_checkpoint_state, strict=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError('Compact model requires NCHW RGB tensors at 256x256')
        features = self.backbone.forward_head(self.backbone.forward_features(images), pre_logits=True)
        if features.ndim != 2 or features.shape[1] != 384:
            raise ValueError('Unexpected backbone feature shape')
        return F.normalize(self.projection(features).float(), dim=1, eps=1e-12)

    @torch.inference_mode()
    def embed_batch(self, images, bboxes=None):
        if self.training:
            raise ValueError('Descriptor extraction requires model.eval()')
        if bboxes is None:
            bboxes = [(0, 0, image.width, image.height) for image in images]
        if len(images) != len(bboxes):
            raise ValueError('Image/bbox lengths differ')
        if not images:
            return np.empty((0, 384), dtype=np.float32)
        tensor = torch.stack([image_tensor(image, bbox, CONFIG['image_size']) for image, bbox in zip(images, bboxes)]).to(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # One forward for the whole request; no hidden per-image network loop.
        # FP32 deliberately differs from training AMP. Batch parity is still
        # verified on the target GPU; FP32 alone is not a proof of equal outputs.
        with torch.autocast(device_type=self.device.type, enabled=False):
            vectors = self(tensor)
        result = vectors.float().cpu().numpy().astype(np.float32)
        if result.shape != (len(images), 384) or not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-4):
            raise ValueError('Invalid compact embeddings')
        return result


def validate_state(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Checkpoint state keys differ from fixed architecture')
    for name, tensor in state.items():
        if (not torch.is_tensor(tensor) or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype
                or (tensor.is_floating_point() and not torch.isfinite(tensor).all().item())):
            raise ValueError('Invalid checkpoint tensor: ' + name)
