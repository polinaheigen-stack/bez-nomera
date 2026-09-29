"""Shared, offline DINOv3-S+ descriptor for the controlled stands 4/5 experiment."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_bundle import digest_json, read_object, require_hash, sha256
from .reidkit_adapter import image_tensor
from .retrieval import validate_retrieval

PRETRAINED_SHA256 = '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615'
CONFIG = {'schema_version': 1, 'timm_model': 'vit_small_plus_patch16_dinov3.lvd1689m',
          'image_size': [256, 256], 'native_dimension': 384, 'embedding_dim': 384,
          'pooling': 'native_pre_logits', 'projection': 'linear384_layernorm_l2',
          'preprocessing': 'owned_reidkit_bicubic_letterbox_imagenet', 'flip_tta': False,
          'training_amp': 'cuda_bfloat16_if_supported_else_float32',
          'inference_precision': 'float32_autocast_off_tf32_off', 'pretrained_sha256': PRETRAINED_SHA256}
RETRIEVAL = {'type': 'k_reciprocal', 'top_k': 50, 'k1': 5, 'lambda_value': .6,
             'score_domain': '1-final_distance', 'query_expansion': False}
PINNED = {'torch': '2.8.0', 'torchvision': '0.23.0', 'timm': '1.0.20',
          'numpy': '2.2.6', 'Pillow': '11.3.0', 'safetensors': '0.6.2'}


def dependencies():
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    for name, expected in PINNED.items():
        actual = versions[name].split('+')[0] if name in ('torch', 'torchvision') else versions[name]
        if actual != expected:
            raise ValueError(f'Pinned dependency required: {name}=={expected}; found {versions[name]}')
    return versions


def source_identity():
    root = Path(__file__).parent
    names = ('compact_model.py', 'compact_learning.py', 'reidkit_adapter.py',
             'reidkit_source/model.py', 'frame_buffer.py', 'batch.py',
             'retrieval.py', 'e25_rerank.py', 'validation.py', 'evaluation.py')
    return {name: sha256(root / name) for name in names}


def verify_pretrained(path, expected_sha):
    require_hash(expected_sha, 'pretrained SHA256')
    if expected_sha != PRETRAINED_SHA256:
        raise ValueError('Pretrained SHA is not the pinned DINOv3-S+ initialization')
    path = Path(path)
    if path.suffix != '.safetensors' or not path.is_file() or sha256(path) != expected_sha:
        raise ValueError('Verified local pretrained safetensors are mandatory')
    return path


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def amp_context(device):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else nullcontext())


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


def inspect_checkpoint(path):
    before = sha256(path)
    record = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(record, dict) or record.get('kind') != 'owned_compact_reid'
            or record.get('format_version') != 1 or record.get('training_finished') is not True
            or record.get('test_only') is not False or record.get('config') != CONFIG
            or record.get('stand_id') not in (4, 5) or type(record.get('selected_epoch')) is not int
            or not 1 <= record['selected_epoch'] <= 20):
        raise ValueError('Completed production compact checkpoint required')
    provenance = record.get('provenance', {})
    if (provenance.get('source_sha256') != source_identity() or provenance.get('dependencies') != dependencies()
            or provenance.get('pretrained_sha256') != PRETRAINED_SHA256
            or provenance.get('selection_basis') != 'dev_raw_retrieval_mAP@10'
            or provenance.get('control_used_for_selection') is not False
            or provenance.get('initial_model_state_sha256') is None):
        raise ValueError('Checkpoint initialization/source/dependency/selection provenance differs')
    require_hash(provenance['initial_model_state_sha256'], 'initial model SHA')
    if sha256(path) != before:
        raise ValueError('Checkpoint changed during load')
    return record


def load_checkpoint(path):
    record = inspect_checkpoint(path)
    model = CompactModel(_checkpoint_state=record['model_state'])
    return model, record


class CompactProvider:
    available = True
    reason = None
    _bundle_mode = False
    dimension = 384
    model_batch_size = None  # Dynamic true batch, specified per actual request.

    def __init__(self, checkpoint, *, device='cuda', calibration=None):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Requested device unavailable; fallback forbidden')
        before = sha256(self.path)
        self.network, self.checkpoint = load_checkpoint(self.path)
        if sha256(self.path) != before:
            raise ValueError('Checkpoint changed during reconstruction')
        self.network = self.network.requires_grad_(False).eval().to(self.device)
        self.predictor = self
        self.runtime_setup = {'mode': 'eager_compact_true_batch', 'status': 'ready',
                              'batch_policy': 'one_network_forward_per_requested_batch',
                              'inference_precision': CONFIG['inference_precision'], 'tf32': False}
        self.inference_fingerprint = digest_json({'weights': before, 'config': CONFIG, 'source': source_identity(),
                                                 'dependencies': dependencies(), 'device': str(self.device)})
        self.model = {'name': 'Owned DINOv3-S+256 compact ReID', 'version': 'stand-' + str(self.checkpoint['stand_id']),
                      'dimension': 384, 'sha256': before, 'device': str(self.device),
                      'inference_fingerprint': self.inference_fingerprint}
        self.retrieval = validate_retrieval(RETRIEVAL)
        self.threshold, self.calibration_sha256 = 0., None
        self.retrieval_fingerprint = digest_json({'retrieval': self.retrieval, 'threshold': 0., 'state': 'uncalibrated_collection'})
        if calibration is not None:
            self.bind_calibration(calibration)

    def bind_calibration(self, path):
        record = read_object(Path(path))
        if (record.get('checkpoint_sha256') != self.model['sha256']
                or record.get('inference_fingerprint') != self.inference_fingerprint
                or record.get('retrieval') != self.retrieval or record.get('source_sha256') != source_identity()
                or record.get('status') != 'frozen' or record.get('control_used_for_selection') is not False
                or type(record.get('threshold')) not in (int, float) or not np.isfinite(record['threshold'])
                or not 0 <= record['threshold'] <= 1):
            raise ValueError('Calibration does not bind to this compact checkpoint/runtime')
        self.threshold, self.calibration_sha256 = float(record['threshold']), sha256(path)
        self.retrieval_fingerprint = digest_json({'retrieval': self.retrieval, 'threshold': self.threshold,
                                                'calibration_sha256': self.calibration_sha256})

    def embed_batch(self, images, bboxes=None, *, image_ids=None):
        if not self.available or (image_ids is not None and len(image_ids) != len(images)):
            raise ValueError('Unavailable compact model or mismatched IDs')
        return self.network.embed_batch(images, bboxes)

    def embed(self, image, bbox=None, *, image_id=None):
        return self.embed_batch([image], [bbox] if bbox is not None else None)[0]

    def weight_inventory(self):
        size = self.path.stat().st_size
        if sha256(self.path) != self.model['sha256'] or not 0 < size <= 2_000_000_000:
            raise ValueError('Compact checkpoint changed or has invalid size')
        return {'status': 'verified', 'source': 'owned_compact_checkpoint', 'total_bytes': size,
                'limit_bytes': 2_000_000_000, 'model_sha256': self.model['sha256'],
                'files': [{'file': str(self.path), 'bytes': size, 'sha256': self.model['sha256'],
                           'role': 'complete_runtime_checkpoint'}],
                'scope': 'Complete compact backbone/projection; no teacher or training classifier loaded at inference.'}

    def describe(self):
        return deepcopy(self.model)

    def runtime_metrics(self):
        return deepcopy(self.runtime_setup)

    def close(self):
        self.available = False
        self.network = None
