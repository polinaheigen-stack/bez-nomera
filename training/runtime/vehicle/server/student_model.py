"""Owned offline student descriptor. No trained student is included in this stand."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import importlib.metadata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_bundle import digest_json, sha256, read_object
from .reidkit_adapter import image_tensor
from .reidkit_source.model import _load_native_weights
from .retrieval import validate_retrieval

CONFIG = {'schema_version': 1, 'timm_model': 'vit_base_patch16_224',
          'image_size': [224, 224], 'embedding_dim': 1024,
          'pooling': 'native_pre_logits', 'projection': 'linear_bias_false_layernorm',
          'preprocessing': 'owned_reidkit_image_tensor_bicubic_letterbox',
          'flip_tta': False, 'amp': 'cuda_bfloat16_if_supported_else_float32'}
RETRIEVAL = {'type': 'k_reciprocal', 'top_k': 50, 'k1': 5, 'lambda_value': .6,
             'score_domain': '1-final_distance', 'query_expansion': False}


def source_identity():
    root = Path(__file__).parent
    names = ('student_model.py', 'student_trial.py', 'reidkit_adapter.py',
             'reidkit_source/model.py', 'frame_buffer.py', 'batch.py',
             'retrieval.py', 'e25_rerank.py', 'validation.py', 'evaluation.py')
    return {name: sha256(root / name) for name in names}


def dependencies():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'torchvision', 'timm', 'numpy', 'Pillow', 'safetensors')}


class StudentModel(nn.Module):
    def __init__(self, config=None, initial_weights=None, *, backbone=None):
        super().__init__()
        self.config = deepcopy(config or CONFIG)
        if self.config != CONFIG:
            raise ValueError('Unknown student architecture/preprocessing')
        if backbone is None:
            import timm
            backbone = timm.create_model(CONFIG['timm_model'], pretrained=False)
            if initial_weights is not None:
                _load_native_weights(backbone, Path(initial_weights))
            backbone.reset_classifier(0)
        elif initial_weights is not None:
            raise ValueError('Test backbone cannot load production initial weights')
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(int(backbone.num_features), 1024, bias=False), nn.LayerNorm(1024))

    def forward(self, images):
        features = self.backbone.forward_head(self.backbone.forward_features(images), pre_logits=True)
        return F.normalize(self.projection(features).float(), dim=1, eps=1e-12)


def load_checkpoint(path):
    record = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(record, dict) or record.get('kind') != 'owned_e25_distilled_student'
            or record.get('format_version') != 1 or record.get('training_finished') is not True
            or record.get('config') != CONFIG or type(record.get('selected_epoch')) is not int
            or record['selected_epoch'] < 1 or not isinstance(record.get('provenance'), dict)):
        raise ValueError('A completed, self-describing owned student checkpoint is required')
    provenance = record['provenance']
    if (provenance.get('source_sha256') != source_identity()
            or provenance.get('dependencies') != dependencies()
            or provenance.get('selection_basis') != 'minimum_validation_cosine_loss'
            or provenance.get('control_used_for_selection') is not False
            or provenance.get('independent_quality_claim_allowed') is not False):
        raise ValueError('Student source/dependency/selection provenance mismatch')
    for name in ('train_cache_sha256', 'val_cache_sha256', 'data_contract_sha256'):
        from .model_bundle import require_hash
        require_hash(provenance.get(name), name)
    model = StudentModel(record['config'])
    state = record.get('model_state')
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Student state keys differ from its architecture')
    for name, value in state.items():
        if (not torch.is_tensor(value) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or (value.is_floating_point() and not torch.isfinite(value).all().item())):
            raise ValueError('Invalid student tensor: ' + name)
    model.load_state_dict(state, strict=True)
    return model, record


def amp_context(device):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else nullcontext())


class StudentProvider:
    """Trial adapter; ready for export only after explicit checkpoint + calibration."""
    available = True
    reason = None
    dimension = 1024
    model_batch_size = 1
    _bundle_mode = False

    def __init__(self, checkpoint, *, device='cuda', calibration=None):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; no CPU fallback')
        before = sha256(self.path)
        self.network, self.checkpoint = load_checkpoint(self.path)
        if sha256(self.path) != before:
            raise ValueError('Student checkpoint changed while loading')
        self.network = self.network.requires_grad_(False).eval().to(self.device)
        self.predictor = self
        self.runtime_setup = {'mode': 'eager_student', 'status': 'ready', 'model_batch_size': 1,
                              'amp': CONFIG['amp'], 'compiled': False}
        self.inference_fingerprint = digest_json({'weights': before, 'config': CONFIG,
            'source': source_identity(), 'dependencies': dependencies(), 'device': str(self.device)})
        self.model = {'name': 'Owned E25 student ViT-B224', 'version': 'P1-P4-v1', 'dimension': 1024,
                      'sha256': before, 'inference_fingerprint': self.inference_fingerprint, 'device': str(self.device)}
        self.retrieval = validate_retrieval(RETRIEVAL)
        # Zero is only an explicit calibration collection state, not a usable deployment threshold.
        self.threshold, self.calibration_sha256 = 0., None
        self.retrieval_fingerprint = digest_json({'retrieval': self.retrieval, 'threshold': 0., 'state': 'uncalibrated'})
        if calibration is not None:
            self.bind_calibration(calibration)

    def bind_calibration(self, path):
        record = read_object(Path(path))
        if (record.get('checkpoint_sha256') != self.model['sha256']
                or record.get('inference_fingerprint') != self.inference_fingerprint
                or record.get('retrieval') != self.retrieval or record.get('control_used_for_selection') is not False
                or record.get('status') != 'frozen' or record.get('source_sha256') != source_identity()
                or type(record.get('threshold')) not in (int, float) or not 0 <= record['threshold'] <= 1):
            raise ValueError('Calibration does not bind to this student/runtime')
        self.threshold = float(record['threshold'])
        self.calibration_sha256 = sha256(path)
        self.retrieval_fingerprint = digest_json({'retrieval': self.retrieval, 'threshold': self.threshold,
                                                'calibration_sha256': self.calibration_sha256})

    @torch.inference_mode()
    def embed_batch(self, images, bboxes, *, image_ids=None):
        if not self.available or len(images) != len(bboxes):
            raise ValueError('Unavailable student or mismatched image/bbox count')
        chunks = []
        for image, bbox in zip(images, bboxes):
            tensor = image_tensor(image, bbox, CONFIG['image_size']).unsqueeze(0).to(self.device)
            with amp_context(self.device):
                vector = self.network(tensor)
            chunks.append(F.normalize(vector.float(), dim=1).cpu().numpy().astype(np.float32))
        return np.concatenate(chunks) if chunks else np.empty((0, 1024), dtype=np.float32)

    def embed(self, image, bbox, *, image_id=None):
        return self.embed_batch([image], [bbox])[0]

    def weight_inventory(self):
        if sha256(self.path) != self.model['sha256']:
            raise ValueError('Student weights changed')
        size = self.path.stat().st_size
        if not 0 < size <= 2_000_000_000:
            raise ValueError('Student checkpoint size out of bounds')
        return {'status': 'verified', 'source': 'owned_student_checkpoint', 'total_bytes': size,
                'limit_bytes': 2_000_000_000, 'model_sha256': self.model['sha256'],
                'files': [{'file': str(self.path), 'bytes': size, 'sha256': self.model['sha256'],
                           'role': 'complete_runtime_checkpoint'}],
                'scope': 'One complete student; teacher checkpoints are training-only and not loaded for student inference.'}

    def close(self):
        self.available = False
        self.network = None
