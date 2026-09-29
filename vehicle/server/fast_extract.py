"""Additive extraction experiment; the original provider owns its model and weights.

Only CPU image preparation is threaded. There is one ordered, true model batch.
This adapter does not assert GPU parity or end-to-end retrieval performance.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from vehicle.server.reidkit_adapter import image_tensor


MAX_BATCH = 64


@contextmanager
def _exact_backends():
    """Use the frozen FP32/TF32 policy without permanently changing the caller."""
    slots = ((torch.backends.cuda.matmul, 'allow_tf32', False),
             (torch.backends.cudnn, 'allow_tf32', False),
             (torch.backends.cudnn, 'benchmark', False),
             (torch.backends.cudnn, 'deterministic', True))
    previous = [getattr(owner, key) for owner, key, value in slots]
    try:
        for owner, key, value in slots:
            setattr(owner, key, value)
        yield
    finally:
        for (owner, key, value), original in zip(slots, previous):
            setattr(owner, key, original)


class FastExtractor:
    def __init__(self, provider, workers=4, precision='fp32', pin_memory=True,
                 *, allow_cpu_diagnostic=False):
        if type(workers) is not int or not 0 <= workers <= 8:
            raise ValueError('workers must be an integer from 0 through 8')
        if precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('precision must be fp32, fp16 or bf16')
        if type(pin_memory) is not bool or type(allow_cpu_diagnostic) is not bool:
            raise ValueError('Boolean options must be bool')
        self.device = torch.device(provider.device)
        if self.device.type == 'cuda' and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device('cuda', torch.cuda.current_device())
        diagnostic = self.device.type == 'cpu' and allow_cpu_diagnostic
        if self.device.type == 'cpu' and not diagnostic:
            raise ValueError('CPU extraction requires explicit allow_cpu_diagnostic=True')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CUDA or explicit CPU diagnostics are supported')
        if diagnostic and precision != 'fp32':
            raise ValueError('CPU diagnostics support FP32 only')
        if not diagnostic:
            from vehicle.server.compact_model import CompactProvider
            if not isinstance(provider, CompactProvider) or not torch.cuda.is_available():
                raise ValueError('Production extraction requires the original strict CUDA CompactProvider')
            if precision == 'bf16':
                with torch.cuda.device(self.device):
                    if not torch.cuda.is_bf16_supported():
                        raise ValueError('BF16 is not supported on the selected CUDA device')
        self.provider, self.network = provider, provider.network
        if not isinstance(self.network, torch.nn.Module) or self.network.training:
            raise ValueError('Provider must contain an eval-mode torch model')
        if not getattr(provider, 'available', True):
            raise ValueError('Provider is unavailable')
        if any(p.device != self.device or p.dtype != torch.float32 for p in self.network.parameters()):
            raise ValueError('Model parameters must remain FP32 on the requested device')
        self.workers, self.precision = workers, precision
        self.pin_memory = pin_memory and self.device.type == 'cuda'
        self._lock = threading.Lock()
        self._closed = False
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._configuration = {
            'schema_version': 1, 'adapter': 'owned_fast_extract', 'source_sha256': source_sha,
            'scope': 'cpu_engineering_only' if diagnostic else 'cuda_extraction_experiment',
            'device': str(self.device), 'workers': workers, 'max_pending_preparations': workers,
            'max_batch': MAX_BATCH, 'precision': precision, 'projection_and_normalization': 'fp32',
            'pin_memory_requested': pin_memory, 'pin_memory': self.pin_memory,
            'h2d_non_blocking': self.pin_memory, 'tf32': False,
            'preprocessing': 'original_image_tensor_256_bicubic_letterbox',
            'decode': 'PIL_open_load_JPEG_PNG_no_EXIF_transform', 'image_cache': False,
            'model_execution': 'one_forward_per_requested_batch_single_consumer',
            'cross_batch_overlap': False,
        }
        binding = {'configuration': self._configuration,
                   'original_inference_fingerprint': getattr(provider, 'inference_fingerprint', None),
                   'model_sha256': getattr(provider, 'model', {}).get('sha256')}
        self.inference_fingerprint = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='owned-image-prep') if workers else None

    @property
    def configuration(self):
        return deepcopy(self._configuration)

    @staticmethod
    def _prepare(path, bbox):
        # Same decode guards and image_tensor as frozen batch.infer_batch.
        # The image is closed even when validation, decoding or preprocessing fails.
        with Image.open(path) as image:
            if image.format not in ('JPEG', 'PNG') or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Only single-frame JPEG/PNG images are supported')
            if len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bbox):
                raise ValueError('bbox requires four integers')
            x, y, width, height = bbox
            if (image.width * image.height > 40_000_000 or x < 0 or y < 0 or width <= 0
                    or height <= 0 or x + width > image.width or y + height > image.height):
                raise ValueError('Invalid bounding box or oversized image')
            image.load()
            return image_tensor(image, bbox, [256, 256])

    def _prepare_batch(self, paths, bboxes):
        if self._pool is None:
            return torch.stack([self._prepare(path, box) for path, box in zip(paths, bboxes)])
        pending, tensors = deque(), []
        rows = iter(zip(paths, bboxes))
        try:
            for _ in range(min(self.workers, len(paths))):
                pending.append(self._pool.submit(self._prepare, *next(rows)))
            while pending:
                tensors.append(pending[0].result())
                pending.popleft()
                row = next(rows, None)
                if row is not None:
                    pending.append(self._pool.submit(self._prepare, *row))
        finally:
            # Running workers own their PIL context. Drain on an early exception.
            for future in pending:
                future.cancel()
            if pending:
                wait(pending)
        return torch.stack(tensors)

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _run(self, paths, bboxes, profiled):
        paths, bboxes = list(paths), list(bboxes)
        if len(paths) != len(bboxes) or len(paths) > MAX_BATCH:
            raise ValueError('Matching paths/bboxes and a batch no larger than 64 are required')
        with self._lock, torch.inference_mode(), _exact_backends():
            if self._closed:
                raise RuntimeError('Extractor is closed')
            if not getattr(self.provider, 'available', True) or self.provider.network is not self.network:
                raise RuntimeError('The original provider was closed or replaced')
            if self.network.training:
                raise ValueError('Model must remain in eval mode')
            stages = dict.fromkeys(('read_preprocess', 'h2d', 'forward', 'd2h', 'total'), 0.0)
            if not paths:
                return np.empty((0, 384), dtype=np.float32), stages
            if profiled:
                self._sync()
            start = mark = time.perf_counter()
            tensor = self._prepare_batch(paths, bboxes)
            if self.pin_memory:
                tensor = tensor.pin_memory()
            now = time.perf_counter()
            stages['read_preprocess'], mark = (now - mark) * 1000, now
            tensor = tensor.to(self.device, non_blocking=self.pin_memory)
            if profiled:
                self._sync()
            now = time.perf_counter()
            stages['h2d'], mark = (now - mark) * 1000, now
            with torch.autocast(device_type=self.device.type, enabled=False):
                if self.precision == 'fp32':
                    vectors = self.network(tensor)
                else:
                    # AMP changes only the backbone's operations, never stored weights.
                    # Keep projection and normalization in FP32 for numerical stability.
                    dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
                    with torch.autocast('cuda', dtype=dtype):
                        features = self.network.backbone.forward_head(
                            self.network.backbone.forward_features(tensor), pre_logits=True)
                    if features.shape != (len(paths), 384):
                        raise ValueError('Unexpected backbone features')
                    vectors = F.normalize(self.network.projection(features.float()).float(), dim=1, eps=1e-12)
            if profiled:
                self._sync()
            now = time.perf_counter()
            stages['forward'], mark = (now - mark) * 1000, now
            result = vectors.float().cpu().numpy().astype(np.float32)
            if profiled:
                self._sync()
            now = time.perf_counter()
            stages['d2h'] = (now - mark) * 1000
            stages['total'] = (now - start) * 1000
            if (result.shape != (len(paths), 384) or not np.isfinite(result).all()
                    or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-4)):
                raise ValueError('Invalid compact embeddings')
            return result, stages

    def extract(self, paths, bboxes):
        return self._run(paths, bboxes, False)[0]

    def profile(self, paths, bboxes):
        """Synchronized diagnostic stages; these are not overlapping throughput timings."""
        vectors, stages = self._run(paths, bboxes, True)
        return {'embeddings': vectors, 'stages_ms': stages, 'configuration': self.configuration,
                'inference_fingerprint': self.inference_fingerprint}

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._pool is not None:
                    self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Extractor is closed')
        return self

    def __exit__(self, *args):
        self.close()
