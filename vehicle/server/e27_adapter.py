"""Strict native stand-5 provider and its measured FP32 extraction pipeline."""
from copy import deepcopy
import io
import threading

import torch

from .stand5_native.compact_model import CompactProvider
from .fast_extract import FastExtractor
from .e27_model import CONFIG, PINNED
from .model_bundle import digest_json, sha256
from .e27_contract import WEIGHTS_SHA256, WEIGHTS_BYTES, verify_native_sources


class E27Adapter:
    dimension = 384
    model_batch_size = None
    preferred_batch_size = 32
    supports_path_extraction = True

    def __init__(self, bundle, device):
        self.device = torch.device(device)
        self._lock = threading.RLock()
        self.native_provider = self.extractor = self.network = None
        self.closed = False
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Requested E27 device unavailable; fallback forbidden')
        path = bundle.weights_path
        verify_native_sources()
        if path.stat().st_size != WEIGHTS_BYTES or sha256(path) != WEIGHTS_SHA256:
            raise ValueError('E27 checkpoint differs from the selected stand-5 weights')
        try:
            # The native constructor validates all ten source hashes and exact
            # dependency provenance, including cu126 wheels on CPU deployments.
            self.native_provider = CompactProvider(path, device=device)
            validate_checkpoint(self.native_provider.checkpoint, bundle)
            self.network = self.native_provider.network
            self.extractor = FastExtractor(self.native_provider, workers=4, precision='fp32',
                                           pin_memory=True, allow_cpu_diagnostic=self.device.type == 'cpu')
            if sha256(path) != WEIGHTS_SHA256:
                raise ValueError('E27 checkpoint changed while loading')
            self.runtime_setup = {'mode': 'fast_fp32_workers4_batch32', 'status': 'ready',
                'batch_policy': 'up_to_32_ordered_inputs_one_forward_per_batch',
                'inference_precision': CONFIG['inference_precision'], 'tf32': False,
                'forward_executed': False, 'warmup_executed': False, 'capture_executed': False,
                'original_stand5_gpu_completed': True, 'selected_extractor_gpu_completed': True,
                'selected_extractor_quality_accepted': True, 'integrated_e27_gpu_acceptance': 'not_performed',
                'extractor': self.extractor.configuration,
                'native_inference_fingerprint': self.native_provider.inference_fingerprint,
                'extractor_inference_fingerprint': self.extractor.inference_fingerprint}
            self.runtime_metadata = deepcopy(self.runtime_setup)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def embed_paths(self, paths, bboxes):
        with self._lock:
            if self.closed or self.extractor is None:
                raise ValueError('E27 model is closed')
            paths, bboxes = list(paths), list(bboxes)
            if len(paths) != len(bboxes) or len(paths) > 32:
                raise ValueError('E27 requires matching paths and bboxes, batch at most 32')
            return self.extractor.extract(paths, bboxes)

    def embed_batch(self, images, bboxes):
        # Compatibility for in-memory callers only. Lossless PNG preserves the
        # decoded pixels; production API/CLI use original paths directly.
        if len(images) != len(bboxes):
            raise ValueError('Image/bbox lengths differ')
        buffers = []
        try:
            for image in images:
                stream = io.BytesIO(); image.save(stream, format='PNG'); stream.seek(0); buffers.append(stream)
            return self.embed_paths(buffers, bboxes)
        finally:
            for stream in buffers:
                stream.close()

    def embed(self, image, bbox):
        return self.embed_batch([image], [bbox])[0]

    def close(self):
        with self._lock:
            if self.closed:
                return
            try:
                if self.extractor is not None:
                    self.extractor.close()
            finally:
                try:
                    if self.native_provider is not None:
                        if self.device.type == 'cuda' and self.network is not None:
                            torch.cuda.synchronize(self.device)
                        self.native_provider.close()
                        self.native_provider.predictor = None
                finally:
                    self.extractor = self.native_provider = self.network = None
                    self.closed = True


def validate_checkpoint(record, bundle):
    if (not isinstance(record, dict) or record.get('kind') != 'owned_compact_reid'
            or record.get('format_version') != 1 or record.get('training_finished') is not True
            or record.get('test_only') is not False or record.get('stand_id') != 5
            or type(record.get('selected_epoch')) is not int or record['selected_epoch'] != 14
            or digest_json(record.get('config')) != digest_json(CONFIG)):
        raise ValueError('Expected completed stand-5 epoch-14 compact checkpoint')
    provenance = record.get('provenance', {})
    expected = bundle.config['checkpoint_provenance']
    if digest_json({key: provenance.get(key) for key in expected}) != digest_json(expected):
        raise ValueError('Checkpoint training/source provenance differs from the frozen selection')
    versions = provenance.get('dependencies', {})
    for name, expected_version in PINNED.items():
        version = versions.get(name, '')
        if (version.split('+')[0] if name in ('torch', 'torchvision') else version) != expected_version:
            raise ValueError('Checkpoint dependency provenance differs: ' + name)
