"""Persistent E25 executor with native numerics and explicit startup work.

CUDA Graphs follow https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs.
This executor owns its graph buffers and callables; it never patches adapters.
"""
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import torch
from PIL import Image


CUDA_MODES = ('overlap', 'cuda_graphs', 'compiled_overlap', 'cuda_graphs_preprocess', 'compiled_precision_preprocess')


def validate_runtime_config(config):
    required = {'cuda_mode', 'cpu_threads', 'interop_threads', 'cpu_mode'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Explicit E25 inference.runtime configuration is required.')
    if config['cuda_mode'] not in CUDA_MODES or config['cpu_mode'] != 'eager':
        raise ValueError('Unsupported E25 CUDA/CPU execution mode; fallback is forbidden.')
    if type(config['cpu_threads']) is not int or not 1 <= config['cpu_threads'] <= 32:
        raise ValueError('E25 cpu_threads must be an integer between 1 and 32.')
    if type(config['interop_threads']) is not int or config['interop_threads'] != 1:
        raise ValueError('E25 requires interop_threads=1.')
    return dict(config)


def configure_threads(config):
    """Apply the bundle's process-wide threading contract before model loading."""
    config = validate_runtime_config(config)
    if torch.get_num_interop_threads() != config['interop_threads']:
        try:
            torch.set_num_interop_threads(config['interop_threads'])
        except RuntimeError as error:
            raise ValueError('Cannot apply E25 interop_threads; restart this process.') from error
    if torch.get_num_threads() != config['cpu_threads']:
        torch.set_num_threads(config['cpu_threads'])
    if (torch.get_num_threads() != config['cpu_threads']
            or torch.get_num_interop_threads() != config['interop_threads']):
        raise ValueError('E25 process thread counts do not match the bundle.')
    return config


def scheduled_one(image, bbox, prepare, launch, download, fuse, *, synchronize):
    """Start both GPU forwards before their blocking result copies.

    CPU inputs stay alive until the transfers complete. Preparing Swin may
    overlap the queued DINO forward; both use the same ordered CUDA stream.
    """
    inputs, pending = [], []
    try:
        for index in range(2):
            value = prepare(index, image, bbox)
            inputs.append(value)
            pending.append(launch(index, value))
        return fuse([download(value) for value in pending])
    except BaseException as error:
        # Keep host inputs and pending outputs in this frame while outstanding
        # H2D/forward work completes. A later preparation or launch can fail
        # after an earlier asynchronous copy already acquired these buffers.
        try:
            synchronize()
        except BaseException as cleanup_error:
            error.add_note(f'Pending CUDA work synchronization also failed: {cleanup_error}')
        raise


class E25Runtime:
    def __init__(self, members, weights, device, config, concatenate):
        self.config = validate_runtime_config(config)
        self.members = list(members)
        self.weights = tuple(weights)
        self.device = torch.device(device)
        self.concatenate = concatenate
        self.mode = self.config['cuda_mode'] if self.device.type == 'cuda' else self.config['cpu_mode']
        if self.device.type not in ('cpu', 'cuda') or len(self.members) != 2:
            raise ValueError('E25 requires exactly two members on CPU or CUDA.')
        names = ('vit_large_patch14_dinov2.lvd142m',
                 'swin_base_patch4_window12_384.ms_in22k_ft_in1k')
        for member, size, name in zip(self.members, ((336, 336), (384, 384)), names):
            if (torch.device(member.device) != self.device or member.dimension != 512
                    or tuple(member.config.get('image_size', ())) != size
                    or member.config.get('timm_model') != name
                    or member.config.get('flip_tta', False) or member.model.training):
                raise ValueError('Expected native DINO336/Swin384, eval mode and no flip TTA.')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; eager/CPU fallback is forbidden.')
        self.models = [member.model for member in self.members]
        self.lock = threading.RLock()
        self.graphs, self.static_inputs, self.static_outputs = [], [], []
        self.ready = self.closed = self.cuda_used = False
        self.dtype = None
        self.metadata = {
            'status': 'setup_required', 'mode': self.mode, 'device': str(self.device),
            'model_batch_size': 1, 'cpu_threads': self.config['cpu_threads'],
            'interop_threads': self.config['interop_threads'], 'thread_scope': 'process-wide',
            'source': Path(__file__).name,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'image_sizes': [[336, 336], [384, 384]], 'amp_dtype': None,
            'preprocessing': ('native PIL geometry and float32 arithmetic; contiguous CHW before normalization'
                              if self.mode in ('cuda_graphs_preprocess', 'compiled_precision_preprocess') else 'unchanged native image_tensor'),
            'fusion': 'unchanged native concatenate_members',
            'graph_pool_sharing': False, 'weights_modified': False,
            'startup_validation_calls': 0,
            'forward_executed': False, 'warmup_executed': False, 'capture_executed': False,
            'synthetic_startup_input': None,
            'precision_cast_emulation': self.mode == 'compiled_precision_preprocess',
        }
        if self.mode in ('cuda_graphs_preprocess', 'compiled_precision_preprocess'):
            preprocessing_path = Path(__file__).with_name('e25_preprocess.py')
            self.metadata.update(preprocessing_source=preprocessing_path.name,
                                 preprocessing_source_sha256=hashlib.sha256(preprocessing_path.read_bytes()).hexdigest())

    def _prepare(self, index, image, bbox):
        if self.mode in ('cuda_graphs_preprocess', 'compiled_precision_preprocess'):
            from .e25_preprocess import batch_one
            return batch_one(image, bbox, self.members[index].config['image_size'])
        from .reidkit_adapter import image_tensor
        return torch.stack([image_tensor(image, bbox, self.members[index].config['image_size'])])

    def _forward_normalized(self, index, tensor):
        with torch.autocast('cuda', dtype=self.dtype):
            vectors = self.models[index](tensor)['embedding']
        # Native ReIDKitAdapter normalizes float32 OUTSIDE the autocast context.
        return torch.nn.functional.normalize(vectors.float(), dim=1)

    def _launch(self, index, tensor):
        if self.mode in ('cuda_graphs', 'cuda_graphs_preprocess'):
            self.static_inputs[index].copy_(tensor, non_blocking=True)
            self.graphs[index].replay()
            return self.static_outputs[index]
        return self._forward_normalized(index, tensor.to(self.device, non_blocking=True))

    @staticmethod
    def _download(value):
        # Blocking .cpu(), followed by owned NumPy storage. No graph buffer or
        # compiled output is retained by the caller after the next invocation.
        return value.cpu().numpy().astype(np.float32)

    def _fuse(self, values):
        fused = self.concatenate(values, self.weights)
        if fused.shape != (1, 1024):
            raise ValueError('Invalid E25 single-image output shape.')
        return fused[0]

    def _run_cuda(self, image, bbox):
        if self.mode in ('compiled_overlap', 'compiled_precision_preprocess'):
            mark = getattr(torch.compiler, 'cudagraph_mark_step_begin', None)
            if callable(mark):
                mark()
        return scheduled_one(image, bbox, self._prepare, self._launch,
                             self._download, self._fuse,
                             synchronize=lambda: torch.cuda.synchronize(self.device))

    def _capture_member(self, index, image, bbox):
        tensor = self._prepare(index, image, bbox).to(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                warmup_output = self._forward_normalized(index, tensor)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        del warmup_output
        graph = torch.cuda.CUDAGraph()
        # No pool argument: each member owns a separate private memory pool.
        with torch.cuda.graph(graph, stream=stream):
            output = self._forward_normalized(index, tensor)
        torch.cuda.synchronize(self.device)
        self.graphs.append(graph)
        self.static_inputs.append(tensor)
        self.static_outputs.append(output)

    def setup(self):
        """Prepare once before provider publication; startup is never latency."""
        with self.lock:
            if self.ready or self.closed:
                raise RuntimeError('E25 runtime setup requires a fresh executor.')
            started = time.perf_counter()
            try:
                if self.device.type == 'cuda':
                    self.cuda_used = True
                    with torch.cuda.device(self.device), torch.inference_mode():
                        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        self.metadata['amp_dtype'] = str(self.dtype)
                        torch.cuda.synchronize(self.device)
                        # A constant image initializes kernels with the native
                        # fixed shapes. It is never exported as a search result.
                        with Image.new('RGB', (384, 384), (124, 116, 104)) as image:
                            bbox = (0, 0, 384, 384)
                            self.metadata['synthetic_startup_input'] = {
                                'kind': 'constant RGB', 'size': [384, 384],
                                'purpose': 'runtime preparation only; not a measured input'}
                            if self.mode == 'compiled_precision_preprocess':
                                options = {'triton.cudagraphs': True, 'emulate_precision_casts': True}
                                self.models = [torch.compile(
                                    model, backend='inductor', options=dict(options),
                                    fullgraph=False, dynamic=False) for model in self.models]
                                self.metadata['compile'] = {
                                    'backend': 'inductor', 'options': options,
                                    'fullgraph': False, 'dynamic': False,
                                    'compiler_caches_cleared': False}
                            elif self.mode == 'compiled_overlap':
                                self.models = [torch.compile(
                                    model, backend='inductor', mode='reduce-overhead',
                                    fullgraph=False, dynamic=False) for model in self.models]
                                self.metadata['compile'] = {
                                    'backend': 'inductor', 'mode': 'reduce-overhead',
                                    'fullgraph': False, 'dynamic': False,
                                    'compiler_caches_cleared': False}
                            elif self.mode in ('cuda_graphs', 'cuda_graphs_preprocess'):
                                for index in range(2):
                                    self._capture_member(index, image, bbox)
                                self.metadata.update(graph_count=len(self.graphs),
                                                     graph_warmup_calls_per_member=3,
                                                     warmup_executed=True, capture_executed=True)
                            self._run_cuda(image, bbox)
                            self.metadata['startup_validation_calls'] = 1
                            self.metadata['forward_executed'] = True
                            self.metadata['warmup_executed'] = True
                        torch.cuda.synchronize(self.device)
                self.ready = True
                self.metadata['status'] = 'ready'
            except BaseException as error:
                self.metadata['status'] = 'setup_failed'
                try:
                    self.close()
                except Exception as cleanup_error:
                    error.add_note(f'Runtime cleanup also failed: {cleanup_error}')
                raise
            finally:
                self.metadata['setup_seconds'] = time.perf_counter() - started
            return dict(self.metadata)

    def embed_one(self, image, bbox):
        with self.lock:
            if not self.ready or self.closed:
                raise RuntimeError('E25 executor is not ready or has been closed.')
            if self.device.type == 'cpu':
                return self._fuse([member.embed_batch([image], [bbox]) for member in self.members])
            with torch.cuda.device(self.device), torch.inference_mode():
                return self._run_cuda(image, bbox)

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                if self.cuda_used:
                    torch.cuda.synchronize(self.device)
            finally:
                self.graphs.clear()
                self.static_inputs.clear()
                self.static_outputs.clear()
                self.models.clear()
                self.members.clear()
                self.ready = False
                self.closed = True
                if self.metadata['status'] != 'setup_failed':
                    self.metadata['status'] = 'closed'
