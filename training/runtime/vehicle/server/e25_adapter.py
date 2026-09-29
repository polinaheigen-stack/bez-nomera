"""E25: native members with one canonical per-image inference path."""
import threading
import numpy as np


def concatenate_members(values, weights):
    """Do not average unrelated coordinates. Dot products equal weighted scores."""
    if len(values) != len(weights) or len(values) != 2:
        raise ValueError('E25 requires exactly two component vectors')
    parts = []
    count = None
    for value, weight in zip(values, weights):
        part = np.asarray(value, dtype=np.float32)
        if part.ndim != 2 or part.shape[1] != 512 or not np.isfinite(part).all():
            raise ValueError('Invalid E25 component embeddings')
        if count is not None and len(part) != count:
            raise ValueError('E25 component batch sizes differ')
        count = len(part)
        norm = np.linalg.norm(part, axis=1, keepdims=True)
        if np.any(norm <= 0) or not np.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid E25 component normalization or coefficient')
        parts.append(part / norm * np.float32(np.sqrt(weight)))
    if not np.isclose(sum(weights), 1.0):
        raise ValueError('E25 coefficients must sum to one')
    fused = np.concatenate(parts, axis=1)
    # Same float32 arithmetic as the frozen cached-vector experiment.
    return (fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)).astype(np.float32)


class E25Adapter:
    dimension = 1024
    # CUDA AMP kernels may change their arithmetic when batch shape changes.
    # Use this shape in web, CLI and benchmarks, including the final partial
    # batch. Outer batches group I/O only; they are not model batching.
    model_batch_size = 1

    def __init__(self, bundle, device):
        from .reidkit_adapter import ReIDKitAdapter
        from .e25_runtime import E25Runtime, configure_threads
        self._lock = threading.RLock()
        self._closed = False
        self._runtime = None
        self.members = []
        self.runtime_setup = {'status': 'not_started'}
        if len(bundle.member_bundles) != 2:
            raise ValueError('Verified E25 member bundles required')
        runtime_config = configure_threads(bundle.config.get('inference', {}).get('runtime'))
        try:
            for member in bundle.member_bundles:
                self.members.append(ReIDKitAdapter(member, device))
            self.device = self.members[0].device
            if any(member.device != self.device for member in self.members):
                raise ValueError('All E25 members must run on the selected device')
            self.weights = [entry['weight'] for entry in bundle.config['members']]
            self._runtime = E25Runtime(self.members, self.weights, self.device,
                                       runtime_config, concatenate_members)
            self.runtime_setup = self._runtime.setup()
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f'Adapter cleanup also failed: {cleanup_error}')
            raise

    @property
    def runtime_metadata(self):
        return dict(self.runtime_setup)

    def _require_open(self):
        if self._closed or self._runtime is None:
            raise RuntimeError('E25 adapter is closed or not initialized.')

    def _embed_one(self, image, bbox):
        with self._lock:
            self._require_open()
            return self._runtime.embed_one(image, bbox)

    def embed(self, image, bbox):
        return self._embed_one(image, bbox)

    def embed_batch(self, images, bboxes):
        with self._lock:
            self._require_open()
            if len(images) != len(bboxes):
                raise ValueError('Images and bounding boxes differ in length')
            if not images:
                return np.empty((0, self.dimension), dtype=np.float32)
            return np.stack([self._embed_one(image, bbox)
                             for image, bbox in zip(images, bboxes)]).astype(np.float32)

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self._runtime is not None:
                    self._runtime.close()
            finally:
                if self._runtime is not None:
                    self.runtime_setup = dict(self._runtime.metadata)
                self._runtime = None
                self.members.clear()
                self._closed = True
