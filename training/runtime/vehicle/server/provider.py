"""Operator-owned model artifacts only. Network downloads and fallbacks are absent."""
import hashlib
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ProductionProvider:
    available = False
    reason = 'Обученные веса и проверенная калибровка ещё не подключены.'
    model = None
    threshold = None
    calibration_sha256 = None
    inference_fingerprint = None
    retrieval = None
    retrieval_fingerprint = None

    def __init__(self, device=None):
        # An explicit web choice must not mutate the batch CLI environment.
        self.device = device if device is not None else os.getenv('VEHICLE_DEVICE', 'cpu')
        self.predictor = None
        self._bundle_mode = False
        self.available = False
        self.model = self.threshold = self.calibration_sha256 = self.inference_fingerprint = None
        self.retrieval = self.retrieval_fingerprint = None
        bundle_path = os.getenv('VEHICLE_MODEL_BUNDLE')
        try:
            if any(os.getenv(name) for name in ('VEHICLE_MODEL', 'VEHICLE_MODEL_SHA256', 'VEHICLE_CALIBRATION', 'VEHICLE_CALIBRATION_SHA256')):
                raise ValueError('Прежняя конфигурация модели не поддерживается. Используйте пакет через VEHICLE_MODEL_BUNDLE.')
            if self.device not in ('cpu', 'cuda'):
                raise ValueError('Устройство должно быть cpu или cuda.')
            if self.device == 'cuda':
                from .runtime_settings import device_options
                gpu = next(option for option in device_options() if option['id'] == 'cuda')
                if not gpu['available']:
                    raise ValueError(gpu['reason'])
            if not bundle_path:
                return
            if not (Path(bundle_path) / 'bundle.json').is_file():
                self.reason = 'Пакет выбранной модели ещё не подключён (bundle.json отсутствует).'
                return
            self._load_bundle(bundle_path)
        except ValueError as error:
            self.reason = str(error)
            LOG.warning('Production provider rejected: %s', self.reason)
        except Exception:
            self.reason = 'Не удалось загрузить модель. Проверьте пути, окружение и журнал сервера.'
            LOG.exception('Production provider failed to initialize')

    def _load_bundle(self, directory):
        from .model_bundle import inspect_bundle, runtime_fingerprint
        from .model_adapters import load_adapter
        bundle = inspect_bundle(directory)
        adapter = load_adapter(bundle, self.device)
        try:
            if str(adapter.device).split(':')[0] != self.device:
                raise ValueError('Модель загрузилась на другом устройстве; автоматическая замена запрещена.')
            if adapter.dimension != bundle.manifest['model']['dimension']:
                raise ValueError('Размерность загруженной модели не совпадает с bundle.json.')
            fingerprint = runtime_fingerprint(bundle, adapter.device)
            metadata = {**bundle.manifest['model'], 'sha256': bundle.weights_sha256,
                        'inference_fingerprint': fingerprint, 'device': str(adapter.device)}
            if bundle.retrieval is not None:
                metadata.update(retrieval_fingerprint=bundle.retrieval_fingerprint,
                                score_domain=bundle.retrieval['score_domain'])
            threshold = bundle.threshold_score if bundle.retrieval is not None else (bundle.threshold_cosine + 1) / 2
        except BaseException:
            # CUDA graphs can retain private pools even when the adapter is not
            # published. Cleanup must not replace the original validation error.
            try:
                close = getattr(adapter, 'close', None)
                if callable(close):
                    close()
            except Exception:
                LOG.exception('Could not close rejected model adapter')
            raise
        # Publish only after every validation succeeds; no partially loaded state.
        self.predictor = adapter
        self.model = metadata
        self.threshold = threshold
        self.retrieval = bundle.retrieval
        self.retrieval_fingerprint = bundle.retrieval_fingerprint
        self.calibration_sha256 = bundle.calibration_sha256
        self.inference_fingerprint = fingerprint
        self._bundle_mode = True
        self.available, self.reason = True, None

    def close(self):
        """Release adapter-owned CUDA graphs after callers have drained work."""
        adapter, self.predictor = self.predictor, None
        self.available = False
        self.reason = 'Модель закрыта.'
        close = getattr(adapter, 'close', None)
        if callable(close):
            close()

    def _validate_vectors(self, values, count):
        import numpy as np
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (count, self.model['dimension']) or not np.isfinite(vectors).all():
            raise ValueError('Модель вернула некорректные векторы признаков.')
        if count and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('Модель вернула ненормированные или нулевые векторы.')
        return vectors

    def embed(self, image, bbox, *, image_id):
        if not self.available:
            raise RuntimeError(self.reason)
        result = self.predictor.embed(image, bbox)
        return self._validate_vectors([result], 1)[0]

    def embed_batch(self, images, bboxes, *, image_ids):
        """IDs are bookkeeping only. Batch has no labels or cross-query features."""
        if not self.available:
            raise RuntimeError(self.reason)
        if not len(images) == len(bboxes) == len(image_ids):
            raise ValueError('Число изображений, рамок и ID должно совпадать.')
        if not images:
            import numpy as np
            return np.empty((0, self.model['dimension']), dtype=np.float32)
        values = self.predictor.embed_batch(images, bboxes)
        return self._validate_vectors(values, len(images))
