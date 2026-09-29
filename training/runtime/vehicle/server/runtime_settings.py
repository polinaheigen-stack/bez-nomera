"""Persistent web runtime choice, independent of batch command-line defaults."""
import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


class UnavailableProvider:
    """Absence of a loaded model, never an inference substitute."""
    available = False
    reason = 'Загрузка выбранной модели…'
    model = threshold = calibration_sha256 = inference_fingerprint = None


def read_device(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if value.get('device') in ('cpu', 'cuda'):
            return value['device']
        raise ValueError('Invalid saved device')
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError):
        LOG.warning('Cannot read saved web device; defaulting to GPU', exc_info=True)
    return 'cuda'


def device_options():
    """Run outside request locks: importing torch and probing CUDA can be slow."""
    cpu = {'id': 'cpu', 'name': 'CPU', 'available': True, 'reason': None}
    gpu = {'id': 'cuda', 'name': 'GPU · NVIDIA CUDA', 'available': False, 'reason': None}
    try:
        import torch
        if not torch.version.cuda:
            gpu['reason'] = 'Установлена сборка PyTorch без CUDA. Для GPU нужна сборка с поддержкой CUDA.'
        elif not torch.cuda.is_available():
            gpu['reason'] = 'CUDA недоступна: проверьте NVIDIA GPU, драйвер и доступ GPU из контейнера.'
        else:
            gpu.update(name=f'GPU · {torch.cuda.get_device_name(0)}', available=True)
    except ImportError:
        cpu.update(available=False, reason='PyTorch не установлен в окружении сервера.')
        gpu['reason'] = cpu['reason']
    except Exception:
        gpu['reason'] = 'Не удалось проверить CUDA. Подробности в журнале сервера.'
        LOG.exception('CUDA availability probe failed')
    return [gpu, cpu]


def release_cuda_cache():
    """Return unused CUDA reservations after the last GPU provider is dropped."""
    import gc
    import sys
    gc.collect()
    torch = sys.modules.get('torch')
    try:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        # Cache cleanup must never undo an already published working model.
        LOG.warning('Could not release unused CUDA memory', exc_info=True)
