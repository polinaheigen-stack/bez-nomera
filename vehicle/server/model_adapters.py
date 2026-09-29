"""Offline adapters with checked, model-specific source and weight contracts."""
from __future__ import annotations

import os

from .model_bundle import verify_dependencies


def load_adapter(bundle, device):
    import torch
    if bundle.adapter not in ('e22_ensemble', 'reidkit_single', 'e25_ensemble', 'e27_compact'):
        raise ValueError('Неизвестный адаптер модели.')
    verify_dependencies(bundle)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if device not in ('cpu', 'cuda'):
        raise ValueError('VEHICLE_DEVICE должен быть cpu или cuda.')
    torch.set_num_threads(2 if bundle.adapter == 'e27_compact' else max(1, min(int(os.getenv('VEHICLE_CPU_THREADS', '2')), 16)))
    if bundle.adapter == 'e27_compact':
        from .e27_adapter import E27Adapter
        return E27Adapter(bundle, device)
    if bundle.adapter == 'e25_ensemble':
        from .e25_adapter import E25Adapter
        return E25Adapter(bundle, device)
    if bundle.adapter == 'e22_ensemble':
        from .e22_adapter import E22Adapter
        return E22Adapter(bundle, device)
    from .reidkit_adapter import ReIDKitAdapter
    return ReIDKitAdapter(bundle, device)
