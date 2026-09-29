"""Checkpoint, ownership and shape contracts; no GPU or trained-model benchmark."""
from copy import deepcopy
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from vehicle.server.e27_adapter import E27Adapter, validate_checkpoint
from vehicle.server.e27_contract import CHECKPOINT_PROVENANCE, DEPENDENCIES, WEIGHTS_BYTES, WEIGHTS_SHA256
from vehicle.server.e27_model import CompactModel, CONFIG, validate_state


class TinyBackbone(nn.Module):
    num_features = 384
    def forward_features(self, value):
        return value.mean((2, 3)).repeat(1, 128)
    def forward_head(self, value, pre_logits=False):
        return value


def record():
    return dict(kind='owned_compact_reid', format_version=1, training_finished=True,
                test_only=False, stand_id=5, selected_epoch=14, config=deepcopy(CONFIG),
                provenance={**deepcopy(CHECKPOINT_PROVENANCE), 'dependencies': deepcopy(DEPENDENCIES)},
                model_state={})


class E27AdapterTests(unittest.TestCase):
    def bundle(self):
        return SimpleNamespace(config={'checkpoint_provenance': deepcopy(CHECKPOINT_PROVENANCE)},
                               weights_path=Path('unused-test-checkpoint.pth'))

    def test_checkpoint_rejects_incomplete_wrong_or_synthetic_model(self):
        validate_checkpoint(record(), self.bundle())
        for key, value in [('training_finished', False), ('test_only', True), ('stand_id', 4),
                           ('selected_epoch', 13), ('kind', 'wrong'), ('format_version', 0)]:
            changed = record(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint(changed, self.bundle())
        for field in ('control_used_for_selection', 'pretrained_sha256'):
            changed = record(); changed['provenance'][field] = 'different'
            with self.assertRaises(ValueError):
                validate_checkpoint(changed, self.bundle())
        changed = record(); changed['provenance']['dependencies']['torch'] = '2.9.0'
        with self.assertRaises(ValueError):
            validate_checkpoint(changed, self.bundle())

    def test_no_cuda_fallback(self):
        with patch('torch.cuda.is_available', return_value=False), self.assertRaises(ValueError):
            E27Adapter(self.bundle(), 'cuda')

    def test_load_uses_native_provider_and_fast_path_and_closes(self):
        network = CompactModel(backbone=TinyBackbone(), test_only=True).eval()
        native = SimpleNamespace(device='cpu', network=network, checkpoint=record(), available=True,
            model={'sha256': WEIGHTS_SHA256}, inference_fingerprint='test-native', close=MagicMock())
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=WEIGHTS_BYTES)), \
             patch('vehicle.server.e27_adapter.sha256', return_value=WEIGHTS_SHA256), \
             patch('vehicle.server.e27_adapter.verify_native_sources'), \
             patch('vehicle.server.e27_adapter.CompactProvider', return_value=native) as loader:
            adapter = E27Adapter(self.bundle(), 'cpu')
        loader.assert_called_once_with(self.bundle().weights_path, device='cpu')
        self.assertEqual(adapter.extractor.configuration['workers'], 4)
        self.assertEqual(adapter.extractor.configuration['precision'], 'fp32')
        image = Image.new('RGB', (40, 30), (120, 50, 90))
        with patch.object(network, 'forward', wraps=network.forward) as forward:
            vectors = adapter.embed_batch([image, image], [(0, 0, 40, 30)] * 2)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(vectors.shape, (2, 384))
        self.assertFalse(adapter.runtime_setup['forward_executed'])
        adapter.close(); adapter.close(); native.close.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'closed'):
            adapter.embed_paths([], [])

    def test_cleanup_cannot_replace_original_load_failure(self):
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=WEIGHTS_BYTES)), \
             patch('vehicle.server.e27_adapter.sha256', return_value=WEIGHTS_SHA256), \
             patch('vehicle.server.e27_adapter.verify_native_sources'), \
             patch('vehicle.server.e27_adapter.CompactProvider', side_effect=RuntimeError('original load failure')), \
             patch.object(E27Adapter, 'close', side_effect=RuntimeError('cleanup failure')):
            with self.assertRaisesRegex(RuntimeError, 'original load failure'):
                E27Adapter(self.bundle(), 'cpu')

    def test_native_sources_are_byte_exact(self):
        from vehicle.server.e27_contract import verify_native_sources
        verify_native_sources()

    def test_owned_batch_outputs_and_single_forward(self):
        model = CompactModel(backbone=TinyBackbone(), test_only=True).eval()
        image = Image.new('RGB', (47, 63), (160, 40, 80))
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            first = model.embed_batch([image, image], [(0, 0, 47, 63)] * 2)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(first.shape, (2, 384))
            self.assertEqual(first.dtype, np.float32)
            saved = first.copy()
            second = model.embed_batch([image], [(0, 0, 47, 63)])
        np.testing.assert_array_equal(first, saved)
        self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 3, 224, 224)))

    def test_state_keys_shape_dtype_and_nonfinite_are_strict(self):
        model = CompactModel(backbone=TinyBackbone(), test_only=True)
        valid = model.state_dict(); validate_state(model, valid)
        for variant in ('missing', 'shape', 'dtype', 'nonfinite'):
            state = {k: v.clone() for k, v in valid.items()}; key = next(iter(state))
            if variant == 'missing': state.pop(key)
            if variant == 'shape': state[key] = state[key][:1]
            if variant == 'dtype': state[key] = state[key].double()
            if variant == 'nonfinite': state[key].view(-1)[0] = float('nan')
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_state(model, state)


if __name__ == '__main__':
    unittest.main()
