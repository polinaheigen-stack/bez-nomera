"""Path-based FP32 integration checks; fixtures never enter production inference."""
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import torch

from vehicle.server.app import create_app
from vehicle.server.batch import infer_batch
from vehicle.server.fast_extract import FastExtractor
from vehicle.server.e27_model import CompactModel
from vehicle.tests.test_e27_adapter import TinyBackbone
from vehicle.tests.test_api import multipart


class PathsProvider:
    available, reason, threshold = True, None, .8
    supports_path_extraction, preferred_batch_size = True, 32
    calibration_sha256 = 'b' * 64
    model = {'name': 'TEST ONLY', 'version': '1', 'sha256': 'a' * 64, 'dimension': 4, 'device': 'cpu-test'}
    def __init__(self): self.calls = []
    def embed_paths(self, paths, bboxes, *, image_ids):
        assert len(paths) == len(bboxes) == len(image_ids)
        assert all(Path(p).is_file() for p in paths)
        self.calls.append(list(image_ids))
        return np.tile(np.array([1, 0, 0, 0], np.float32), (len(paths), 1))
    def embed(self, *a, **k): raise AssertionError('Path pipeline must not decode through legacy embed')
    def embed_batch(self, *a, **k): raise AssertionError('Path pipeline must not use legacy decoded batches')


class E27PathsTests(unittest.TestCase):
    def test_native_fast_cpu_single_batch_and_error_recovery(self):
        torch.set_num_threads(2)
        network = CompactModel(backbone=TinyBackbone(), test_only=True).eval()
        provider = SimpleNamespace(device='cpu', network=network, available=True,
                                   model={'sha256': 'test'}, inference_fingerprint='test')
        with tempfile.TemporaryDirectory() as temporary:
            paths, boxes, images = [], [], []
            for i in range(4):
                image = Image.new('RGB', (47 + i, 65), (40 + i, 90, 180))
                path = Path(temporary) / f'{i}.png';image.save(path)
                paths.append(path);boxes.append((1, 2, 40, 55));images.append(image)
            native = network.embed_batch(images, boxes)
            with FastExtractor(provider, workers=4, precision='fp32', pin_memory=True, allow_cpu_diagnostic=True) as fast:
                with patch.object(network, 'forward', wraps=network.forward) as forward:
                    batch = fast.extract(paths, boxes)
                    self.assertEqual(forward.call_count, 1)
                np.testing.assert_array_equal(batch, native)
                kept = batch.copy()
                singles = np.concatenate([fast.extract([p], [b]) for p, b in zip(paths, boxes)])
                np.testing.assert_allclose(singles, batch, atol=1e-6, rtol=1e-6)
                np.testing.assert_array_equal(batch, kept)
                self.assertFalse(np.shares_memory(batch, singles))
                with self.assertRaises(ValueError): fast.extract(paths, [(0, 0, 999, 999)] + boxes[1:])
                np.testing.assert_array_equal(fast.extract(paths, boxes), batch)
            with self.assertRaisesRegex(RuntimeError, 'closed'): fast.extract([], [])

    def test_batch_selects_paths_without_decoding_in_legacy_context(self):
        provider = PathsProvider()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'a.png';Image.new('RGB', (41, 32)).save(path)
            rows = [{'image_id': 'opaque', 'bbox': [1, 2, 30, 20]}]
            with patch('vehicle.server.batch.decoded_images', side_effect=AssertionError('legacy decode')):
                values, info = infer_batch(provider, rows, {'opaque': path})
            self.assertEqual(values.shape, (1, 4));self.assertEqual(info[0]['width'], 41)
            self.assertEqual(provider.calls, [['opaque']])

    def test_web_gallery_and_queries_use_ordered_32_item_batches(self):
        provider = PathsProvider()
        with tempfile.TemporaryDirectory() as temporary, TestClient(create_app(provider, temporary)) as client:
            def wait(url, pending):
                limit = time.monotonic() + 10
                while time.monotonic() < limit:
                    data = client.get(url).json()
                    if data['status'] not in pending: return data
                    time.sleep(.01)
                self.fail('Background job did not finish')
            gids = [f'g{i}' for i in range(40)];qids = [f'q{i}' for i in range(33)]
            data, files = multipart(gids, 'name', 'Batch gallery')
            result = client.post('/api/v1/galleries', data=data, files=files);self.assertEqual(result.status_code, 202)
            gid = result.json()['id'];self.assertEqual(wait('/api/v1/galleries/' + gid, {'indexing'})['status'], 'ready')
            data, files = multipart(qids, 'gallery_id', gid)
            result = client.post('/api/v1/runs', data=data, files=files);self.assertEqual(result.status_code, 202)
            run = wait('/api/v1/runs/' + result.json()['id'], {'queued', 'running'})
            self.assertEqual(run['status'], 'completed');self.assertEqual(run['failed'], 0)
            self.assertEqual(provider.calls, [gids[:32], gids[32:], qids[:32], qids[32:]])
            self.assertEqual([row['query']['image_id'] for row in run['results']], qids)
            self.assertTrue(run['export_ready']);self.assertEqual(client.get(run['artifacts_url']).status_code, 200)


if __name__ == '__main__': unittest.main()
