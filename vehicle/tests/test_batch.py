"""Synthetic test vectors verify infrastructure only; they are never production weights."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from vehicle.server.batch import run_batch
from vehicle.server.validation import validate_export, write_outputs


class FixtureProvider:
    available = True
    reason = None
    threshold = .8
    calibration_sha256 = 'b' * 64
    inference_fingerprint = 'c' * 64
    model = {'name': 'SYNTHETIC TEST ONLY', 'version': 'test', 'sha256': 'a' * 64, 'inference_fingerprint': 'c' * 64, 'dimension': 2, 'device': 'cpu-test'}

    def embed_batch(self, images, bboxes, *, image_ids):
        if 'broken' in image_ids:
            raise ValueError('Intentional test failure')
        return np.array([[-1, 0] if image_id == 'reject' else [1, 0] for image_id in image_ids], dtype=np.float32)


def dataset(root, query_ids=('query', 'reject')):
    images = root / 'images'
    images.mkdir(exist_ok=True)
    for name, ids in (('query', query_ids), ('gallery', [f'g{i}' for i in range(10)])):
        with (root / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
            for image_id in ids:
                writer.writerow([image_id, 1, 2, 20, 15])
                Image.new('RGB', (40, 30), '#6789ab').save(images / f'{image_id}.png')
    return images, root / 'query.csv', root / 'gallery.csv'


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_direct_files_no_archive_and_no_web_upload_or_storage_quota(self):
        images, query, gallery = dataset(self.root)
        output = self.root / 'out'
        with patch.dict(os.environ, {'VEHICLE_MAX_STORAGE_MB': '0'}), patch('vehicle.server.service.Service', side_effect=AssertionError('HTTP service must not be used')):
            proof = run_batch(FixtureProvider(), images, query, gallery, output, batch_size=4)
        self.assertEqual(proof['status'], 'completed')
        self.assertFalse(any(output.rglob('*.zip')))
        self.assertEqual({path.name for path in output.iterdir()}, {'submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json', 'provenance.json', 'events.jsonl', 'validation.json'})
        self.assertEqual(proof['input_order']['query'], ['query', 'reject'])
        self.assertEqual(proof['validation']['rejected_query_count'], 1)
        self.assertEqual(proof['inference_fingerprint'], FixtureProvider.inference_fingerprint)
        self.assertEqual(np.load(output / 'embeddings.npy').shape, (12, 2))
        self.assertFalse(proof['metrics_measured'])
        self.assertEqual(len(proof['inputs']['gallery']), 10)

    def test_batch_failure_is_not_a_rejection_or_partial_contest_export(self):
        images, query, gallery = dataset(self.root, ('query', 'broken'))
        output = self.root / 'failed'
        with self.assertRaisesRegex(ValueError, 'Intentional'):
            run_batch(FixtureProvider(), images, query, gallery, output)
        self.assertFalse((output / 'submission.csv').exists())
        self.assertEqual(json.loads((output / 'provenance.json').read_text())['status'], 'failed')

    def test_changed_input_csv_cannot_publish_final_export(self):
        images, query, gallery = dataset(self.root)
        original_query = query.read_bytes()
        provider = FixtureProvider()
        original_embed = provider.embed_batch
        def mutate_csv(*args, **kwargs):
            values = original_embed(*args, **kwargs)
            query.write_bytes(original_query.replace(b',20,15', b',19,15'))
            return values
        provider.embed_batch = mutate_csv
        output = self.root / 'changed-input'
        with self.assertRaisesRegex(ValueError, 'CSV changed during inference'):
            run_batch(provider, images, query, gallery, output, batch_size=4)
        self.assertFalse((output / 'submission.csv').exists())
        self.assertFalse((output / 'embeddings.npy').exists())
        proof = json.loads((output / 'provenance.json').read_text())
        self.assertEqual(proof['status'], 'failed')
        self.assertEqual(proof['input_csv_sha256']['query'], hashlib.sha256(original_query).hexdigest())

    def test_cli_fails_without_model_and_does_not_create_export(self):
        images, query, gallery = dataset(self.root)
        env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
        result = subprocess.run([sys.executable, '-m', 'vehicle.server.batch', '--images', str(images), '--query-csv', str(query), '--gallery-csv', str(gallery), '--output', str(self.root / 'out')], env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'out').exists())

    def test_arbitrary_original_dimensions_and_batch_independence(self):
        images, query, gallery = dataset(self.root)
        Image.new('RGB', (1920, 1080), '#6789ab').save(images / 'query.png')
        first, second = self.root / 'single', self.root / 'batched'
        run_batch(FixtureProvider(), images, query, gallery, first, batch_size=1)
        run_batch(FixtureProvider(), images, query, gallery, second, batch_size=8)
        for name in ('submission.csv', 'candidates.csv', 'embeddings.npy'):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertEqual(json.loads((first / 'provenance.json').read_text())['inputs']['query'][0]['width'], 1920)

    def test_malformed_exports_fail_closed(self):
        query_ids, gallery_ids = ['q', 'reject'], [f'g{i}' for i in range(10)]
        output = self.root / 'out'
        qv = np.array([[1, 0], [-1, 0]], dtype=np.float32)
        gv = np.tile(np.array([[1, 0]], dtype=np.float32), (10, 1))
        write_outputs(output, query_ids, gallery_ids, qv, gv, .8)
        baseline = (output / 'submission.csv').read_bytes()
        (output / 'submission.csv').write_bytes(b'query_id,gallery_id_1\n' + baseline)
        with self.assertRaisesRegex(ValueError, 'no header'):
            validate_export(output, query_ids, gallery_ids)
        (output / 'submission.csv').write_bytes(baseline)
        np.save(output / 'embeddings.npy', np.concatenate([qv, gv]).astype(np.float64))
        with self.assertRaisesRegex(ValueError, 'float32'):
            validate_export(output, query_ids, gallery_ids)
        write_outputs(output, query_ids, gallery_ids, qv, gv, .8)
        with (output / 'candidates.csv').open('a', encoding='utf-8') as stream:
            stream.write('reject,g0,NaN\n')
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_export(output, query_ids, gallery_ids)

    def test_unknown_or_duplicate_ranked_ids_and_wrong_candidate_decisions_fail(self):
        q, g = ['q'], [f'g{i}' for i in range(10)]
        out = self.root / 'out'
        qv, gv = np.array([[1, 0]], np.float32), np.tile(np.array([[1, 0]], np.float32), (10, 1))
        write_outputs(out, q, g, qv, gv, .8)
        with (out / 'submission.csv').open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow(['q'] + ['g0'] * 10)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            validate_export(out, q, g)
        write_outputs(out, q, g, qv, gv, .8)
        (out / 'candidates.csv').write_text('query_id,gallery_id,confidence\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'decisions'):
            validate_export(out, q, g, threshold=.8, check_ranking=True)


if __name__ == '__main__':
    unittest.main()
