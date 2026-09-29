"""GPU orchestration guards; fixture providers exist exclusively in tests."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from vehicle.server.gpu_check import hardware_info, require_device, selected_runtime
from vehicle.server.gpu_validation import (check_control_inputs, check_independence,
                                          reference_comparison, run_validation, small_inputs)
from vehicle.tests.test_batch import dataset, FixtureProvider
from vehicle.server.validation import write_outputs


class VisualFixtureProvider(FixtureProvider):
    model = {**FixtureProvider.model, 'device': 'cpu'}

    def embed_batch(self, images, bboxes, *, image_ids):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(images), 1))


class GpuValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images, self.query, self.gallery = dataset(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_cuda_unavailable_cannot_become_cpu_success_and_preserves_evidence(self):
        output = self.root / 'no-gpu'
        with patch('torch.cuda.is_available', return_value=False):
            result = run_validation(self.images, self.query, self.gallery, output, device='cuda')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failed_step'], 'hardware')
        self.assertIn('CPU fallback is disabled', result['error'])
        self.assertFalse(result['submission_ready'])
        self.assertTrue((output / 'hardware.json').is_file())
        self.assertTrue((output / 'gpu-validation.json').is_file())
        self.assertFalse((output / 'open-test').exists())

    def test_explicit_cpu_only_and_no_existing_evidence_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'explicit --allow-cpu-smoke'):
            run_validation(self.images, self.query, self.gallery, self.root / 'unsafe', device='cpu')
        self.assertFalse((self.root / 'unsafe').exists())
        used = self.root / 'used'
        used.mkdir()
        (used / 'keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            run_validation(self.images, self.query, self.gallery, used, device='cpu', allow_cpu_smoke=True)
        self.assertEqual((used / 'keep.txt').read_text(), 'keep')

    def test_cpu_smoke_produces_real_batch_contract_but_never_gpu_readiness(self):
        provider = VisualFixtureProvider()
        with patch('vehicle.server.gpu_validation.load_provider_timed', return_value=(provider, {'status': 'completed'})), \
             patch('vehicle.server.gpu_validation.weight_inventory', return_value={'status': 'verified'}), \
             patch.dict(os.environ, {'VEHICLE_SOURCE_REVISION': 'abc-dirty', 'VEHICLE_SOURCE_MANIFEST_SHA256': 'f' * 64}), \
             patch('vehicle.server.gpu_validation.child_check', side_effect=AssertionError('CPU must not benchmark or run full repeatability')):
            result = run_validation(self.images, self.query, self.gallery, self.root / 'smoke',
                                    device='cpu', allow_cpu_smoke=True, batch_size=4)
        self.assertEqual(result['status'], 'awaiting_gpu', result.get('error'))
        self.assertTrue(result['passed'])
        self.assertFalse(result['all_acceptance_checks_completed'])
        self.assertFalse(result['submission_ready'])
        self.assertEqual(result['source']['revision'], 'abc-dirty')
        self.assertEqual(result['source']['source_manifest_sha256'], 'f' * 64)
        self.assertIn('gpu_validation.py', result['source']['server_sha256'])
        self.assertFalse(result['steps']['open_test']['full_supplied_inputs'])
        self.assertIsNone(result['steps']['benchmark']['passed'])
        self.assertEqual(np.load(self.root / 'smoke/open-test/embeddings.npy').shape, (12, 2))

    def test_id_based_fixture_is_detected_as_inference_leak(self):
        result = check_independence(FixtureProvider(), self.images, self.query, self.gallery)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['opaque_id_independence'])

    def test_cuda_independence_uses_cuda_numeric_portability_tolerance(self):
        provider = VisualFixtureProvider()
        provider.device = 'cuda'
        result = check_independence(provider, self.images, self.query, self.gallery)
        self.assertTrue(result['passed'])
        self.assertEqual(result['tolerance'], {'atol': 2e-4, 'rtol': 1e-4})

    def test_bad_control_fails_before_model_load(self):
        control = self.root / 'control'
        control.mkdir()
        with patch('vehicle.server.gpu_validation.load_provider_timed', side_effect=AssertionError('Model must not load before input checks')):
            result = run_validation(self.images, self.query, self.gallery, self.root / 'bad-control',
                                    device='cpu', allow_cpu_smoke=True, control_dir=control)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failed_step'], 'inputs')

    def test_runtime_environment_restored(self):
        with patch.dict(os.environ, {'VEHICLE_DEVICE': 'cpu', 'VEHICLE_MODEL': 'legacy', 'VEHICLE_MODEL_BUNDLE': 'old'}):
            with selected_runtime('cuda', self.root / 'new'):
                self.assertEqual(os.environ['VEHICLE_DEVICE'], 'cuda')
                self.assertNotIn('VEHICLE_MODEL', os.environ)
                self.assertEqual(Path(os.environ['VEHICLE_MODEL_BUNDLE']), self.root / 'new')
            self.assertEqual(os.environ['VEHICLE_MODEL'], 'legacy')
            self.assertEqual(os.environ['VEHICLE_MODEL_BUNDLE'], 'old')

    def test_reference_transfer_failure_does_not_hide_ranking_difference(self):
        reference, actual = self.root / 'reference', self.root / 'actual'
        reference.mkdir()
        actual.mkdir()
        vectors = np.array([[1, 0]] + [[np.cos(i / 20), np.sin(i / 20)] for i in range(10)], dtype=np.float32)
        np.save(reference / 'embeddings.npy', vectors)
        changed = vectors.copy()
        changed[0] = [-1, 0]
        np.save(actual / 'embeddings.npy', changed)
        report = reference_comparison(reference, actual, 1)
        self.assertFalse(report['passed'])
        self.assertFalse(report['top10_exact'])
        self.assertEqual(report['embedding_cosine_top10_changed_query_indices'], [0])

    def test_reference_checks_actual_submission_policy_and_keeps_historical_mismatch_visible(self):
        reference, actual = self.root / 'reference', self.root / 'actual'
        reference.mkdir()
        actual.mkdir()
        vectors = np.array([[1, 0]] + [[np.cos(i / 20), np.sin(i / 20)] for i in range(10)], dtype=np.float32)
        gids = [f'g{i}' for i in range(10)]
        np.save(reference / 'embeddings.npy', vectors)
        write_outputs(actual, ['q'], gids, vectors[:1], vectors[1:], .8)
        (reference / 'submission.csv').write_text('q,' + ','.join(reversed(gids)) + '\n')
        report = reference_comparison(reference, actual, 1, threshold=.8)
        self.assertTrue(report['passed'])
        self.assertTrue(report['serving_policy_submission_exact'])
        self.assertTrue(report['accepted_pairs_exact'])
        self.assertFalse(report['historical_submission_exact'])
        self.assertTrue(report['ranking_review_required'])
        (actual / 'submission.csv').write_text((reference / 'submission.csv').read_text())
        second = reference_comparison(reference, actual, 1, threshold=.8)
        self.assertFalse(second['serving_policy_submission_exact'])
        self.assertFalse(second['top10_exact'])

    def test_small_subset_is_real_deterministic_and_preserves_coordinates(self):
        q1, g1, qi1, gi1 = small_inputs(self.query, self.gallery, self.root / 'subset1')
        q2, g2, qi2, gi2 = small_inputs(self.query, self.gallery, self.root / 'subset2')
        self.assertEqual(q1.read_bytes(), q2.read_bytes())
        self.assertEqual(g1.read_bytes(), g2.read_bytes())
        self.assertEqual((qi1, gi1), (qi2, gi2))
        self.assertIn(b'query,1,2,20,15', q1.read_bytes())

    def test_frozen_control_manifest_detects_tampering(self):
        import shutil
        from vehicle.server.provider import sha256
        source = Path(__file__).resolve().parents[2] / 'control'
        control = self.root / 'control'
        control.mkdir()
        names = ['query.csv', 'gallery.csv', 'ground_truth.csv', 'embeddings.npy']
        for name in names:
            shutil.copyfile(source / name, control / name)
        manifest = {'files_sha256': {name: sha256(control / name) for name in names}}
        (control / 'source-manifest.json').write_text(json.dumps(manifest))
        check_control_inputs(control)
        self.assertEqual(np.load(control / 'embeddings.npy').shape, (401, 384))
        (control / 'query.csv').write_bytes((control / 'query.csv').read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'differs from its source manifest'):
            check_control_inputs(control)


if __name__ == '__main__':
    unittest.main()
