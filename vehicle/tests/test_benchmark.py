"""Synthetic providers stay inside tests; measurements here are not model evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vehicle.server.benchmark import load_provider_timed, measure, run_benchmark, weight_inventory
from vehicle.server.provider import sha256
from vehicle.tests.test_batch import FixtureProvider, dataset


class BenchmarkTests(unittest.TestCase):
    def test_bundle_counts_complete_checkpoint_only_and_rejects_unlisted_auxiliary_weights(self):
        from vehicle.tests.test_model_bundle import BundleTests
        from vehicle.server.model_bundle import inspect_bundle, runtime_fingerprint
        fixture = BundleTests('test_metadata_check_is_explicitly_not_inference')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root / 'LICENSE').write_text('Test license text is not a weight file.' * 100)
        bundle = inspect_bundle(fixture.root)
        provider = FixtureProvider()
        provider._bundle_mode = True
        provider.model = {**provider.model, 'device': 'cpu', 'sha256': sha256(fixture.root / 'model.pt')}
        provider.inference_fingerprint = runtime_fingerprint(bundle, 'cpu')
        env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
        env['VEHICLE_MODEL_BUNDLE'] = str(fixture.root)
        with patch.dict(os.environ, env, clear=True):
            evidence = weight_inventory(provider)
            self.assertEqual(evidence['status'], 'verified')
            self.assertEqual(evidence['total_bytes'], (fixture.root / 'model.pt').stat().st_size)
            self.assertEqual(len(evidence['files']), 1)
            self.assertEqual(evidence['files'][0]['file'], 'model.pt')
            (fixture.root / 'auxiliary.pth').write_bytes(b'TEST ONLY unlisted weights')
            with self.assertRaisesRegex(ValueError, 'Лишний файл весов'):
                weight_inventory(provider)

    def test_bundle_cannot_substitute_new_manifest_for_loaded_identity(self):
        from vehicle.tests.test_model_bundle import BundleTests
        from vehicle.server.model_bundle import inspect_bundle, runtime_fingerprint
        fixture = BundleTests('test_metadata_check_is_explicitly_not_inference')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        provider = FixtureProvider()
        provider._bundle_mode = True
        provider.model = {**provider.model, 'device': 'cpu', 'sha256': sha256(fixture.root / 'model.pt')}
        provider.inference_fingerprint = runtime_fingerprint(inspect_bundle(fixture.root), 'cpu')
        (fixture.root / 'model.pt').write_bytes(b'Different TEST ONLY metadata checkpoint')
        fixture.calibration['model_sha256'] = sha256(fixture.root / 'model.pt')
        fixture.save()
        env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
        env['VEHICLE_MODEL_BUNDLE'] = str(fixture.root)
        with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, 'differs from the loaded model'):
            weight_inventory(provider)

    def test_legacy_counts_exact_selected_file_and_detects_same_size_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / 'selected.pth'
            weights.write_bytes(b'TEST ONLY WEIGHTS')
            (root / 'calibration.json').write_text('{"test": true}')
            (root / 'LICENSE').write_text('License text is excluded' * 100)
            provider = FixtureProvider()
            provider.model = {**provider.model, 'sha256': sha256(weights)}
            env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
            env.update(VEHICLE_MODEL=str(weights), VEHICLE_MODEL_SHA256=provider.model['sha256'])
            with patch.dict(os.environ, env, clear=True):
                evidence = weight_inventory(provider)
                self.assertEqual(evidence['total_bytes'], len(b'TEST ONLY WEIGHTS'))
                self.assertEqual(evidence['files'][0]['sha256'], provider.model['sha256'])
                weights.write_bytes(b'X' * len(b'TEST ONLY WEIGHTS'))
                with self.assertRaisesRegex(ValueError, 'changed or differs'):
                    weight_inventory(provider)

    def test_unknown_source_has_no_invented_size(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
        with patch.dict(os.environ, env, clear=True):
            evidence = weight_inventory(FixtureProvider())
        self.assertEqual(evidence['status'], 'unavailable')
        self.assertIsNone(evidence['total_bytes'])
        self.assertEqual(evidence['files'], [])
        self.assertTrue(evidence['reason'])

    def test_unbound_weights_fail_benchmark_before_forward_with_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, query, _ = dataset(root)
            weights = root / 'selected.pth'
            weights.write_bytes(b'TEST ONLY selected but not loaded weights')
            provider = FixtureProvider()
            provider.model = {**provider.model, 'device': 'cpu'}
            env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
            env.update(VEHICLE_MODEL=str(weights), VEHICLE_MODEL_SHA256=sha256(weights))
            output = root / 'performance.json'
            with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, 'does not match'):
                measure(provider, images, query, output, allow_cpu=True)
            evidence = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(evidence['status'], 'failed')
            self.assertEqual(evidence['weights']['status'], 'failed')
            self.assertIsNone(evidence['total_weights_bytes'])
            self.assertIsNone(evidence['latency'])

    def test_initialization_timing_includes_constructor_and_final_cuda_sync(self):
        sequence = []
        provider = FixtureProvider()
        provider.model = {**provider.model, 'device': 'cuda:0'}
        def construct():
            sequence.append('constructor')
            return provider
        with patch.dict(os.environ, {'VEHICLE_DEVICE': 'cuda:0'}), \
             patch('vehicle.server.benchmark.ProductionProvider', side_effect=construct), \
             patch('torch.cuda.is_initialized', return_value=False), \
             patch('torch.cuda.synchronize', side_effect=lambda device: sequence.append('sync')), \
             patch('vehicle.server.benchmark.time.perf_counter', side_effect=[10., 12.]):
            loaded, evidence = load_provider_timed()
        self.assertIs(loaded, provider)
        self.assertEqual(sequence, ['constructor', 'sync'])
        self.assertEqual(evidence['seconds'], 2.)
        self.assertEqual(evidence['status'], 'completed')
        self.assertTrue(evidence['cuda_synchronized'])
        self.assertIn('forward', evidence['excludes'])

    def test_missing_model_preserves_failed_attempt_without_latency(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evidence' / 'performance.json'
            env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
            with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, 'веса'):
                run_benchmark('unused-images', 'unused-input.csv', output, allow_cpu=True)
            evidence = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(evidence['status'], 'failed')
            self.assertEqual(evidence['model_load']['status'], 'unavailable')
            self.assertGreaterEqual(evidence['model_load']['seconds'], 0)
            self.assertFalse(evidence['measured'])
            self.assertIsNone(evidence['latency'])
            self.assertEqual(evidence['throughput'], [])

    def test_failed_constructor_preserves_timing_and_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'performance.json'
            with patch('vehicle.server.benchmark.ProductionProvider', side_effect=RuntimeError('test load failure')), \
                 self.assertRaisesRegex(RuntimeError, 'load failure'):
                run_benchmark('unused-images', 'unused.csv', output)
            evidence = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(evidence['model_load']['status'], 'failed')
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, 'new output'):
                run_benchmark('unused-images', 'unused.csv', output)
            self.assertEqual(output.read_bytes(), original)

    def test_forward_failure_keeps_load_phase_separate_and_honest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, query, _ = dataset(root, ('broken',))
            provider = FixtureProvider()
            provider.model = {**provider.model, 'device': 'cpu'}
            output = root / 'performance.json'
            load = {'status': 'completed', 'seconds': 3.25}
            with self.assertRaisesRegex(ValueError, 'Intentional'):
                measure(provider, images, query, output, allow_cpu=True, model_load=load)
            evidence = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(evidence['model_load'], load)
            self.assertEqual(evidence['status'], 'failed')
            self.assertFalse(evidence['measured'])
            self.assertIsNone(evidence['latency'])
            self.assertTrue(output.with_suffix('.events.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
