"""Bundle rejection and exact preprocessing parity. Fixtures are not trained weights."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib.metadata
import io
import json
import os
import unittest
from unittest.mock import patch


from vehicle.server.model_bundle import inspect_bundle, main, verify_dependencies, runtime_fingerprint
from vehicle.tests.bundle_fixture import E22MetadataFixture
from vehicle.server.provider import ProductionProvider


class BundleTests(E22MetadataFixture, unittest.TestCase):
    def test_metadata_check_is_explicitly_not_inference(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['check', '--bundle', str(self.root)]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['status'], 'metadata_only')
        self.assertFalse(report['inference_executed'])
        self.assertEqual(len(report['bundle_inference_fingerprint']), 64)
        self.assertIsNone(report['inference_fingerprint'])

    def test_load_rejects_metadata_fixture_as_actual_checkpoint(self):
        # The fixture deliberately has no trained tensor state. The source gate
        # also fails. --load must not turn metadata success into fake readiness.
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(['check', '--bundle', str(self.root), '--load']), 2)

    def test_template_unknown_adapter_and_malformed_identity_rejected(self):
        for field, value in (('status', 'pending_model'), ('adapter', 'user.module:load'), ('adapter', 'legacy_baseline'), ('adapter', 'five_stands'), ('schema_version', True),
                             ('model', {'name': '', 'version': 'x', 'dimension': 6})):
            original = deepcopy(self.manifest)
            self.manifest[field] = value
            self.save_manifest()
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspect_bundle(self.root)
            self.manifest = original
        self.save_manifest()

    def test_artifact_path_escape_and_tampering_rejected(self):
        for filename in ('../model.pt', '/absolute/path.pt', 'nested\\model.pt'):
            self.manifest['artifacts']['weights']['file'] = filename
            self.save_manifest()
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                inspect_bundle(self.root)
        self.manifest['artifacts']['weights']['file'] = 'model.pt'
        self.save_manifest()
        (self.root / 'model.pt').write_bytes(b'Modified after hashing')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            inspect_bundle(self.root)

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        (self.root / 'bundle.json').write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Повторяющийся'):
            inspect_bundle(self.root)

    def test_calibration_exact_code_protocol_and_threshold_required(self):
        changes = [('code_hash', 'e' * 64), ('model_sha256', 'e' * 64),
                   ('calibration_protocol', 'pairwise'), ('threshold_cosine', True),
                   ('threshold_cosine', float('nan')), ('threshold_cosine', -1.1),
                   ('threshold_cosine', 1.1)]
        for field, value in changes:
            old = self.calibration[field]
            self.calibration[field] = value
            self.save()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                inspect_bundle(self.root)
            self.calibration[field] = old

    def test_threshold_change_keeps_vectors_but_changes_calibration_identity(self):
        first = inspect_bundle(self.root)
        self.calibration['threshold_cosine'] = .7
        self.calibration['original']['selected']['threshold'] = .7
        self.save()
        second = inspect_bundle(self.root)
        self.assertEqual(first.inference_fingerprint, second.inference_fingerprint)
        self.assertNotEqual(first.calibration_sha256, second.calibration_sha256)
        self.config['members'][0]['config']['seed'] = 2030
        self.save()
        self.assertNotEqual(second.inference_fingerprint, inspect_bundle(self.root).inference_fingerprint)

    def test_dependency_version_mismatch_is_fatal(self):
        self.manifest['inference']['dependency_versions']['numpy'] = '0.0.0'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'numpy'):
            verify_dependencies(inspect_bundle(self.root))

    def test_torch_base_version_allows_build_but_fingerprint_records_it(self):
        self.manifest['inference']['dependency_versions']['torch'] = '2.8.0'
        self.save_manifest()
        bundle = inspect_bundle(self.root)
        actual = importlib.metadata.version
        def versions(build):
            return lambda name: build if name == 'torch' else actual(name)
        with patch('vehicle.server.model_bundle.importlib.metadata.version', side_effect=versions('2.8.0+cpu')):
            verify_dependencies(bundle)
            cpu = runtime_fingerprint(bundle, 'cpu')
        with patch('vehicle.server.model_bundle.importlib.metadata.version', side_effect=versions('2.8.0+cu126')):
            verify_dependencies(bundle)
            cuda = runtime_fingerprint(bundle, 'cuda')
        self.assertNotEqual(cpu, cuda)
        self.manifest['inference']['dependency_versions']['torch'] = '2.8.0+cu126'
        self.save_manifest()
        with patch('vehicle.server.model_bundle.importlib.metadata.version', side_effect=versions('2.8.0+cpu')), self.assertRaises(ValueError):
            verify_dependencies(inspect_bundle(self.root))

    def test_extra_weights_and_dimension_outside_service_contract_rejected(self):
        (self.root / 'teacher.PTH').write_bytes(b'TRAINING ONLY asset must not travel to runtime')
        with self.assertRaisesRegex(ValueError, 'Лишний файл весов'):
            inspect_bundle(self.root)
        (self.root / 'teacher.PTH').unlink()
        self.manifest['model']['dimension'] = 16385
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'dimension'):
            inspect_bundle(self.root)

    def test_multiple_env_sources_and_missing_bundle_stay_closed(self):
        for env in ({'VEHICLE_MODEL_BUNDLE': str(self.root), 'VEHICLE_MODEL': '/different/model.pt'},
                    {'VEHICLE_MODEL_BUNDLE': str(self.root / 'missing')}):
            with patch.dict(os.environ, env, clear=True), patch('vehicle.server.provider.LOG'):
                provider = ProductionProvider()
            self.assertFalse(provider.available)
            self.assertIsNone(provider.predictor)
            self.assertIsNone(provider.model)

    def test_empty_operator_directory_is_pending_not_corrupt_model(self):
        empty = self.root / 'awaiting-selected-model'
        empty.mkdir()
        with patch.dict(os.environ, {'VEHICLE_MODEL_BUNDLE': str(empty)}, clear=True), patch('vehicle.server.provider.LOG') as log:
            provider = ProductionProvider()
        self.assertFalse(provider.available)
        self.assertEqual(provider.reason, 'Пакет выбранной модели ещё не подключён (bundle.json отсутствует).')
        self.assertIsNone(provider.predictor)
        log.exception.assert_not_called()

if __name__ == '__main__':
    unittest.main()
