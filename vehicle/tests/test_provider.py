"""Production E22 gates. Test adapters are never packaged as working models."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vehicle.server.provider import ProductionProvider
from vehicle.server.batch import find_images
from vehicle.tests.bundle_fixture import E22MetadataFixture


class ProviderTests(E22MetadataFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.env = {'VEHICLE_MODEL_BUNDLE': str(self.root), 'VEHICLE_DEVICE': 'cpu'}

    def construct(self, *, environment=None, loader_error=None, model_device='cpu', dimension=512, device=None):
        adapter = MagicMock()
        adapter.device, adapter.dimension = model_device, dimension
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1
        adapter.embed.return_value = vector
        adapter.embed_batch.side_effect = lambda images, bboxes: np.tile(vector, (len(images), 1))
        with patch.dict(os.environ, self.env if environment is None else environment, clear=True), \
                patch('vehicle.server.model_adapters.load_adapter', return_value=adapter, side_effect=loader_error) as loader, \
                patch('vehicle.server.provider.LOG'):
            provider = ProductionProvider(device=device)
        return provider, loader, adapter

    def assert_closed(self, provider):
        self.assertFalse(provider.available)
        self.assertIsNone(provider.model)
        self.assertIsNone(provider.predictor)
        self.assertIsNone(provider.threshold)
        self.assertTrue(provider.reason)
        with self.assertRaises(RuntimeError):
            provider.embed(None, (0, 0, 1, 1), image_id='never-inferred')

    def test_missing_or_obsolete_configuration_never_loads(self):
        environments = [{}, {'VEHICLE_MODEL_BUNDLE': str(self.root / 'missing')}]
        for key in ('VEHICLE_MODEL', 'VEHICLE_MODEL_SHA256', 'VEHICLE_CALIBRATION', 'VEHICLE_CALIBRATION_SHA256'):
            environments.extend([{key: 'obsolete'}, {**self.env, key: 'obsolete'}])
        for env in environments:
            with self.subTest(env=env):
                provider, loader, _ = self.construct(environment=env)
                self.assert_closed(provider)
                loader.assert_not_called()

    def test_artifact_tampering_never_reaches_loader(self):
        for filename in ('model.pt', 'config.json', 'calibration.json'):
            path = self.root / filename
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            provider, loader, _ = self.construct()
            self.assert_closed(provider)
            loader.assert_not_called()
            path.write_bytes(original)

    def test_wrong_calibration_identity_and_protocol_never_load(self):
        for key, value in (('model_sha256', 'f' * 64), ('code_hash', 'e' * 64),
                           ('calibration_protocol', 'query_top1_micro_f1_v1')):
            old = self.calibration[key]
            self.calibration[key] = value
            self.save()
            provider, loader, _ = self.construct()
            self.assert_closed(provider)
            loader.assert_not_called()
            self.calibration[key] = old

    def test_nonfinite_invalid_and_boolean_thresholds_never_load(self):
        for threshold in (None, True, '0.5', float('nan'), float('inf'), -1.001, 1.001):
            with self.subTest(threshold=threshold):
                self.calibration['threshold_cosine'] = threshold
                self.save()
                provider, loader, _ = self.construct()
                self.assert_closed(provider)
                loader.assert_not_called()

    def test_invalid_metadata_and_untrained_member_stay_closed(self):
        (self.root / 'bundle.json').write_text('{invalid', encoding='utf-8')
        provider, loader, _ = self.construct()
        self.assert_closed(provider)
        loader.assert_not_called()
        self.config['members'][0]['config']['selected_epoch'] = 0
        self.save()
        provider, loader, _ = self.construct()
        self.assert_closed(provider)
        loader.assert_not_called()
        self.config['members'][0]['config']['selected_epoch'] = 1
        self.config['members'][0]['config']['debug_only'] = True
        self.save()
        provider, loader, _ = self.construct()
        self.assert_closed(provider)
        loader.assert_not_called()

    def test_removed_legacy_bundle_rejected_before_loader(self):
        self.manifest['adapter'] = 'legacy_baseline'
        self.save_manifest()
        provider, loader, _ = self.construct()
        self.assert_closed(provider)
        loader.assert_not_called()

    def test_predictor_rejection_stays_closed(self):
        for error in (ValueError('Unsupported model artifact'), RuntimeError('CUDA unavailable'), FileNotFoundError('model.pt')):
            provider, _, _ = self.construct(loader_error=error)
            self.assert_closed(provider)

    def test_wrong_device_or_dimension_never_publishes_partial_model(self):
        for kwargs in ({'model_device': 'cuda:0'}, {'dimension': 2048}):
            provider, _, _ = self.construct(**kwargs)
            self.assert_closed(provider)

    def test_full_image_and_bbox_forwarded_without_id_features(self):
        provider, loader, model = self.construct()
        self.assertTrue(provider.available, provider.reason)
        self.assertIsNone(provider.reason)
        self.assertAlmostEqual(provider.threshold, .8)
        self.assertEqual(provider.model['sha256'], self.manifest['artifacts']['weights']['sha256'])
        self.assertEqual(provider.model['dimension'], 512)
        self.assertEqual(provider.calibration_sha256, self.manifest['artifacts']['calibration']['sha256'])
        self.assertEqual(loader.call_args.args[1], 'cpu')
        full_image, bbox = object(), (123, 45, 300, 220)
        one = provider.embed(full_image, bbox, image_id='first')
        model.embed.assert_called_once_with(full_image, bbox)
        many = provider.embed_batch([full_image, full_image], [bbox, bbox], image_ids=['different', 'arbitrary'])
        model.embed_batch.assert_called_once_with([full_image, full_image], [bbox, bbox])
        np.testing.assert_array_equal(one, many[0])
        np.testing.assert_array_equal(many[0], many[1])
        self.assertEqual(provider.embed_batch([], [], image_ids=[]).shape, (0, 512))
        with self.assertRaises(ValueError):
            provider.embed_batch([full_image], [], image_ids=['bad'])

    def test_invalid_vectors_rejected_for_single_and_batch(self):
        provider, _, adapter = self.construct()
        for values in (np.zeros((1, 512)), np.full((1, 512), np.nan), np.ones((1, 511))):
            adapter.embed.return_value = values[0]
            with self.assertRaises(ValueError):
                provider.embed(object(), (0, 0, 1, 1), image_id='single')
            adapter.embed_batch.side_effect = None
            adapter.embed_batch.return_value = values
            with self.assertRaises(ValueError):
                provider.embed_batch([object()], [(0, 0, 1, 1)], image_ids=['batch'])

    def test_explicit_cpu_choice_overrides_batch_environment(self):
        provider, loader, _ = self.construct(environment={**self.env, 'VEHICLE_DEVICE': 'cuda'}, device='cpu')
        self.assertTrue(provider.available, provider.reason)
        self.assertEqual(provider.model['device'], 'cpu')
        self.assertEqual(loader.call_args.args[1], 'cpu')


class BatchPathTests(unittest.TestCase):
    def test_actual_file_entries_avoid_windows_suffix_duplicates(self):
        scratch = Path(__file__).resolve().parents[1] / 'var' / 'tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            (root / 'car.jpg').write_bytes(b'Filename enumeration fixture, not an image or model.')
            (root / 'other.PNG').write_bytes(b'Filename enumeration fixture, not an image or model.')
            result = find_images(root, [{'image_id': 'car'}, {'image_id': 'other'}])
            self.assertEqual(result['car'].name, 'car.jpg')
            self.assertEqual(result['other'].name, 'other.PNG')
            (root / 'car.png').write_bytes(b'Deliberate duplicate image ID.')
            with self.assertRaises(ValueError):
                find_images(root, [{'image_id': 'car'}])
            with self.assertRaises(ValueError):
                find_images(root, [{'image_id': 'absent'}])


if __name__ == '__main__':
    unittest.main()
