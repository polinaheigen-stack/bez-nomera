"""Orchestration tests prove labels enter scoring only after immutable predictions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('e27_export', ROOT / 'scripts/export_results.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
VERSION = 'e27-dinov3-smallplus256-fast32-gallery4nn-p2'


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.query, self.gallery, self.gt = [self.root / name for name in ('query.csv', 'gallery.csv', 'gt.csv')]
        for path in (self.query, self.gallery, self.gt):
            path.write_text('test only')
        self.output = self.root / 'new-results'
        self.events = []
        self.provider = types.SimpleNamespace(available=True, reason=None, model={'version': VERSION},
                                             close=lambda: self.events.append('close'))
        def infer(provider, images, query, gallery, output, *, batch_size):
            self.events.append('inference')
            output.mkdir()
            return {'model': provider.model, 'input_order': {'query': ['q'], 'gallery': ['g']}}
        def evaluate(*args):
            self.events.append('evaluation')
            self.assertTrue((self.output).is_dir())
            return {'official': {'test_only': True}}
        self.factory = Mock(return_value=self.provider)
        self.batch = Mock(side_effect=infer)
        self.evaluate = Mock(side_effect=evaluate)

    def execute(self, **kwargs):
        with patch.object(exporter, 'components', return_value=(self.factory, self.batch, self.evaluate)):
            return exporter.export_results(self.images, self.query, self.gallery, self.output, **kwargs)

    def test_labels_are_not_inference_arguments_and_evaluation_is_last(self):
        report = self.execute(ground_truth=self.gt)
        self.assertEqual(self.events, ['inference', 'close', 'evaluation'])
        self.assertNotIn(self.gt, self.batch.call_args.args)
        self.assertEqual(self.batch.call_args.kwargs, {'batch_size': 32})
        self.assertTrue(report['quality_measured'])
        self.assertFalse(report['official_gpu_speed_measured'])

    def test_no_labels_means_no_quality_claim(self):
        report = self.execute(device='cpu')
        self.evaluate.assert_not_called()
        self.assertFalse(report['quality_measured'])
        self.factory.assert_called_once_with(device='cpu')
        self.assertEqual(json.loads((self.output / 'EXPORT-REPORT.json').read_text())['status'], 'completed')

    def test_single_query_batch_remains_available(self):
        self.execute(batch_size=1)
        self.assertEqual(self.batch.call_args.kwargs['batch_size'], 1)

    def test_unsupported_batch_size_never_loads_model(self):
        for size in (0, 33, True):
            with self.assertRaises(ValueError):
                self.execute(batch_size=size)
        self.factory.assert_not_called()

    def test_unavailable_gpu_is_not_replaced_by_cpu(self):
        self.provider.available = False
        self.provider.reason = 'CUDA unavailable'
        with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            self.execute()
        self.factory.assert_called_once_with(device='cuda')
        self.batch.assert_not_called()
        self.evaluate.assert_not_called()
        self.assertEqual(self.events, ['close'])

    def test_inference_failure_closes_model_and_never_scores(self):
        self.batch.side_effect = ValueError('bad input')
        with self.assertRaisesRegex(ValueError, 'bad input'):
            self.execute(ground_truth=self.gt)
        self.assertEqual(self.events, ['close'])
        self.evaluate.assert_not_called()

    def test_previous_results_are_preserved(self):
        self.output.mkdir()
        (self.output / 'report.json').write_text('previous')
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            self.execute()
        self.factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
