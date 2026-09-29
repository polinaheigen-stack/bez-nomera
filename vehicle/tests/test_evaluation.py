"""Regression fixtures for the actual organizer evaluator, including its edge cases."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vehicle.server.evaluation import evaluate_export, OFFICIAL_PATH, OFFICIAL_SHA256
from vehicle.server.provider import sha256
from vehicle.server.validation import write_outputs


class OfficialEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        qids, gids = ['q', 'unknown'], [f'g{i}' for i in range(11)]
        qv = np.array([[1, 0], [-1, 0]], dtype=np.float32)
        x = np.linspace(1., .01, 11, dtype=np.float32)
        gv = np.stack([x, np.sqrt(1 - x * x)], axis=1)
        output = self.root / 'out'
        write_outputs(output, qids, gids, qv, gv, .99)
        for name, ids in (('query', qids), ('gallery', gids)):
            with (self.root / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
                for image_id in ids:
                    writer.writerow([image_id, 0, 0, 20, 20])
        with (self.root / 'gt.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'vehicle_id', 'camera_id', 'split'])
            writer.writerow(['q', 'A', '0', 'query'])
            writer.writerow(['unknown', 'Z', '0', 'query'])
            for i, gid in enumerate(gids):
                writer.writerow([gid, 'A' if i in (0, 10) else f'N{i}', '0' if i == 0 else '1', 'gallery'])
        return output, self.root / 'query.csv', self.root / 'gallery.csv', self.root / 'gt.csv'

    def test_original_source_hash_is_unchanged(self):
        self.assertEqual(sha256(OFFICIAL_PATH), OFFICIAL_SHA256)

    @unittest.skipUnless(importlib.util.find_spec('pandas'), 'Official evaluator requires pandas')
    def test_actual_csv_no_junk_refill_and_same_camera_candidate_semantics(self):
        args = self.fixture()
        report = evaluate_export(*args)
        official = report['official']
        # g0 is junk; g10 (the sole cross-camera positive) was ranked 11th.
        # The exact top-10 export cannot refill its missing tenth slot from embeddings.
        self.assertEqual(official['ranking']['mAP@10'], 0.)
        self.assertEqual(official['ranking']['Rank-1'], 0.)
        self.assertEqual(official['ranking']['n_openset_excluded'], 1)
        self.assertAlmostEqual(official['full_ranking']['mAP_full'], .1)
        # The unchanged official candidate branch accepts same-camera g0 as TP
        # because a separate cross-camera positive exists in the full gallery.
        self.assertEqual(official['candidates']['TP'], 1)
        self.assertEqual(official['candidates']['TN'], 1)
        self.assertEqual(official['candidates']['F1'], 1.)
        self.assertEqual(official['candidates']['TNR'], 1.)
        self.assertTrue(report['measured'])
        self.assertFalse(any(args[0].glob('*.zip')))

    def test_modified_evaluator_is_rejected_before_execution(self):
        args = self.fixture()
        altered = self.root / 'altered.py'
        altered.write_bytes(OFFICIAL_PATH.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            evaluate_export(*args, evaluator=altered)

    def test_missing_label_is_rejected_instead_of_scoring_a_different_set(self):
        args = self.fixture()
        gt = args[-1]
        lines = gt.read_text().splitlines()
        gt.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            evaluate_export(*args)

    @unittest.skipUnless(importlib.util.find_spec('pandas'), 'Official evaluator requires pandas')
    def test_bad_provenance_cannot_publish_or_overwrite_measured_results(self):
        args = self.fixture()
        output = args[0]
        previous = {'measured': False, 'note': 'existing report'}
        (output / 'metrics.json').write_text(json.dumps(previous), encoding='utf-8')
        (output / 'evaluation-report.json').write_text(json.dumps(previous), encoding='utf-8')
        (output / 'provenance.json').write_text(json.dumps({'status': 'completed', 'files_sha256': {}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed since inference'):
            evaluate_export(*args)
        self.assertEqual(json.loads((output / 'metrics.json').read_text()), previous)
        self.assertEqual(json.loads((output / 'evaluation-report.json').read_text()), previous)

    def test_changed_evaluation_inputs_cannot_publish_measured_report(self):
        args = self.fixture()
        output, query, gallery, gt = args
        evaluator = self.root / 'official-copy.py'
        evaluator.write_bytes(OFFICIAL_PATH.read_bytes())
        previous = {'measured': False, 'note': 'unchanged prior report'}
        (output / 'metrics.json').write_text(json.dumps(previous), encoding='utf-8')
        targets = [query, gallery, gt, output / 'submission.csv', output / 'candidates.csv', output / 'embeddings.npy', evaluator]
        for target in targets:
            with self.subTest(file=target.name):
                before = target.read_bytes()
                def fake_evaluator(command, **kwargs):
                    Path(command[command.index('--json') + 1]).write_text('{}', encoding='utf-8')
                    target.write_bytes(before + b'\n')
                    return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
                try:
                    with patch('vehicle.server.evaluation.subprocess.run', side_effect=fake_evaluator):
                        with self.assertRaisesRegex(ValueError, 'inputs changed during evaluation'):
                            evaluate_export(*args, evaluator=evaluator)
                    self.assertEqual(json.loads((output / 'metrics.json').read_text()), previous)
                    self.assertFalse((output / 'evaluation-report.json').exists())
                finally:
                    target.write_bytes(before)


if __name__ == '__main__':
    unittest.main()
