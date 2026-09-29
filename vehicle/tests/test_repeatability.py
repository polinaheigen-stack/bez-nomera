"""Process orchestration tests: test-only providers never enter shipped runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vehicle.server.repeatability import run_repeatability
from vehicle.tests.test_batch import dataset


class RepeatabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images, self.query, self.gallery = dataset(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_two_fresh_processes_use_same_batch_and_existing_parity(self):
        real_popen = subprocess.Popen
        commands = []
        # Replace only the provider in these test child interpreters. The actual
        # production command, batching/export/parity remain under examination.
        bootstrap = ('from vehicle.tests.test_batch import FixtureProvider; '
                     'import vehicle.server.batch as batch; '
                     'batch.ProductionProvider = FixtureProvider; batch.main()')
        def test_child(command, **kwargs):
            commands.append(command)
            return real_popen([command[0], '-c', bootstrap, *command[3:]], **kwargs)
        with patch('vehicle.server.repeatability.subprocess.Popen', side_effect=test_child):
            report = run_repeatability(self.images, self.query, self.gallery, self.root / 'proof', batch_size=4)
        self.assertTrue(report['passed'], report.get('error'))
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(command[1:3] == ['-m', 'vehicle.server.batch'] for command in commands))
        self.assertTrue(all(command[-2:] == ['--batch-size', '4'] for command in commands))
        self.assertNotEqual(report['runs'][0]['pid'], report['runs'][1]['pid'])
        self.assertTrue(all(run['wall_seconds'] > 0 for run in report['runs']))
        self.assertTrue(all(run['provenance_sha256'] for run in report['runs']))
        self.assertFalse(report['quality_claim'])
        self.assertTrue(report['parity']['passed'])
        self.assertEqual(len(report['inputs']['gallery']), 10)

    def test_actual_cli_without_weights_saves_failed_evidence_and_no_exports(self):
        output = self.root / 'missing-weights'
        env = {k: v for k, v in os.environ.items() if not k.startswith('VEHICLE_')}
        result = subprocess.run([sys.executable, '-m', 'vehicle.server.repeatability',
                                 '--images', str(self.images), '--query-csv', str(self.query),
                                 '--gallery-csv', str(self.gallery), '--output', str(output)],
                                env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        evidence = json.loads((output / 'repeatability.json').read_text(encoding='utf-8'))
        self.assertFalse(evidence['passed'])
        self.assertEqual(evidence['status'], 'failed')
        self.assertEqual(evidence['runs'][0]['status'], 'failed')
        self.assertNotEqual(evidence['runs'][0]['returncode'], 0)
        self.assertTrue((output / 'run-1.stderr.log').read_bytes())
        self.assertFalse(list(output.rglob('submission.csv')))
        self.assertFalse((output / 'run-2').exists())
        self.assertFalse((output / 'parity.json').exists())

    def test_mutated_input_between_processes_fails_closed(self):
        real_popen = subprocess.Popen
        bootstrap = ('from vehicle.tests.test_batch import FixtureProvider; '
                     'import vehicle.server.batch as batch; '
                     'batch.ProductionProvider = FixtureProvider; batch.main()')
        calls = []
        def test_child(command, **kwargs):
            calls.append(command)
            if len(calls) == 2:
                (self.images / 'query.png').write_bytes((self.images / 'query.png').read_bytes() + b'test-change')
            return real_popen([command[0], '-c', bootstrap, *command[3:]], **kwargs)
        with patch('vehicle.server.repeatability.subprocess.Popen', side_effect=test_child):
            report = run_repeatability(self.images, self.query, self.gallery, self.root / 'changed')
        self.assertFalse(report['passed'])
        self.assertIn('identity differs', report['error'])
        self.assertEqual(report['runs'][0]['status'], 'completed')
        self.assertEqual(report['runs'][1]['status'], 'failed')
        self.assertFalse((self.root / 'changed' / 'parity.json').exists())

    def test_timeout_preserves_logs_and_kills_child(self):
        real_popen = subprocess.Popen
        children = []
        def sleeping_child(command, **kwargs):
            child = real_popen([command[0], '-c', 'import time; time.sleep(60)'], **kwargs)
            children.append(child)
            return child
        with patch('vehicle.server.repeatability.subprocess.Popen', side_effect=sleeping_child):
            report = run_repeatability(self.images, self.query, self.gallery, self.root / 'timeout', timeout_seconds=.05)
        self.assertFalse(report['passed'])
        self.assertEqual(report['runs'][0]['status'], 'failed')
        self.assertIsNotNone(children[0].poll())
        self.assertTrue((self.root / 'timeout' / 'repeatability.json').exists())

    def test_existing_output_and_bad_batch_are_rejected(self):
        output = self.root / 'already-used'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'new output'):
            run_repeatability(self.images, self.query, self.gallery, output)
        self.assertEqual(marker.read_text(), 'keep')
        with self.assertRaisesRegex(ValueError, 'Batch size'):
            run_repeatability(self.images, self.query, self.gallery, self.root / 'bad', batch_size=0)
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
