"""Delivery launcher tests; Docker and model inference are never started."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('delivery_start_web', ROOT / 'scripts/start_web.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class DeliveryLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'models/e27').mkdir(parents=True)
        (self.root / 'models/e27/bundle.json').write_text(json.dumps({'adapter': 'e27_compact'}))
        (self.root / 'SOURCE_SHA256.json').write_text('{"test_only": true}')

    def test_gpu_is_default_and_fresh_restart_is_ordered_after_verification(self):
        original = {'SOURCE_MANIFEST_SHA256': 'old', 'WEB_PORT': '1'}
        commands, env = launcher.launch_plan(self.root, original)
        self.assertEqual(original['SOURCE_MANIFEST_SHA256'], 'old')
        self.assertEqual(len(env['SOURCE_MANIFEST_SHA256']), 64)
        self.assertEqual(env['WEB_PORT'], '8027')
        self.assertEqual(commands[0][1:], ['verify_context.py'])
        self.assertEqual(commands[1][-2:], ['build', 'web'])
        self.assertEqual(commands[2][-3:], ['stop', 'web', 'web-cpu'])
        self.assertEqual(commands[3][-5:], ['up', '-d', '--no-build', '--force-recreate', 'web'])

    def test_cpu_selects_cpu_service_without_gpu_fallback(self):
        commands, env = launcher.launch_plan(self.root, device='cpu', port=9001)
        self.assertEqual(commands[-1][-1], 'web-cpu')
        self.assertEqual(commands[1][-1], 'web-cpu')
        self.assertEqual(env['WEB_PORT'], '9001')

    def test_dry_run_executes_nothing(self):
        with patch.object(launcher.subprocess, 'run') as run:
            report = launcher.start(self.root, dry_run=True)
        run.assert_not_called()
        self.assertEqual(report['status'], 'planned_only')
        self.assertFalse(report['verification_executed'])

    def test_bad_source_prevents_build_and_stopping_existing_web(self):
        with patch.object(launcher.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['verify_context.py'])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.start(self.root)
        self.assertEqual(run.call_count, 1)

    def test_failed_build_does_not_stop_running_web(self):
        with patch.object(launcher.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(1, ['docker'])]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.start(self.root)
        self.assertEqual(run.call_count, 2)

    def test_invalid_device_or_port_and_missing_manifest_are_rejected(self):
        for kwargs in ({'device': 'auto'}, {'port': 80}, {'port': 65536}):
            with self.assertRaises(ValueError):
                launcher.launch_plan(self.root, **kwargs)
        (self.root / 'SOURCE_SHA256.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            launcher.launch_plan(self.root)


if __name__ == '__main__':
    unittest.main()
