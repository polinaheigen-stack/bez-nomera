"""Portable launch plans and verified image source metadata; no Docker execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launcher = load('portable_launcher', 'scripts/start_web.py')
entrypoint = load('verified_entrypoint', 'vehicle/server/runtime_entrypoint.py')


class MetadataTests(unittest.TestCase):
    def test_plan_displays_portable_python_and_executes_current_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models/e27').mkdir(parents=True)
            (root / 'models/e27/bundle.json').write_text(json.dumps({'adapter': 'e27_compact'}))
            (root / 'SOURCE_SHA256.json').write_bytes(b'{}')
            with patch.object(launcher.sys, 'executable', '/temporary/python'), patch.object(launcher.subprocess, 'run') as run:
                plan = launcher.start(root)
                self.assertEqual(plan['commands'][0], ['python', 'verify_context.py'])
                self.assertEqual(run.call_args_list[0].args[0][0], '/temporary/python')

    def test_build_marker_is_resolved_to_verified_digest(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'VEHICLE_SOURCE_MANIFEST_SHA256': 'computed-in-image'}):
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_bytes(b'{"release":"E27"}\n')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            proof = root / 'proof.json'
            proof.write_text(json.dumps({'status': 'verified', 'source_manifest_sha256': digest}))
            entrypoint.bind_source_manifest(proof, manifest)
            self.assertEqual(os.environ['VEHICLE_SOURCE_MANIFEST_SHA256'], digest)

    def test_modified_image_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'VEHICLE_SOURCE_MANIFEST_SHA256': 'computed-in-image'}):
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_bytes(b'changed')
            proof = root / 'proof.json'
            proof.write_text(json.dumps({'status': 'verified', 'source_manifest_sha256': 'a' * 64}))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                entrypoint.bind_source_manifest(proof, manifest)

    def test_compose_does_not_inherit_stale_source_digest(self):
        compose = (ROOT / 'docker-compose.yml').read_text(encoding='utf-8')
        self.assertIn('SOURCE_MANIFEST_SHA256: computed-in-image', compose)
        self.assertNotIn('${SOURCE_MANIFEST_SHA256', compose)


if __name__ == '__main__':
    unittest.main()
