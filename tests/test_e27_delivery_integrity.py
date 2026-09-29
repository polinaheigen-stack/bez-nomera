"""Hash integrity and model import safety; test files stay in temporary directories."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load('e27_delivery_verifier', 'verify_context.py')
importer = load('e27_delivery_importer', 'scripts/import_model.py')


def digest(value):
    return hashlib.sha256(value).hexdigest()


class SourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weight = b'test checkpoint bytes'
        (self.root / 'models/e27').mkdir(parents=True)
        (self.root / 'models/e27/model.pth').write_bytes(self.weight)
        (self.root / 'README.md').write_bytes(b'readme')
        self.entries = [{'path': p.relative_to(self.root).as_posix(), 'bytes': p.stat().st_size,
                         'sha256': verifier.sha256(p)} for p in sorted(self.root.rglob('*')) if p.is_file()]
        self.write()
        self.addCleanup(patch.stopall)
        patch.object(verifier, 'WEIGHT_BYTES', len(self.weight)).start()
        patch.object(verifier, 'WEIGHT_SHA256', digest(self.weight)).start()

    def write(self):
        (self.root / 'SOURCE_SHA256.json').write_text(json.dumps({'release': 'E27', 'files': self.entries}))

    def test_exact_inventory_and_ignored_runtime_outputs(self):
        (self.root / 'results').mkdir()
        (self.root / 'results/run.json').write_text('{}')
        report = verifier.verify(self.root)
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['files'], 2)

    def test_changed_input_rejected(self):
        (self.root / 'README.md').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Changed build input'):
            verifier.verify(self.root)

    def test_old_checkpoint_and_unknown_source_are_rejected(self):
        old = self.root / 'old.pth'
        old.write_bytes(b'old')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            verifier.verify(self.root)

    def test_circular_duplicate_and_traversing_entries_rejected(self):
        original = list(self.entries)
        for name in ('SOURCE_SHA256.json', '../outside', 'C:/outside', 'nested/../file'):
            self.entries = original + [{'path': name, 'bytes': 0, 'sha256': 'a' * 64}]
            self.write()
            with self.assertRaises(ValueError):
                verifier.verify(self.root)
        self.entries = original + [original[0]]
        self.write()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            verifier.verify(self.root)

    def test_missing_weight_rejected(self):
        (self.root / 'models/e27/model.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            verifier.verify(self.root)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.source = base / 'repo', base / 'delivery'
        self.destination = self.root / 'models/e27'
        self.destination.mkdir(parents=True)
        self.source.mkdir()
        weight = b'temporary test checkpoint'
        config = b'{"test_only":true}'
        self.sha = digest(weight)
        bundle = {'adapter': 'e27_compact', 'model': {'version': 'test-only'},
                  'artifacts': {'weights': {'file': 'model.pth', 'sha256': self.sha},
                                'config': {'file': 'config.json', 'sha256': digest(config)}}}
        for directory in (self.destination, self.source):
            (directory / 'bundle.json').write_text(json.dumps(bundle))
            (directory / 'config.json').write_bytes(config)
        (self.source / 'model.pth').write_bytes(weight)
        self.addCleanup(patch.stopall)
        patch.object(importer, 'WEIGHT_BYTES', len(weight)).start()
        patch.object(importer, 'WEIGHT_SHA256', self.sha).start()

    def test_import_and_second_import_are_safe(self):
        first = importer.import_model(self.source, root=self.root)
        self.assertEqual(first['copied'], ['models/e27/model.pth'])
        self.assertEqual(importer.sha256(self.destination / 'model.pth'), self.sha)
        second = importer.import_model(self.source, root=self.root)
        self.assertEqual(second['copied'], [])
        self.assertEqual(list(self.destination.glob('.import-*')), [])

    def test_check_only_does_not_copy(self):
        self.assertEqual(importer.import_model(self.source, root=self.root, check_only=True)['status'], 'verified')
        self.assertFalse((self.destination / 'model.pth').exists())

    def test_mismatching_metadata_or_weights_cannot_be_imported(self):
        (self.source / 'config.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed incoming'):
            importer.import_model(self.source, root=self.root)
        self.assertFalse((self.destination / 'model.pth').exists())

    def test_existing_different_checkpoint_is_preserved(self):
        (self.destination / 'model.pth').write_bytes(b'do not overwrite')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            importer.import_model(self.source, root=self.root)
        self.assertEqual((self.destination / 'model.pth').read_bytes(), b'do not overwrite')


if __name__ == '__main__':
    unittest.main()
