"""Test-only E22 metadata. The weight file is deliberately not a checkpoint."""
import importlib.metadata
import json
from pathlib import Path
import tempfile

from vehicle.server.e22_contract import PROTOCOL, SOURCE_COMMIT, source_hash
from vehicle.server.model_bundle import sha256
from vehicle.tests.test_e22 import fixture_config


class E22MetadataFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'model.pt').write_bytes(b'METADATA TEST ONLY. Not a torch checkpoint.')
        self.config = fixture_config()
        self.calibration = {
            'model_sha256': sha256(self.root / 'model.pt'), 'code_hash': source_hash(),
            'threshold_cosine': .6, 'calibration_protocol': PROTOCOL,
            'original': {'score_mode': 'cosine', 'query_expansion': {'k': 0},
                         'selected': {'threshold': .6}},
        }
        self.manifest = {
            'schema_version': 1, 'status': 'ready', 'adapter': 'e22_ensemble',
            'model': {'name': 'UNIT TEST fixture — not quality evidence', 'version': 'test', 'dimension': 512},
            'artifacts': {},
            'provenance': {'inference_code_sha256': source_hash(), 'handoff_metadata_sha256': 'b' * 64,
                           'source_commit': SOURCE_COMMIT, 'source_role': 'supplied_reproduction_source'},
            'inference': {'adapter_revision': 1, 'preprocessing': 'native_checkpoint', 'score': 'cosine_similarity',
                          'dependency_versions': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'Pillow')}},
        }
        self.save()

    def save(self):
        (self.root / 'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        self.calibration['inference_config_sha256'] = sha256(self.root / 'config.json')
        (self.root / 'calibration.json').write_text(json.dumps(self.calibration), encoding='utf-8')
        self.manifest['artifacts'] = {k: {'file': f, 'sha256': sha256(self.root / f)} for k, f in
                                     (('weights', 'model.pt'), ('config', 'config.json'), ('calibration', 'calibration.json'))}
        self.save_manifest()

    def save_manifest(self):
        (self.root / 'bundle.json').write_text(json.dumps(self.manifest), encoding='utf-8')
