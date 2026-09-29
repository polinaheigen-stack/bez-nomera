"""E22 rejection tests. Metadata fixtures never become inference packages."""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest

import torch

from vehicle.server.e22_adapter import load_encoder_state
from vehicle.server.e22_contract import AGGREGATION, MEMBERS, SCHEMA, SOURCE_COMMIT, PROTOCOL, source_hash, checkpoint_config
from vehicle.server.model_bundle import inspect_bundle, sha256


def fixture_config():
    members = []
    for name, backbone, architecture in MEMBERS:
        members.append({'name': name, 'weight': 1.0, 'source_sha256': 'a'*64,
                        'config': {'backbone': backbone, 'architecture': architecture, 'embedding_dim': 512,
                                   'pooling': 'avg', 'bnneck_mode': 'shared', 'metric_head': 'linear',
                                   'backbone_stride': 32, 'resize_mode': 'stretch', 'crop_padding': 0,
                                   'crop_padding_ratio': 0.0, 'dino_part_stripes': 0, 'selected_epoch': 1,
                                   'image_size': [256, 128], 'dino_variant': 'b14', 'dino_unfreeze_blocks': 2}})
    return {'schema': SCHEMA, 'embedding_dim': 512, 'aggregation': AGGREGATION, 'flip_tta': True, 'members': members}


class E22ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'model.pth').write_bytes(b'METADATA FIXTURE ONLY; NOT TRAINED WEIGHTS')
        self.config = fixture_config()
        self.threshold = .487464964389801
        self.calibration = {'calibration_protocol': PROTOCOL, 'model_sha256': sha256(self.root/'model.pth'),
                            'code_hash': source_hash(), 'threshold_cosine': self.threshold,
                            'original': {'score_mode': 'cosine', 'query_expansion': {'k': 0},
                                         'selected': {'threshold': self.threshold, 'f1': .032}}}
        self.manifest = {'schema_version': 1, 'status': 'ready', 'adapter': 'e22_ensemble',
                         'model': {'name': 'TEST fixture', 'version': 'test', 'dimension': 512},
                         'artifacts': {}, 'provenance': {'source_commit': SOURCE_COMMIT,
                             'source_role': 'supplied_reproduction_source', 'inference_code_sha256': source_hash(),
                             'handoff_metadata_sha256': 'b'*64},
                         'inference': {'adapter_revision': 1, 'preprocessing': 'native_checkpoint',
                                       'score': 'cosine_similarity', 'dependency_versions':
                                       {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'Pillow')}}}
        self.save()

    def save(self):
        (self.root/'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        self.calibration['inference_config_sha256'] = sha256(self.root/'config.json')
        (self.root/'calibration.json').write_text(json.dumps(self.calibration), encoding='utf-8')
        self.manifest['artifacts'] = {k: {'file': f, 'sha256': sha256(self.root/f)} for k, f in
                                     (('weights', 'model.pth'), ('config', 'config.json'), ('calibration', 'calibration.json'))}
        (self.root/'bundle.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_threshold_is_imported_without_overwriting_diagnostic_statistics(self):
        bundle = inspect_bundle(self.root)
        self.assertEqual(bundle.threshold_cosine, self.threshold)
        self.assertAlmostEqual((bundle.threshold_cosine+1)/2, .7437324821949005)
        self.assertEqual(bundle.calibration['original']['selected']['f1'], .032)

    def test_wrong_members_tta_geometry_and_untrained_state_rejected(self):
        original = deepcopy(self.config)
        changes = [lambda c: c.update(flip_tta=False), lambda c: c['members'].reverse(),
                   lambda c: c['members'][0].update(weight=2),
                   lambda c: c['members'][0]['config'].update(image_size=[128,256]),
                   lambda c: c['members'][1]['config'].update(selected_epoch=0),
                   lambda c: c['members'][1]['config'].update(dino_part_stripes=3)]
        for change in changes:
            self.config = deepcopy(original)
            change(self.config)
            self.save()
            with self.subTest(config=self.config), self.assertRaises(ValueError):
                inspect_bundle(self.root)

    def test_incorrect_scale_or_model_or_code_is_rejected(self):
        for key, value in [('threshold_cosine', (self.threshold+1)/2), ('model_sha256', 'd'*64),
                           ('code_hash', 'd'*64), ('calibration_protocol', 'official_f1')]:
            old = self.calibration[key]
            self.calibration[key] = value
            self.save()
            with self.subTest(field=key), self.assertRaises(ValueError):
                inspect_bundle(self.root)
            self.calibration[key] = old

    def test_invalid_checkpoint_schema_is_rejected_before_inference(self):
        for value in ({'config': {}}, {'schema': SCHEMA, 'experiment': 'E-22', 'member_weights': [1,1], 'members': []}):
            with self.assertRaises(ValueError):
                checkpoint_config(value)

    def test_exact_encoder_keys_shapes_types_and_finite_values_required(self):
        model = torch.nn.Linear(3,2)
        correct = deepcopy(model.state_dict())
        load_encoder_state(model, correct)
        cases = [lambda s: s.pop('weight'), lambda s: s.update(unexpected=torch.ones(1)),
                 lambda s: s.update(weight=torch.ones(3,3)), lambda s: s.update(weight=s['weight'].double()),
                 lambda s: s['weight'].fill_(float('nan'))]
        for modify in cases:
            state = deepcopy(correct)
            modify(state)
            with self.assertRaises(ValueError):
                load_encoder_state(model, state)


if __name__ == '__main__':
    unittest.main()
