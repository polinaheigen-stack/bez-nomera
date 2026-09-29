"""Software replay against immutable native study outputs; no model/GPU run."""
import ast
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vehicle.server.e27_retrieval import POLICY, THRESHOLD, clear_cache, gallery_context
from vehicle.server.retrieval import rank_vectors, validate_retrieval
from vehicle.server.validation import write_outputs

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / 'fixtures/e27-control'


class E27ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = json.loads((FIXTURE / 'origin.json').read_text())
        for name, digest in cls.origin['files_sha256'].items():
            if hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Native fixture changed: ' + name)
        cls.order = json.loads((FIXTURE / 'embedding_order.json').read_text())
        cls.vectors = np.load(FIXTURE / 'embeddings.npy', allow_pickle=False)
        cls.query = cls.vectors[:len(cls.order['query'])]
        cls.gallery = cls.vectors[len(cls.order['query']):]
        with (FIXTURE / 'submission.csv').open(newline='') as stream:
            cls.expected_ranks = list(csv.reader(stream))
        with (FIXTURE / 'candidates.csv').open(newline='') as stream:
            cls.expected_candidates = {r['query_id']: (r['gallery_id'], float(r['confidence']))
                                       for r in csv.DictReader(stream)}

    def tearDown(self):
        clear_cache()

    def test_all_native_rankings_rejections_and_confidences(self):
        # Native study allows score atol=1e-5 across CPU BLAS; use stricter 1e-7.
        # Every top-10 rank and accept/refuse decision must still agree exactly.
        original = self.vectors.tobytes()
        for row, query in zip(self.expected_ranks, self.query):
            scores, order, accepted = rank_vectors(query, self.gallery, THRESHOLD, POLICY)
            self.assertEqual(row[1:], [self.order['gallery'][int(i)] for i in order[:10]])
            expected = self.expected_candidates.get(row[0])
            self.assertEqual(bool(accepted), expected is not None)
            if expected:
                self.assertEqual([self.order['gallery'][accepted[0]]], [expected[0]])
                self.assertAlmostEqual(float(scores[accepted[0]]), expected[1], delta=1e-7)
        self.assertEqual(self.vectors.tobytes(), original)

    def test_export_preserves_raw_vectors_and_native_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            result = write_outputs(path, self.order['query'], self.order['gallery'],
                                   self.query, self.gallery, THRESHOLD, POLICY)
            self.assertTrue(result['valid'])
            self.assertEqual((path / 'embeddings.npy').read_bytes(), (FIXTURE / 'embeddings.npy').read_bytes())
            self.assertEqual((path / 'submission.csv').read_bytes(), (FIXTURE / 'submission.csv').read_bytes())
            with (path / 'candidates.csv').open(newline='') as stream:
                actual = {r['query_id']: (r['gallery_id'], float(r['confidence'])) for r in csv.DictReader(stream)}
            self.assertEqual(set(actual), set(self.expected_candidates))
            for key, (image, score) in actual.items():
                self.assertEqual(image, self.expected_candidates[key][0])
                self.assertAlmostEqual(score, self.expected_candidates[key][1], delta=1e-7)

    def test_query_order_and_repeats_do_not_change_results(self):
        def answer(query):
            scores, order, accepted = rank_vectors(query, self.gallery, THRESHOLD, POLICY)
            return scores.copy(), order.copy(), accepted
        a = answer(self.query[0])
        answer(self.query[2])
        b = answer(self.query[0])
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2], b[2])

    def test_gallery_cache_owns_data_and_invalidates_changed_content(self):
        gallery = self.gallery.copy()
        first = gallery_context(gallery)
        self.assertIs(first, gallery_context(gallery.copy()))
        original = first.raw.copy()
        gallery[[0, 1]] = gallery[[1, 0]]
        second = gallery_context(gallery)
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(first.raw, original)
        self.assertFalse(first.raw.flags.writeable)
        self.assertFalse(first.mixed.flags.writeable)

    def test_refusal_keeps_top_ten(self):
        _, zero_order, _ = rank_vectors(self.query[0], self.gallery, 0., POLICY)
        _, one_order, accepted = rank_vectors(self.query[0], self.gallery, 1., POLICY)
        np.testing.assert_array_equal(zero_order, one_order)
        self.assertEqual(accepted, [])
        self.assertEqual(len(one_order[:10]), 10)

    def test_native_model_math_methods_are_unchanged(self):
        module = ast.parse((ROOT / 'vehicle/server/e27_model.py').read_text())
        model = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'CompactModel')
        for method in model.body:
            if isinstance(method, ast.FunctionDef) and method.name in self.origin['native_model_methods_ast_sha256']:
                actual = hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, self.origin['native_model_methods_ast_sha256'][method.name])

    def test_unknown_or_tuned_policy_is_rejected(self):
        for key, value in [('alpha', 0.25), ('gallery_neighbors', 2), ('mix', 0.5),
                           ('query_expansion', True), ('reject_all', True)]:
            policy = deepcopy(POLICY)
            policy[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_retrieval(policy)

    def test_invalid_vectors_are_rejected(self):
        for value in (np.zeros(384, np.float32), self.query[0].astype(np.float64),
                      np.full(384, np.nan, np.float32), np.ones(512, np.float32)):
            with self.assertRaises(ValueError):
                rank_vectors(value, self.gallery, THRESHOLD, POLICY)


if __name__ == '__main__':
    unittest.main()
