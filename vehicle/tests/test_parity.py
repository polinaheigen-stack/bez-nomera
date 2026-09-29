import tempfile
from pathlib import Path
import unittest

import numpy as np

from vehicle.server.parity import compare_exports
from vehicle.server.validation import write_outputs


class ParityTests(unittest.TestCase):
    def test_same_export_passes_and_changed_refusal_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, actual = Path(folder) / 'ref', Path(folder) / 'actual'
            query_ids, gallery_ids = ['q'], [f'g{i}' for i in range(10)]
            query = np.array([[1., 0.]], dtype=np.float32)
            gallery = np.tile(np.array([[.6, .8]], dtype=np.float32), (10, 1))
            for path in (reference, actual):
                write_outputs(path, query_ids, gallery_ids, query, gallery, .7)
            report = compare_exports(reference, actual, query_ids, gallery_ids)
            self.assertTrue(report['passed'])
            self.assertFalse(report['quality_claim'])
            write_outputs(actual, query_ids, gallery_ids, query, gallery, .9)
            report = compare_exports(reference, actual, query_ids, gallery_ids)
            self.assertFalse(report['passed'])
            self.assertTrue(report['checks']['embeddings'])
            self.assertFalse(report['checks']['refusals'])

    def test_vector_change_detected_even_with_same_ranking(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, actual = Path(folder) / 'ref', Path(folder) / 'actual'
            query_ids, gallery_ids = ['q'], [f'g{i}' for i in range(10)]
            gallery = np.tile(np.array([[1., 0.]], dtype=np.float32), (10, 1))
            write_outputs(reference, query_ids, gallery_ids, np.array([[1., 0.]], dtype=np.float32), gallery, .5)
            write_outputs(actual, query_ids, gallery_ids, np.array([[.8, .6]], dtype=np.float32), gallery, .5)
            report = compare_exports(reference, actual, query_ids, gallery_ids)
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['embeddings'])
            self.assertTrue(report['checks']['ranking'])


if __name__ == '__main__':
    unittest.main()
