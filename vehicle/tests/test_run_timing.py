"""Service timing semantics only; these numbers are not recognition benchmarks."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from vehicle.server.service import Service
from vehicle.server.schemas import RunTiming


class RunTimingTests(unittest.TestCase):
    def setUp(self):
        self.service = Service.__new__(Service)
        self.item = {'public': {'results': [
            {'status': 'matched', 'duration_ms': 200.},
            {'status': 'rejected', 'duration_ms': 100.},
            {'status': 'error', 'duration_ms': 50.}]},
            '_queued_monotonic': 10., '_started_monotonic': 12.}

    def test_queue_and_failed_images_are_excluded_from_rate(self):
        before = deepcopy(self.item)
        with patch('vehicle.server.service.time.perf_counter', return_value=13.):
            timing = self.service._timing(self.item)
        self.assertEqual(timing, {'queue_ms': 2000., 'processing_ms': 1000.,
            'successful_images': 2, 'images_per_second': 2., 'mean_image_ms': 150.})
        RunTiming.model_validate(timing)
        self.assertEqual(before, self.item)

    def test_timing_stops_before_export_or_after_cancellation(self):
        self.item['_processing_finished'] = 13.
        with patch('vehicle.server.service.time.perf_counter', return_value=999.):
            timing = self.service._timing(self.item)
        self.assertEqual(timing['processing_ms'], 1000.)
        self.assertEqual(timing['images_per_second'], 2.)

    def test_no_measurement_is_invented_before_results(self):
        self.item['public']['results'] = []
        with patch('vehicle.server.service.time.perf_counter', return_value=12.):
            timing = self.service._timing(self.item)
        self.assertIsNone(timing['images_per_second'])
        self.assertIsNone(timing['mean_image_ms'])
        self.assertIsNone(self.service._timing({'public': {'results': []}}))

    def test_persisted_timing_remains_stable_after_restart(self):
        saved = {'queue_ms': 0., 'processing_ms': 1000., 'successful_images': 2,
                 'images_per_second': 2., 'mean_image_ms': 150.}
        self.assertEqual(self.service._timing({'public': {'timing': saved}}), saved)


if __name__ == '__main__':
    unittest.main()
