"""Per-run labels score immutable predictions; they never affect inference."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient

from vehicle.server.app import create_app
from vehicle.server.service import ServiceError
from vehicle.tests.test_api import TestProvider, multipart


def labels(query_ids=('query', 'reject')):
    rows = ['image_id,vehicle_id,camera_id,split']
    rows.extend(f'{qid},{"A" if qid == "query" else "Z"},0,query' for qid in query_ids)
    rows.extend(f'g{i},{"A" if i < 3 else "B" + str(i)},1,gallery' for i in range(10))
    return ('\n'.join(rows) + '\n').encode()


class RunEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = TestProvider()
        self.app = create_app(self.provider, self.temp.name)
        self.client = TestClient(self.app).__enter__()
        self.service = self.app.state.services[0]
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def wait(self, url, pending):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, response.text)
            value = response.json()
            if value['status'] not in pending:
                return value
            time.sleep(.01)
        self.fail('Job timed out')

    def gallery(self):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'Evaluation test')
        response = self.client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        gallery = self.wait('/api/v1/galleries/' + response.json()['id'], {'indexing'})
        self.assertEqual(gallery['status'], 'ready')
        return gallery['id']

    def create_run(self, ids=('query', 'reject'), gallery=None, wait=True):
        data, files = multipart(ids, 'gallery_id', gallery or self.gallery())
        response = self.client.post('/api/v1/runs', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        run = response.json()
        return self.wait('/api/v1/runs/' + run['id'], {'queued', 'running'}) if wait else run

    def post_labels(self, rid, contents=None):
        return self.client.post('/api/v1/runs/' + rid + '/evaluation',
                                files={'ground_truth': ('ground_truth.csv', labels() if contents is None else contents, 'text/csv')})

    def wait_quality(self, rid):
        return self.wait('/api/v1/runs/' + rid + '/evaluation', {'waiting', 'running'})

    def block_query(self):
        original = self.provider.embed
        def embed(image, bbox, *, image_id):
            if image_id == 'query':
                if not self.release.wait(10):
                    raise RuntimeError('Test gate timed out')
            return original(image, bbox, image_id=image_id)
        self.provider.embed = embed

    def test_no_labels_unavailable_and_actual_report_bound_to_run(self):
        run = self.create_run()
        rid = run['id']
        self.assertEqual(run['evaluation'], {'status': 'unavailable', 'report': None, 'error': None, 'ground_truth_sha256': None})
        self.assertEqual(self.client.get('/api/v1/runs/' + rid + '/evaluation/report').status_code, 409)
        original_zip = self.client.get(run['artifacts_url']).content
        response = self.post_labels(rid)
        self.assertEqual(response.status_code, 202, response.text)
        state = self.wait_quality(rid)
        self.assertEqual(state['status'], 'completed', state)
        self.assertEqual(state['ground_truth_sha256'], hashlib.sha256(labels()).hexdigest())
        self.assertEqual(state['report']['run_id'], rid)
        self.assertEqual(state['report']['metrics']['rank_1'], 1.)
        self.assertEqual(state['report']['metrics']['tnr'], 1.)
        self.assertIn('именно этого запуска', state['report']['notes'][0])
        self.assertEqual(self.client.get('/api/v1/runs/' + rid).json()['evaluation'], state)
        self.assertEqual(self.client.get('/api/v1/runs').json()[0]['evaluation'], state)
        self.assertEqual(self.client.get(run['artifacts_url']).content, original_zip)
        response = self.client.get('/api/v1/runs/' + rid + '/evaluation/report')
        self.assertEqual(response.status_code, 200, response.text)
        full = response.json()
        self.assertEqual(full['run_id'], rid)
        self.assertEqual(full['scope'], 'exact_web_run_with_supplied_labels')
        self.assertEqual(full['ground_truth_sha256'], state['ground_truth_sha256'])
        self.assertEqual(full['official']['ranking']['Rank-1'], 1.)
        self.assertIn('export_sha256', full)
        self.assertIn('attachment;', response.headers['content-disposition'])

    def test_waits_for_export_duplicate_rejected_and_labels_do_not_affect_search(self):
        gid = self.gallery()
        reference = self.create_run(gallery=gid)
        self.block_query()
        running = self.create_run(gallery=gid, wait=False)
        rid = running['id']
        response = self.post_labels(rid)
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()['status'], 'waiting')
        self.assertIsNone(response.json()['report'])
        self.assertEqual(self.post_labels(rid).status_code, 409)
        self.release.set()
        result = self.wait('/api/v1/runs/' + rid, {'queued', 'running'})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.wait_quality(rid)['status'], 'completed')
        with zipfile.ZipFile(io.BytesIO(self.client.get(reference['artifacts_url']).content)) as a, \
             zipfile.ZipFile(io.BytesIO(self.client.get(result['artifacts_url']).content)) as b:
            for name in ('submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json'):
                self.assertEqual(a.read(name), b.read(name))

    def test_replacement_failure_and_wrong_scope_cannot_show_old_report(self):
        run = self.create_run()
        rid = run['id']
        self.assertEqual(self.post_labels(rid).status_code, 202)
        self.assertEqual(self.wait_quality(rid)['status'], 'completed')
        with patch('vehicle.server.evaluation.evaluate_export', side_effect=RuntimeError('Intentional evaluator failure')):
            response = self.post_labels(rid)
            self.assertEqual(response.status_code, 202)
            self.assertIsNone(response.json()['report'])
            state = self.wait_quality(rid)
        self.assertEqual(state['status'], 'failed')
        self.assertIsNone(state['report'])
        self.assertEqual(self.client.get('/api/v1/runs/' + rid + '/evaluation/report').status_code, 409)
        bad = labels().replace(b'g9,B9,1,gallery\n', b'foreign,B9,1,gallery\n')
        self.assertEqual(self.post_labels(rid, bad).status_code, 422)
        self.assertIsNone(self.client.get('/api/v1/runs/' + rid + '/evaluation').json()['report'])
        self.assertEqual(self.post_labels(rid).status_code, 202)
        self.assertEqual(self.wait_quality(rid)['status'], 'completed')

    def test_cancelled_or_failed_inference_settles_waiting_quality(self):
        gid = self.gallery()
        self.block_query()
        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                self.release.clear()
                ids = ('query', 'reject') if cancel else ('query', 'broken')
                run = self.create_run(ids, gallery=gid, wait=False)
                self.assertEqual(self.post_labels(run['id'], labels(ids)).json()['status'], 'waiting')
                if cancel:
                    self.client.post('/api/v1/runs/' + run['id'] + '/cancel')
                self.release.set()
                terminal = self.wait('/api/v1/runs/' + run['id'], {'queued', 'running'})
                self.assertEqual(terminal['status'], 'cancelled' if cancel else 'failed')
                quality = self.wait_quality(run['id'])
                self.assertEqual(quality['status'], 'failed')
                self.assertIsNone(quality['report'])
                self.assertEqual(self.post_labels(run['id'], labels(ids)).status_code, 409)

    def test_limits_unknown_run_and_storage_failure_do_not_stick_waiting(self):
        self.assertEqual(self.post_labels('0' * 32).status_code, 404)
        run = self.create_run()
        rid = run['id']
        self.assertEqual(self.post_labels(rid, b'x' * (1024 * 1024 + 1)).status_code, 413)
        self.assertEqual(self.post_labels(rid, b'x' * (1024 * 1024 + 65537)).status_code, 413)
        self.assertEqual(self.client.post('/api/v1/runs/' + rid + '/evaluation', files={'wrong': ('x.csv', labels())}).status_code, 422)
        with patch.object(self.service, 'ensure_storage', side_effect=ServiceError(413, 'STORAGE_FULL', 'No disk')):
            self.assertEqual(self.post_labels(rid).status_code, 413)
        state = self.client.get('/api/v1/runs/' + rid + '/evaluation').json()
        self.assertEqual(state['status'], 'failed')
        self.assertIsNone(state['report'])
        self.assertEqual(self.post_labels(rid).status_code, 202)
        self.assertEqual(self.wait_quality(rid)['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
