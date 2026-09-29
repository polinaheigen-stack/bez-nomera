"""Infrastructure contract tests. Synthetic provider is confined to this test file."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from vehicle.server.app import create_app
from vehicle.server.provider import ProductionProvider
from vehicle.server.service import Service, ServiceError, parse_csv


class TestProvider:
    available = True
    reason = None
    threshold = .8
    model = {'name': 'TEST ONLY', 'version': '1', 'sha256': 'a' * 64, 'dimension': 4, 'device': 'cpu-test'}
    calibration_sha256 = 'b' * 64
    delay = 0

    def embed(self, image, bbox, *, image_id):
        if self.delay:
            time.sleep(self.delay)
        if image_id == 'broken':
            raise ValueError('Intentional test inference failure')
        if image_id == 'reject':
            return np.array([-1, 0, 0, 0], dtype=np.float32)
        if image_id.startswith('g') and int(image_id[1:]) >= 3:
            return np.array([0, 1, 0, 0], dtype=np.float32)
        return np.array([1, 0, 0, 0], dtype=np.float32)


def image_bytes(size=(32, 24)):
    target = io.BytesIO()
    Image.new('RGB', size, '#667788').save(target, format='PNG')
    return target.getvalue()


def multipart(ids, field, value, box=(1, 2, 20, 15)):
    text = 'image_id,x,y,w,h\n' + ''.join(f'{i},{box[0]},{box[1]},{box[2]},{box[3]}\n' for i in ids)
    return {field: value}, [('csv', ('input.csv', text.encode(), 'text/csv'))] + [('images', (i + '.png', image_bytes(), 'image/png')) for i in ids]


class ApiTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'var' / 'tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.provider = TestProvider()
        self.app = create_app(self.provider, self.temp.name)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def wait(self, url, pending):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, response.text)
            value = response.json()
            if value['status'] not in pending:
                return value
            time.sleep(.01)
        self.fail('Job timed out')

    def gallery(self):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'Test gallery')
        response = self.client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        item = self.wait('/api/v1/galleries/' + response.json()['id'], {'indexing'})
        self.assertEqual(item['status'], 'ready')
        return item['id']

    def run_ids(self, ids, gid=None):
        gid = gid or self.gallery()
        data, files = multipart(ids, 'gallery_id', gid)
        response = self.client.post('/api/v1/runs', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        return self.wait('/api/v1/runs/' + response.json()['id'], {'queued', 'running'})

    def test_run_timing_counts_refusal_and_freezes_before_export(self):
        run = self.run_ids(['query', 'reject'])
        timing = run['timing']
        self.assertEqual(timing['successful_images'], 2)
        self.assertGreater(timing['processing_ms'], 0)
        self.assertGreater(timing['images_per_second'], 0)
        self.assertGreaterEqual(timing['queue_ms'], 0)
        self.assertAlmostEqual(timing['mean_image_ms'], sum(r['duration_ms'] for r in run['results']) / 2, places=2)
        evidence = self.client.get('/api/v1/runs/' + run['id'] + '/evidence').json()
        self.assertEqual(evidence['timing'], timing)
        with zipfile.ZipFile(io.BytesIO(self.client.get(run['artifacts_url']).content)) as archive:
            self.assertEqual(json.loads(archive.read('provenance.json'))['timing'], timing)
        time.sleep(.01)
        self.assertEqual(self.client.get('/api/v1/runs/' + run['id']).json()['timing'], timing)

    def test_actual_prod_unavailable_without_artifacts(self):
        with patch.dict(os.environ, {}, clear=True):
            provider = ProductionProvider()
        with tempfile.TemporaryDirectory(dir=self.temp.name) as root, TestClient(create_app(provider, root)) as client:
            status = client.get('/api/v1/status').json()
            self.assertFalse(status['available'])
            self.assertIsNone(status['model'])
            self.assertEqual(client.post('/api/v1/runs').status_code, 503)
            self.assertEqual(client.post('/api/v1/galleries').status_code, 503)
            self.assertEqual(client.get('/api/demo/v1/status').status_code, 404)
            report = client.get('/api/v1/report').json()
            self.assertFalse(report['measured'])
            self.assertIsNone(report['metrics'])

    def test_removed_demo_cannot_be_enabled_by_legacy_environment(self):
        with patch.dict(os.environ, {'VEHICLE_ENABLE_DEMO': '1'}, clear=True), \
                tempfile.TemporaryDirectory(dir=self.temp.name) as root, TestClient(create_app(data_root=root)) as client:
            status = client.get('/api/v1/status').json()
            self.assertEqual(status['mode'], 'prod')
            self.assertFalse(status['available'])
            self.assertEqual(client.get('/api/demo/v1/status').status_code, 404)
            self.assertEqual(client.get('/api/demo/v1/scenarios').status_code, 404)
            self.assertFalse(any('/demo/' in path for path in client.get('/openapi.json').json()['paths']))
            self.assertIsNone(client.get('/api/v1/report').json()['metrics'])

    def test_export_schema_order_rejection_and_stable_ties(self):
        run = self.run_ids(['query', 'reject'])
        self.assertEqual(run['status'], 'completed')
        self.assertEqual([r['status'] for r in run['results']], ['matched', 'rejected'])
        self.assertEqual([c['image']['image_id'] for c in run['results'][0]['candidates'][:3]], ['g0', 'g1', 'g2'])
        self.assertFalse(any(c['accepted'] for c in run['results'][1]['candidates']))
        response = self.client.get(run['artifacts_url'])
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            rows = list(csv.reader(io.StringIO(archive.read('submission.csv').decode())))
            self.assertEqual([r[0] for r in rows], ['query', 'reject'])
            self.assertTrue(all(len(r) == 11 and len(set(r[1:])) == 10 for r in rows))
            candidates = list(csv.DictReader(io.StringIO(archive.read('candidates.csv').decode())))
            self.assertEqual({r['query_id'] for r in candidates}, {'query'})
            vectors = np.load(io.BytesIO(archive.read('embeddings.npy')), allow_pickle=False)
            self.assertEqual(vectors.shape, (12, 4))
            self.assertEqual(vectors.dtype, np.float32)
            order = json.loads(archive.read('embedding_order.json'))
            self.assertEqual(order['query'], ['query', 'reject'])
            self.assertEqual(order['gallery'], [f'g{i}' for i in range(10)])
            proof = json.loads(archive.read('provenance.json'))
            self.assertEqual(proof['status'], 'completed')
            self.assertEqual(proof['mode'], 'prod')
            self.assertEqual(set(proof['files_sha256']), {'submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json'})
            self.assertIn('events.jsonl', archive.namelist())
        ref = run['results'][0]['query']
        with Image.open(io.BytesIO(self.client.get(ref['crop_url']).content)) as image:
            self.assertEqual(image.size, (20, 15))
        self.assertEqual(self.client.get('/api/v1/assets/' + '0' * 32).status_code, 404)

    def test_errors_are_not_rejections_and_prevent_export(self):
        run = self.run_ids(['query', 'broken', 'reject'])
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(run['failed'], 1)
        self.assertEqual([r['status'] for r in run['results']], ['matched', 'error', 'rejected'])
        self.assertFalse(run['export_ready'])
        response = self.client.get(f"/api/v1/runs/{run['id']}/export")
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json()['error']['request_id'])

    def test_bbox_validation_and_atomic_cleanup(self):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'bad', (1, 2, 100, 20))
        response = self.client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get('/api/v1/galleries').json(), [])
        self.assertEqual(list((Path(self.temp.name) / 'prod').iterdir()), [])

    def test_upload_path_and_type_are_rejected(self):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'bad')
        files[1] = ('images', ('../g0.png', image_bytes(), 'image/png'))
        self.assertEqual(self.client.post('/api/v1/galleries', data=data, files=files).status_code, 422)
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'bad')
        files[1] = ('images', ('g0.png', b'not an image', 'image/png'))
        with self.assertLogs('vehicle.server.service', level='WARNING') as logged:
            response = self.client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error']['code'], 'INVALID_IMAGE')
        self.assertEqual(response.json()['error']['message'], 'Не удалось прочитать изображение. Загрузите неповреждённый JPEG или PNG.')
        self.assertNotIn('upload-', response.text)
        self.assertNotIn(self.temp.name, response.text)
        self.assertTrue(any('UnidentifiedImageError' in entry for entry in logged.output))
        self.assertEqual(self.client.get('/api/v1/galleries').json(), [])
        self.assertEqual(list((Path(self.temp.name) / 'prod').iterdir()), [])

    def test_csv_duplicate_ids_nonfinite_and_too_small_gallery(self):
        for text in ('image_id,x,y,w,h\na,0,0,1,1\na,0,0,1,1\n', 'image_id,x,y,w,h\na,NaN,0,1,1\n', 'image_id,x,y,w,h,camera_id\na,0,0,1,1,4\n'):
            with self.assertRaises(ServiceError):
                parse_csv(text.encode())
        data, files = multipart(['g0'], 'name', 'small')
        self.assertEqual(self.client.post('/api/v1/galleries', data=data, files=files).status_code, 422)

    def test_cancel_is_terminal_and_no_export(self):
        gid = self.gallery()
        self.provider.delay = .08
        data, files = multipart([f'q{i}' for i in range(8)], 'gallery_id', gid)
        run = self.client.post('/api/v1/runs', data=data, files=files).json()
        result = self.client.post(f"/api/v1/runs/{run['id']}/cancel").json()
        self.assertEqual(result['status'], 'cancelled')
        time.sleep(.12)
        result = self.client.get(f"/api/v1/runs/{run['id']}").json()
        self.assertEqual(result['status'], 'cancelled')
        self.assertFalse(result['export_ready'])
        self.assertEqual(self.client.get(f"/api/v1/runs/{run['id']}/export").status_code, 409)

    def test_arbitrary_original_dimensions_and_input_independence(self):
        gid = self.gallery()
        first = self.run_ids(['query'], gid)
        second = self.run_ids(['reject', 'query'], gid)
        self.assertEqual(first['results'][0]['candidates'], second['results'][1]['candidates'])
        data, files = multipart(['large'], 'gallery_id', gid, (10, 20, 900, 700))
        files[1] = ('images', ('large.png', image_bytes((1600, 1200)), 'image/png'))
        response = self.client.post('/api/v1/runs', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        result = self.wait('/api/v1/runs/' + response.json()['id'], {'queued', 'running'})
        self.assertEqual(result['results'][0]['query']['width'], 1600)

    def test_persistence_and_only_production_mode(self):
        run = self.run_ids(['query'])
        restored = Service('prod', self.provider, self.temp.name)
        try:
            self.assertEqual(restored.get_run(run['id'])['status'], 'completed')
            self.assertTrue(restored.export_path(run['id']).is_file())
            self.assertEqual(restored.get_gallery(run['gallery_id'])['status'], 'ready')
            with self.assertRaisesRegex(ValueError, 'Only production'):
                Service('demo', self.provider, self.temp.name)
        finally:
            restored.close()

    def test_history_is_newest_first_with_id_ties_after_restore(self):
        gallery_ids = [self.gallery() for _ in range(4)]
        run_ids = [self.run_ids(['query'], gid)['id'] for gid in gallery_ids]
        service = self.app.state.services[0]
        times = ['2026-09-24T10:00:00+00:00', '2026-09-24T12:00:00+00:00',
                 '2026-09-24T11:00:00+00:00', '2026-09-24T12:00:00+00:00']
        with service.lock:
            for gid, rid, timestamp in zip(gallery_ids, run_ids, times):
                service.galleries[gid]['public']['created_at'] = timestamp
                service.runs[rid]['public']['created_at'] = timestamp
                service._save(service.galleries[gid], 'gallery')
                service._save(service.runs[rid], 'run')
        expected_galleries = [max(gallery_ids[1], gallery_ids[3]), min(gallery_ids[1], gallery_ids[3]), gallery_ids[2], gallery_ids[0]]
        expected_runs = [max(run_ids[1], run_ids[3]), min(run_ids[1], run_ids[3]), run_ids[2], run_ids[0]]
        self.assertEqual([g['id'] for g in self.client.get('/api/v1/galleries').json()], expected_galleries)
        self.assertEqual([r['id'] for r in self.client.get('/api/v1/runs').json()], expected_runs)
        # Filesystem enumeration is not chronological and must never define UI history.
        state_files = [service.root / object_id / 'state.json' for object_id in expected_galleries + expected_runs]
        with patch.object(Path, 'glob', return_value=iter(state_files)):
            restored = Service('prod', self.provider, self.temp.name)
        try:
            self.assertEqual([g['id'] for g in restored.list_galleries()], expected_galleries)
            self.assertEqual([r['id'] for r in restored.list_runs()], expected_runs)
            self.assertTrue(all(r['status'] == 'completed' for r in restored.list_runs()))
        finally:
            restored.close()

    def test_report_requires_model_provenance_and_valid_ranges(self):
        report = {'mode': 'prod', 'measured': True, 'model': self.provider.model, 'dataset': 'test-heldout', 'hardware': 'test-only', 'run_id': 'evidence-test', 'metrics': {'map_at_10': .4}, 'notes': []}
        path = Path(self.temp.name) / 'report.json'
        path.write_text(json.dumps(report), encoding='utf-8')
        with patch.dict(os.environ, {'VEHICLE_EVALUATION_REPORT': str(path)}):
            self.assertTrue(self.client.get('/api/v1/report').json()['measured'])
            report['metrics']['map_at_10'] = 5
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertFalse(self.client.get('/api/v1/report').json()['measured'])
            report['metrics']['map_at_10'] = .4
            report['model'] = {**self.provider.model, 'sha256': 'c' * 64}
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertFalse(self.client.get('/api/v1/report').json()['measured'])

    def test_report_directory_selects_exact_runtime_and_calibration_without_restart(self):
        self.provider.inference_fingerprint = '1' * 64
        self.provider.model = {**self.provider.model, 'inference_fingerprint': self.provider.inference_fingerprint}
        directory = Path(self.temp.name) / 'reports'
        directory.mkdir()
        report = {'mode': 'prod', 'measured': True, 'model': self.provider.model,
                  'calibration_sha256': self.provider.calibration_sha256, 'dataset': 'test-heldout',
                  'hardware': 'test-only', 'run_id': 'report-directory-test', 'metrics': {'map_at_10': .4}, 'notes': []}
        path = directory / f'{self.provider.inference_fingerprint}-{self.provider.calibration_sha256}.json'
        with patch.dict(os.environ, {'VEHICLE_EVALUATION_REPORT': '', 'VEHICLE_EVALUATION_REPORTS_DIR': str(directory)}):
            missing = self.client.get('/api/v1/report').json()
            self.assertFalse(missing['measured'])
            self.assertIn('ещё не подключён', ' '.join(missing['notes']))
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertTrue(self.client.get('/api/v1/report').json()['measured'])
            self.provider.inference_fingerprint = '2' * 64
            self.assertFalse(self.client.get('/api/v1/report').json()['measured'])
            self.provider.inference_fingerprint = '1' * 64
            self.provider.calibration_sha256 = 'c' * 64
            self.assertFalse(self.client.get('/api/v1/report').json()['measured'])
            self.provider.calibration_sha256 = 'b' * 64
            self.assertTrue(self.client.get('/api/v1/report').json()['measured'])
            report['model'] = {**report['model'], 'inference_fingerprint': '3' * 64}
            path.write_text(json.dumps(report), encoding='utf-8')
            rejected = self.client.get('/api/v1/report').json()
            self.assertFalse(rejected['measured'])
            self.assertIn('отклонён', ' '.join(rejected['notes']))

    def test_empty_or_unmeasured_metric_payload_is_not_a_measurement(self):
        report = {'mode': 'prod', 'measured': True, 'model': self.provider.model, 'dataset': 'test-heldout',
                  'hardware': 'test-only', 'run_id': 'evidence-test', 'metrics': {}, 'notes': []}
        path = Path(self.temp.name) / 'empty-report.json'
        cases = ({'measured': True, 'metrics': {}}, {'measured': True, 'metrics': None},
                 {'measured': True, 'metrics': {'map_at_10': None}},
                 {'measured': False, 'metrics': {'map_at_10': .9}}, {'mode': 'demo', 'metrics': {'map_at_10': .9}})
        for change in cases:
            path.write_text(json.dumps({**report, **change}), encoding='utf-8')
            with self.subTest(change=change), patch.dict(os.environ, {'VEHICLE_EVALUATION_REPORT': str(path)}):
                result = self.client.get('/api/v1/report').json()
                self.assertFalse(result['measured'])
                self.assertIsNone(result['metrics'])

    def test_openapi_and_unknown_routes_stay_json(self):
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/api/v1/runs/{run_id}/evidence', schema['paths'])
        self.assertEqual(self.client.get('/api/v1/no-such-method').status_code, 404)
        self.assertIn('error', self.client.get('/api/v1/no-such-method').json())
        self.assertEqual(self.client.get('/docs').status_code, 404)

    def test_declared_oversize_is_rejected_before_body(self):
        response = self.client.post('/api/v1/runs', content=b'', headers={'content-length': str(3 * 1024**3)})
        self.assertEqual(response.status_code, 413)
        self.assertIn('request_id', response.json()['error'])

    def test_storage_quota_rejects_without_partial_results(self):
        service = self.app.state.services[0]
        service.storage_limit = 1
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'limited')
        response = self.client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()['error']['code'], 'STORAGE_FULL')
        self.assertEqual(self.client.get('/api/v1/galleries').json(), [])

    def test_queue_bound_and_changed_model_rejects_cached_gallery(self):
        gid = self.gallery()
        service = self.app.state.services[0]
        deadline = time.monotonic() + 2
        while service.pending and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(service.pending, 0)
        service.pending = service.limits.max_pending_runs
        data, files = multipart(['query'], 'gallery_id', gid)
        self.assertEqual(self.client.post('/api/v1/runs', data=data, files=files).status_code, 429)
        service.pending = 0
        different = TestProvider()
        different.model = {**different.model, 'sha256': 'c' * 64}
        restored = Service('prod', different, self.temp.name)
        try:
            self.assertEqual(restored.get_gallery(gid)['status'], 'failed')
        finally:
            restored.close()

    def test_no_camera_or_vehicle_ids_in_model_request(self):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'clean')
        header = b'image_id,x,y,w,h,vehicle_id\n'
        text = header + b''.join(f'g{i},1,2,20,15,99\n'.encode() for i in range(10))
        files[0] = ('csv', ('input.csv', text, 'text/csv'))
        self.assertEqual(self.client.post('/api/v1/galleries', data=data, files=files).status_code, 422)

    def test_preprocessing_fingerprint_invalidates_gallery_but_threshold_does_not(self):
        self.provider.inference_fingerprint = 'd' * 64
        gid = self.gallery()
        changed = TestProvider()
        changed.inference_fingerprint = 'e' * 64
        restored = Service('prod', changed, self.temp.name)
        try:
            self.assertEqual(restored.get_gallery(gid)['status'], 'failed')
        finally:
            restored.close()
        changed.inference_fingerprint = self.provider.inference_fingerprint
        changed.threshold = .95
        restored = Service('prod', changed, self.temp.name)
        try:
            self.assertEqual(restored.get_gallery(gid)['status'], 'ready')
        finally:
            restored.close()

    def test_model_change_in_same_process_rejects_stale_gallery(self):
        gid = self.gallery()
        self.provider.inference_fingerprint = 'f' * 64
        data, files = multipart(['query'], 'gallery_id', gid)
        result = self.client.post('/api/v1/runs', data=data, files=files)
        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.json()['error']['code'], 'GALLERY_MODEL_CHANGED')


if __name__ == '__main__':
    unittest.main()
