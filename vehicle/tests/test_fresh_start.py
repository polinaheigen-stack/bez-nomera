"""Fresh web startup clears only owned uploads; browser navigation never clears them."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from vehicle.server.app import create_app
from vehicle.tests.test_api import TestProvider, multipart
from vehicle.tests.test_runtime_settings import Provider, available_devices


class FreshStartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = self.root / 'web-state'

    def app(self, reset):
        with patch.dict(os.environ, {'VEHICLE_RESET_ON_START': reset}):
            app = create_app(TestProvider(), data_root=self.storage)
        self.addCleanup(app.state.services[0].close)
        return app

    def wait(self, client, url, active):
        until = time.monotonic() + 10
        while time.monotonic() < until:
            response = client.get(url)
            self.assertEqual(response.status_code, 200, response.text)
            value = response.json()
            if value['status'] not in active:
                return value
            time.sleep(.01)
        self.fail('Test inference did not finish.')

    def gallery(self, client):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'Startup contract')
        response = client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        gallery = self.wait(client, '/api/v1/galleries/' + response.json()['id'], {'indexing'})
        self.assertEqual(gallery['status'], 'ready')
        return gallery['id']

    def seed_completed_state(self):
        with TestClient(self.app('0')) as client:
            gid = self.gallery(client)
            data, files = multipart(['query'], 'gallery_id', gid)
            response = client.post('/api/v1/runs', data=data, files=files)
            self.assertEqual(response.status_code, 202, response.text)
            run = self.wait(client, '/api/v1/runs/' + response.json()['id'], {'queued', 'running'})
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(client.get(run['artifacts_url']).status_code, 200)
        return gid, run

    def test_reset_occurs_at_startup_and_preserves_settings_and_external_materials(self):
        gid, run = self.seed_completed_state()
        prod = self.storage / 'prod'
        settings = prod / 'runtime-settings.json'
        settings.write_bytes(b'{"device": "cpu"}\n')
        upload = prod / 'upload-abc_1234'
        upload.mkdir()
        (upload / 'unfinished-image').write_bytes(b'partial upload')
        preserved = {
            settings: settings.read_bytes(),
            self.root / 'models/e22.pth': b'outside-storage-model-sentinel',
            self.root / 'reports/evaluation-report.json': b'outside-storage-report-sentinel',
            self.root / 'dataset/images/source.jpg': b'outside-storage-input-sentinel',
            self.storage / 'other-mode/keep.txt': b'outside-prod',
            prod / 'notes/keep.txt': b'not-an-owned-upload-directory',
            prod / 'upload-manual/keep.txt': b'not-a-temporary-upload-directory',
        }
        for path, contents in preserved.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)

        app = self.app('1')
        # Merely importing/constructing the ASGI app must not destroy state.
        self.assertTrue((prod / gid / 'state.json').is_file())
        self.assertTrue((prod / run['id'] / 'export.zip').is_file())
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/v1/galleries').json(), [])
            self.assertEqual(client.get('/api/v1/runs').json(), [])
            self.assertFalse((prod / gid).exists())
            self.assertFalse((prod / run['id']).exists())
            self.assertFalse(upload.exists())
            for url in (f'/api/v1/galleries/{gid}', f'/api/v1/runs/{run["id"]}',
                        run['artifacts_url'], run['results'][0]['query']['image_url']):
                self.assertEqual(client.get(url).status_code, 404, url)
            for path, contents in preserved.items():
                self.assertEqual(path.read_bytes(), contents, str(path))

    def test_browser_refresh_and_repeated_runtime_start_keep_current_uploads(self):
        with TestClient(self.app('1')) as client:
            gid = self.gallery(client)
            service = client.app.state.services[0]
            service.start_runtime()
            for _ in range(2):
                client.get('/')
                client.get('/api/v1/status')
                client.get('/api/v1/report')
                galleries = client.get('/api/v1/galleries').json()
                self.assertEqual([item['id'] for item in galleries], [gid])
                self.assertEqual(galleries[0]['status'], 'ready')
            self.assertTrue((self.storage / 'prod' / gid / 'state.json').is_file())

    def test_reset_disabled_restores_completed_gallery_and_run(self):
        gid, run = self.seed_completed_state()
        with TestClient(self.app('0')) as client:
            self.assertEqual([item['id'] for item in client.get('/api/v1/galleries').json()], [gid])
            restored = client.get('/api/v1/runs/' + run['id']).json()
            self.assertEqual(restored['status'], 'completed')
            self.assertEqual(client.get(restored['artifacts_url']).status_code, 200)

    def test_saved_cpu_choice_is_used_after_fresh_start(self):
        settings = self.storage / 'prod/runtime-settings.json'
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({'device': 'cpu'}), encoding='utf-8')
        calls = []

        def factory(*, device):
            calls.append(device)
            return Provider(device)

        with patch.dict(os.environ, {'VEHICLE_RESET_ON_START': '1'}):
            app = create_app(data_root=self.storage, provider_factory=factory, device_probe=available_devices)
        self.addCleanup(app.state.services[0].close)
        with TestClient(app) as client:
            until = time.monotonic() + 10
            while time.monotonic() < until:
                current = client.get('/api/v1/settings').json()
                if not current['switching']:
                    break
                time.sleep(.01)
            self.assertEqual(current['selected_device'], 'cpu')
            self.assertEqual(current['active_device'], 'cpu')
            self.assertEqual(calls, ['cpu'])
            self.assertEqual(json.loads(settings.read_text())['device'], 'cpu')

    def test_invalid_reset_flag_is_rejected(self):
        for value in ('true', 'false', 'yes', '2', ''):
            with self.subTest(value=value), patch.dict(os.environ, {'VEHICLE_RESET_ON_START': value}):
                with self.assertRaises(ValueError):
                    create_app(TestProvider(), data_root=self.storage)

    def test_owned_directory_link_cannot_delete_outside_storage(self):
        prod = self.storage / 'prod'
        prod.mkdir(parents=True)
        outside = self.root / 'external-data'
        outside.mkdir()
        sentinel = outside / 'keep.txt'
        sentinel.write_bytes(b'must survive')
        link = prod / ('a' * 32)
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f'Directory symlink unavailable: {error}')
        app = self.app('1')
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            with TestClient(app):
                pass
        self.assertEqual(sentinel.read_bytes(), b'must survive')


if __name__ == '__main__':
    unittest.main()
