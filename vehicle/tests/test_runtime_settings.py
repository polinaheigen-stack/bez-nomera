"""Device transition contracts. Synthetic inference exists only in this test file."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from starlette.requests import Request

from vehicle.server.app import create_app
from vehicle.server.provider import ProductionProvider
from vehicle.tests.test_api import multipart


def available_devices():
    return [{'id': device, 'name': device.upper(), 'available': True, 'reason': None}
            for device in ('cuda', 'cpu')]


class Provider:
    available, reason, threshold = True, None, .8
    calibration_sha256 = 'b' * 64

    def __init__(self, device):
        self.inference_fingerprint = f'test-fingerprint-{device}'
        self.model = {'name': 'TEST ONLY', 'version': '1', 'sha256': 'a' * 64,
                      'dimension': 4, 'device': device, 'inference_fingerprint': self.inference_fingerprint}
        self.entered = self.release = None
        self.fail = False

    def embed(self, image, bbox, *, image_id):
        if self.entered:
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError('Test gate timed out')
        if self.fail:
            raise ValueError('Intentional test reindex failure')
        return np.array([1, 0, 0, 0] if self.model['device'] == 'cpu' else [0, 1, 0, 0], dtype=np.float32)


class RuntimeSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []

    def factory(self, *, device):
        self.calls.append(device)
        return Provider(device)

    def app(self, **kwargs):
        return create_app(data_root=self.temp.name, provider_factory=kwargs.pop('provider_factory', self.factory),
                          device_probe=kwargs.pop('device_probe', available_devices), **kwargs)

    def wait(self, client, predicate, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.01)
        self.fail('Background operation timed out')

    def idle(self, client):
        self.wait(client, lambda: not client.get('/api/v1/settings').json()['switching']
                  and client.app.state.services[0].pending == 0)
        return client.get('/api/v1/settings').json()

    def gallery(self, client):
        data, files = multipart([f'g{i}' for i in range(10)], 'name', 'Persistent gallery')
        response = client.post('/api/v1/galleries', data=data, files=files)
        self.assertEqual(response.status_code, 202, response.text)
        self.idle(client)
        gid = response.json()['id']
        self.assertEqual(client.get(f'/api/v1/galleries/{gid}').json()['status'], 'ready')
        return gid

    def test_default_gpu_then_cpu_persists_across_restart_without_environment_changes(self):
        with patch.dict(os.environ, {'VEHICLE_DEVICE': 'cuda'}):
            with TestClient(self.app()) as client:
                settings = self.idle(client)
                self.assertEqual((settings['selected_device'], settings['active_device']), ('cuda', 'cuda'))
                self.assertEqual(self.calls, ['cuda'])
                self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
                self.assertEqual(self.idle(client)['active_device'], 'cpu')
                self.assertEqual(os.environ['VEHICLE_DEVICE'], 'cuda')
                self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
                self.assertEqual(self.calls, ['cuda', 'cpu'])
            with TestClient(self.app()) as client:
                self.assertEqual(self.idle(client)['selected_device'], 'cpu')
                self.assertEqual(self.calls, ['cuda', 'cpu', 'cpu'])
        self.assertEqual(json.loads((Path(self.temp.name) / 'prod/runtime-settings.json').read_text())['device'], 'cpu')

    def test_launch_device_and_saved_choice_priority(self):
        from vehicle.server.runtime_settings import read_device
        path = Path(self.temp.name) / 'device.json'
        for configured, expected in [('cpu', 'cpu'), ('cuda', 'cuda'), ('bad', 'cuda')]:
            with patch.dict(os.environ, {'VEHICLE_DEVICE': configured}):
                self.assertEqual(read_device(path), expected)
        path.write_text('{"device":"cuda"}')
        with patch.dict(os.environ, {'VEHICLE_DEVICE': 'cpu'}):
            self.assertEqual(read_device(path), 'cuda')
        with patch.dict(os.environ, {'VEHICLE_DEVICE': 'cpu'}):
            with TestClient(self.app()) as client:
                self.assertEqual(self.idle(client)['active_device'], 'cpu')
                self.assertEqual(self.calls, ['cpu'])

    def test_unavailable_gpu_keeps_settings_accessible_and_cpu_can_be_selected(self):
        options = available_devices()
        options[0].update(available=False, reason='Test CUDA unavailable')
        with TestClient(self.app(device_probe=lambda: options)) as client:
            state = self.idle(client)
            self.assertEqual(state['selected_device'], 'cuda')
            self.assertIsNone(state['active_device'])
            self.assertTrue(state['can_switch'])
            self.assertIn('CUDA unavailable', state['error'])
            self.assertFalse(client.get('/api/v1/status').json()['available'])
            self.assertEqual(self.calls, [])
            response = client.put('/api/v1/settings/device', json={'device': 'cuda'})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['error']['code'], 'DEVICE_UNAVAILABLE')
            self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
            self.assertEqual(self.idle(client)['active_device'], 'cpu')
            self.assertEqual(self.calls, ['cpu'])

    def test_failed_choice_without_active_model_rolls_back_persisted_selection(self):
        options = available_devices()
        options[0].update(available=False, reason='Test CUDA unavailable')
        def failed_factory(*, device):
            raise ValueError('Test CPU loader rejected model')
        with TestClient(self.app(device_probe=lambda: options, provider_factory=failed_factory)) as client:
            self.idle(client)
            self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
            state = self.idle(client)
            self.assertEqual(state['selected_device'], 'cuda')
            self.assertIsNone(state['active_device'])
            self.assertIn('CPU loader rejected', state['error'])
            self.assertEqual(json.loads((Path(self.temp.name) / 'prod/runtime-settings.json').read_text())['device'], 'cuda')

    def test_slow_loading_does_not_block_status_or_settings_and_failure_rolls_back(self):
        entered, release = threading.Event(), threading.Event()
        def factory(*, device):
            if device == 'cpu':
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test gate timed out')
                raise ValueError('Test model failed to load')
            return self.factory(device=device)
        with TestClient(self.app(provider_factory=factory)) as client:
            self.idle(client)
            old = client.app.state.services[0].provider
            try:
                response = client.put('/api/v1/settings/device', json={'device': 'cpu'})
                self.assertEqual(response.status_code, 202)
                self.assertTrue(entered.wait(2))
                start = time.monotonic()
                state = client.get('/api/v1/settings').json()
                status = client.get('/api/v1/status').json()
                self.assertLess(time.monotonic() - start, 1)
                self.assertEqual((state['selected_device'], state['active_device']), ('cpu', 'cuda'))
                self.assertTrue(state['switching'])
                self.assertFalse(state['can_switch'])
                self.assertFalse(status['available'])
                self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cuda'}).status_code, 409)
                self.assertEqual(client.post('/api/v1/galleries').status_code, 409)
            finally:
                release.set()
            state = self.idle(client)
            self.assertEqual((state['selected_device'], state['active_device']), ('cuda', 'cuda'))
            self.assertIn('failed to load', state['error'])
            self.assertIs(client.app.state.services[0].provider, old)
            self.assertEqual(json.loads((Path(self.temp.name) / 'prod/runtime-settings.json').read_text())['device'], 'cuda')

    def test_startup_load_is_background_and_wrong_device_is_rejected(self):
        entered, release = threading.Event(), threading.Event()
        def factory(*, device):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test gate timed out')
            return Provider('cpu')  # A loader's silent fallback must be rejected.
        with TestClient(self.app(provider_factory=factory)) as client:
            try:
                self.assertTrue(entered.wait(2))
                start = time.monotonic()
                state = client.get('/api/v1/settings').json()
                self.assertEqual(state['selected_device'], 'cuda')
                self.assertTrue(state['switching'])
                self.assertIsNone(state['active_device'])
                self.assertFalse(client.get('/api/v1/status').json()['available'])
                self.assertLess(time.monotonic() - start, 1)
            finally:
                release.set()
            state = self.idle(client)
            self.assertIsNone(state['active_device'])
            self.assertIn('другом устройстве', state['error'])

    def test_reindex_changes_vectors_but_preserves_completed_run_and_export_evidence(self):
        with TestClient(self.app()) as client:
            self.idle(client)
            gid = self.gallery(client)
            data, files = multipart(['query'], 'gallery_id', gid)
            response = client.post('/api/v1/runs', data=data, files=files)
            self.assertEqual(response.status_code, 202, response.text)
            self.idle(client)
            rid = response.json()['id']
            service = client.app.state.services[0]
            before_run = client.get(f'/api/v1/runs/{rid}').json()
            before_evidence = client.get(f'/api/v1/runs/{rid}/evidence').json()
            run_dir = service.runs[rid]['directory']
            before_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.iterdir()}
            old_vectors = np.array(service.galleries[gid]['vectors'])
            self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
            self.idle(client)
            gallery = service.galleries[gid]
            self.assertEqual(gallery['public']['status'], 'ready')
            self.assertEqual(gallery['public']['processed'], 10)
            self.assertEqual(gallery['model_sha256'], 'test-fingerprint-cpu')
            self.assertFalse(np.array_equal(old_vectors, gallery['vectors']))
            self.assertEqual(client.get(f'/api/v1/runs/{rid}').json(), before_run)
            self.assertEqual(client.get(f'/api/v1/runs/{rid}/evidence').json(), before_evidence)
            self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.iterdir()}, before_hashes)
        # Force a stale fingerprint across restart and exercise Windows mmap closure.
        def changed_factory(*, device):
            provider = self.factory(device=device)
            provider.inference_fingerprint += '-new-code'
            provider.model['inference_fingerprint'] = provider.inference_fingerprint
            return provider
        with TestClient(self.app(provider_factory=changed_factory)) as client:
            self.idle(client)
            gallery = client.app.state.services[0].galleries[gid]
            self.assertEqual(gallery['public']['status'], 'ready')
            self.assertEqual(gallery['model_sha256'], 'test-fingerprint-cpu-new-code')
            self.assertEqual(client.get(f'/api/v1/runs/{rid}/evidence').json(), before_evidence)

    def test_reindex_failure_is_explicit_and_never_searches_old_vectors(self):
        def factory(*, device):
            provider = self.factory(device=device)
            provider.fail = device == 'cpu'
            return provider
        with TestClient(self.app(provider_factory=factory)) as client:
            self.idle(client)
            gid = self.gallery(client)
            self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
            state = self.idle(client)
            self.assertEqual(state['active_device'], 'cpu')
            self.assertTrue(state['error'])
            self.assertEqual(client.get(f'/api/v1/galleries/{gid}').json()['status'], 'failed')
            self.assertIsNone(client.app.state.services[0].galleries[gid]['vectors'])
            data, files = multipart(['query'], 'gallery_id', gid)
            response = client.post('/api/v1/runs', data=data, files=files)
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['error']['code'], 'GALLERY_NOT_READY')

    def test_switch_is_blocked_by_running_queued_and_cancelling_jobs(self):
        with TestClient(self.app()) as client:
            self.idle(client)
            gid = self.gallery(client)
            provider = client.app.state.services[0].provider
            provider.entered, provider.release = threading.Event(), threading.Event()
            try:
                ids = []
                for query in ('first', 'second'):
                    data, files = multipart([query], 'gallery_id', gid)
                    response = client.post('/api/v1/runs', data=data, files=files)
                    self.assertEqual(response.status_code, 202, response.text)
                    ids.append(response.json()['id'])
                self.assertTrue(provider.entered.wait(2))
                self.assertFalse(client.get('/api/v1/settings').json()['can_switch'])
                for rid in ids:
                    client.post(f'/api/v1/runs/{rid}/cancel')
                response = client.put('/api/v1/settings/device', json={'device': 'cpu'})
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()['error']['code'], 'DEVICE_BUSY')
            finally:
                provider.release.set()
            self.idle(client)
            self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cpu'}).status_code, 202)
            self.idle(client)

    def test_multipart_upload_reserves_runtime_before_parsing_and_publication(self):
        entered, release = threading.Event(), threading.Event()
        original = Request.form
        @asynccontextmanager
        async def delayed_form(request, *args, **kwargs):
            entered.set()
            await asyncio.to_thread(release.wait, 5)
            async with original(request, *args, **kwargs) as form:
                yield form
        with TestClient(self.app()) as client, ThreadPoolExecutor(max_workers=1) as executor:
            self.idle(client)
            data, files = multipart([f'g{i}' for i in range(10)], 'name', 'Race gallery')
            with patch.object(Request, 'form', delayed_form):
                request = executor.submit(client.post, '/api/v1/galleries', data=data, files=files)
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(client.app.state.services[0].uploads, 1)
                    response = client.put('/api/v1/settings/device', json={'device': 'cpu'})
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(response.json()['error']['code'], 'DEVICE_BUSY')
                    self.assertFalse(client.get('/api/v1/settings').json()['can_switch'])
                finally:
                    release.set()
                response = request.result(timeout=5)
                self.assertEqual(response.status_code, 202, response.text)
            self.idle(client)
            service = client.app.state.services[0]
            self.assertEqual(service.uploads, 0)
            self.assertEqual(service.galleries[response.json()['id']]['model_sha256'], 'test-fingerprint-cuda')

    def test_invalid_input_and_injected_provider_compatibility(self):
        with TestClient(self.app()) as client:
            self.idle(client)
            for payload in ({}, {'device': 'gpu'}, {'device': None}):
                self.assertEqual(client.put('/api/v1/settings/device', json=payload).status_code, 422)
        with TestClient(create_app(Provider('cpu'), data_root=self.temp.name)) as client:
            self.assertTrue(client.get('/api/v1/status').json()['available'])
            self.assertFalse(client.get('/api/v1/settings').json()['can_switch'])
            self.assertEqual(client.put('/api/v1/settings/device', json={'device': 'cuda'}).status_code, 409)

    def test_production_cuda_rejection_never_silently_falls_back_to_cpu(self):
        options = available_devices()
        options[0].update(available=False, reason='CUDA is unavailable in test environment')
        with patch.dict(os.environ, {'VEHICLE_DEVICE': 'cpu'}, clear=True), \
                patch('vehicle.server.runtime_settings.device_options', return_value=options):
            provider = ProductionProvider(device='cuda')
        self.assertFalse(provider.available)
        self.assertIsNone(provider.model)
        self.assertEqual(provider.device, 'cuda')
        self.assertIn('CUDA is unavailable', provider.reason)


if __name__ == '__main__':
    unittest.main()
