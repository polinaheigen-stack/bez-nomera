import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vehicle.server.runtime_entrypoint import main


class RuntimeEntrypointTests(unittest.TestCase):
    def test_web_remains_reachable_without_gpu(self):
        args = ['entrypoint', 'python', '-m', 'uvicorn', 'vehicle.server.app:app']
        no_gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(sys, 'argv', args), patch.dict('os.environ', {'VEHICLE_DEVICE': 'cuda'}), \
                patch.dict(sys.modules, {'torch': no_gpu}), patch('os.execvp') as execute:
            main()
        execute.assert_called_once_with('python', args[1:])

    def test_cli_cuda_does_not_fall_back(self):
        args = ['entrypoint', 'python', '-m', 'vehicle.server.batch']
        no_gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(sys, 'argv', args), patch.dict('os.environ', {'VEHICLE_DEVICE': 'cuda'}), \
                patch.dict(sys.modules, {'torch': no_gpu}), patch('os.execvp') as execute:
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested but unavailable'):
                main()
        execute.assert_not_called()

    def test_cpu_cli_executes_requested_command(self):
        args = ['entrypoint', 'python', '-m', 'vehicle.server.batch']
        with patch.object(sys, 'argv', args), patch.dict('os.environ', {'VEHICLE_DEVICE': 'cpu'}), \
                patch('os.execvp') as execute:
            main()
        execute.assert_called_once_with('python', args[1:])
