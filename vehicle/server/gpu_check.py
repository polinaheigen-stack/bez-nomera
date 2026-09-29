"""Inspect the actual container/runtime. Requested CUDA never falls back to CPU."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import subprocess

from .service import atomic_json, now


@contextmanager
def selected_runtime(device, bundle=None):
    """CLI selection is authoritative, and child validation processes inherit it."""
    keys = ('VEHICLE_DEVICE', 'VEHICLE_MODEL_BUNDLE', 'VEHICLE_MODEL',
            'VEHICLE_MODEL_SHA256', 'VEHICLE_CALIBRATION', 'VEHICLE_CALIBRATION_SHA256')
    previous = {key: os.environ.get(key) for key in keys}
    os.environ['VEHICLE_DEVICE'] = str(device)
    if bundle is not None:
        os.environ['VEHICLE_MODEL_BUNDLE'] = str(Path(bundle).resolve())
        for key in keys[2:]:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def hardware_info(device='cuda'):
    import torch
    import psutil
    requested = torch.device(device)
    if requested.type not in ('cuda', 'cpu'):
        raise ValueError('Only cpu or cuda devices are supported.')
    available = torch.cuda.is_available()
    report = {'schema_version': 1, 'time': now(), 'status': 'inspected',
              'requested_device': str(requested), 'python': platform.python_version(),
              'platform': platform.platform(), 'processor': platform.processor(),
              'logical_cpus': os.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
              'torch': torch.__version__, 'torch_cuda_runtime': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'cuda_available': available,
              'cuda_device_count': torch.cuda.device_count() if available else 0,
              'torch_cuda_architectures': torch.cuda.get_arch_list() if available else [],
              'gpus': [], 'nvidia_smi': None, 'cuda_probe_passed': False,
              'model_load': None, 'submission_ready': False}
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report['gpus'].append({'index': index, 'name': props.name,
                                   'total_memory_bytes': props.total_memory,
                                   'compute_capability': [props.major, props.minor]})
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=10)
        report['nvidia_smi'] = {'returncode': result.returncode, 'stdout': result.stdout,
                                'stderr': result.stderr}
    except (OSError, subprocess.SubprocessError) as error:
        report['nvidia_smi'] = {'unavailable': str(error)}
    return report


def require_device(report):
    import torch
    device = torch.device(report['requested_device'])
    if device.type == 'cuda':
        if not report['cuda_available']:
            raise ValueError('CUDA was requested but is unavailable. Check the NVIDIA driver, Docker GPU access and CUDA-enabled PyTorch; CPU fallback is disabled.')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= report['cuda_device_count']:
            raise ValueError('Requested CUDA device index is unavailable.')
        probe = torch.ones(1, device=device) * 2
        torch.cuda.synchronize(device)
        if probe.item() != 2:
            raise ValueError('CUDA tensor execution failed.')
        report['cuda_probe_passed'] = True
        report['selected_gpu_index'] = index
        del probe
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', help='cuda, cuda:0, or cpu (diagnostics only)')
    parser.add_argument('--bundle', type=Path, help='Model bundle; otherwise VEHICLE_MODEL_BUNDLE is used')
    parser.add_argument('--load-model', action='store_true', help='Also strictly load the selected real model')
    parser.add_argument('--output', type=Path, help='New hardware.json; never overwrite existing evidence')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.exit(2, 'Use a new output file; existing hardware evidence is not overwritten.\n')
    report = {'schema_version': 1, 'status': 'failed', 'requested_device': args.device,
              'time': now(), 'submission_ready': False}
    try:
        report = hardware_info(args.device)
        require_device(report)
        if args.load_model:
            from .benchmark import load_provider_timed, weight_inventory
            with selected_runtime(args.device, args.bundle):
                provider, model_load = load_provider_timed()
                report['model_load'] = model_load
                if not provider.available:
                    raise ValueError(provider.reason or 'Model unavailable.')
                report['weights'] = weight_inventory(provider)
                report['model'] = provider.model
    except Exception as error:
        report.update(status='failed', error=str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
