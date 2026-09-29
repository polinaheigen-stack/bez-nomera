"""Measure complete image decode/crop/inference; CUDA synchronization brackets every sample.

Default protocol: 50 warmup calls, 300 batch-1 latency samples, at least 10 seconds
of throughput for each batch size 1, 8, 16, 32. CPU requires explicit --allow-cpu.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .batch import find_images, fingerprint, infer_batch
from .provider import ProductionProvider, sha256
from .service import atomic_json, now, parse_csv, ServiceError


def weight_inventory(provider):
    """Verify and count all checkpoints bound to the loaded provider."""
    from .model_bundle import MAX_WEIGHTS_BYTES, inspect_bundle, runtime_fingerprint
    result = {'status': 'unavailable', 'total_bytes': None, 'limit_bytes': MAX_WEIGHTS_BYTES,
              'files': [], 'scope': 'All runtime weights of the selected supported provider; excludes config, calibration, code and training-only assets.'}
    if not provider.available:
        result['reason'] = provider.reason or 'Model unavailable.'
        return result
    # The P4 trial owns a self-describing student checkpoint rather than an E25
    # serving bundle. Only its concrete reviewed adapter may supply this inventory.
    if type(provider).__module__ == 'vehicle.server.student_model':
        from .student_model import StudentProvider
        if isinstance(provider, StudentProvider):
            return provider.weight_inventory()
    if type(provider).__module__ == 'vehicle.server.compact_model':
        from .compact_model import CompactProvider
        if isinstance(provider, CompactProvider):
            return provider.weight_inventory()
    bundle_path = os.getenv('VEHICLE_MODEL_BUNDLE')
    legacy_path = os.getenv('VEHICLE_MODEL')
    if getattr(provider, '_bundle_mode', False):
        if not bundle_path or legacy_path:
            raise ValueError('The active bundle source is missing or ambiguous; weight inventory cannot be verified.')
        # This also rejects any additional recognized weight files: the current
        # adapters load the full backbone and all runtime heads from one checkpoint.
        bundle = inspect_bundle(bundle_path)
        if runtime_fingerprint(bundle, provider.model['device']) != fingerprint(provider):
            raise ValueError('Bundle on disk differs from the loaded model; weight inventory rejected.')
        members = bundle.member_bundles or (bundle,)
        inventory = []
        for member in members:
            selected = member.weights_path
            before = selected.stat().st_size
            actual_hash = sha256(selected)
            if actual_hash != member.weights_sha256 or selected.stat().st_size != before:
                raise ValueError('Runtime member checkpoint changed or has an invalid hash.')
            if before <= 0:
                raise ValueError('Empty runtime checkpoint.')
            inventory.append({'file': selected.relative_to(bundle.root).as_posix(),
                              'bytes': before, 'sha256': actual_hash,
                              'role': 'complete_runtime_checkpoint'})
        total = sum(item['bytes'] for item in inventory)
        if total > MAX_WEIGHTS_BYTES or bundle.weights_sha256 != provider.model['sha256']:
            raise ValueError('Runtime weight size or composite model hash is invalid.')
        result['source'] = 'selected_bundle'
        result.update(status='verified', total_bytes=total, files=inventory,
                      model_sha256=bundle.weights_sha256,
                      note='Every complete member checkpoint, including its runtime heads, is counted; no weights are downloaded at runtime.')
        return result
    elif legacy_path and not bundle_path:
        selected = Path(legacy_path).resolve()
        filename = str(selected)
        expected = os.getenv('VEHICLE_MODEL_SHA256', '').lower()
        result['source'] = 'legacy_model'
    else:
        result['reason'] = 'The loaded provider has no verifiable selected artifact source; total size is unknown.'
        return result
    if expected != (provider.model or {}).get('sha256'):
        raise ValueError('Selected weight hash does not match the loaded model.')
    before = selected.stat().st_size
    actual_hash = sha256(selected)
    if actual_hash != expected or selected.stat().st_size != before:
        raise ValueError('Selected weight file changed or differs from the loaded model.')
    if not 0 < before <= MAX_WEIGHTS_BYTES:
        raise ValueError('Selected runtime weights exceed the permitted size or are empty.')
    result.update(status='verified', total_bytes=before,
                  files=[{'file': filename, 'bytes': before, 'sha256': actual_hash,
                          'role': 'complete_runtime_checkpoint'}],
                  note='Supported loaders use one complete checkpoint, including any auxiliary runtime heads; no separate auxiliary weights are loaded.')
    return result


def host_details(gpu):
    import psutil
    result = {'total_ram_mb': psutil.virtual_memory().total / 1024**2,
              'physical_cpus': psutil.cpu_count(logical=False), 'nvidia_driver_versions': None}
    if gpu:
        try:
            query = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10, check=True)
            versions = sorted(set(line.strip() for line in query.stdout.splitlines() if line.strip()))
            result['nvidia_driver_versions'] = versions or None
        except (OSError, subprocess.SubprocessError):
            result['nvidia_driver_note'] = 'Driver version unavailable; attach nvidia-smi output separately.'
    return result


def peak_rss_mb():
    """Process-lifetime peak resident memory, including model loading."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / 1024**2
    import resource
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum / (1024**2 if sys.platform == 'darwin' else 1024)


def load_provider_timed():
    """Measure construction separately, including any adapter startup preparation."""
    import torch
    requested = torch.device(os.getenv('VEHICLE_DEVICE', 'cpu'))
    pre_sync = requested.type == 'cuda' and torch.cuda.is_initialized()
    if pre_sync:
        torch.cuda.synchronize(requested)
    record = {'status': 'running', 'seconds': None, 'cuda_synchronized': False,
              'preexisting_cuda_work_drained': pre_sync, 'device_requested': str(requested),
              'scope': 'ProductionProvider constructor through device completion; one fresh process, not guaranteed cold disk cache',
              'includes': ['artifact_hashes', 'provenance_and_config_validation', 'checkpoint_read',
                           'model_construction', 'device_transfer', 'lazy_imports_inside_constructor', 'cuda_initialization_if_needed'],
              'excludes': ['python_process_start', 'imports_before_constructor', 'preexisting_cuda_work', 'forward', 'warmup'],
              'model': None, 'inference_fingerprint': None}
    start = time.perf_counter()
    provider = None
    try:
        provider = ProductionProvider()
        if provider.available:
            setup = getattr(getattr(provider, 'predictor', None), 'runtime_setup', None)
            if isinstance(setup, dict):
                record['runtime_setup'] = deepcopy(setup)
                record['runtime_mode'] = setup.get('mode')
                record['includes'].append('adapter_runtime_preparation')
                record['scope'] = ('ProductionProvider constructor including declared adapter startup preparation '
                                   'through device completion; one fresh process, not guaranteed cold disk cache')
                if setup.get('forward_executed', bool(setup.get('startup_validation_calls', 0))):
                    record['includes'].append('adapter_startup_validation_forward')
                    record['excludes'].remove('forward')
                if setup.get('warmup_executed', bool(setup.get('graph_warmup_calls_per_member', 0))):
                    record['includes'].append('adapter_graph_warmup')
                    record['excludes'].remove('warmup')
                if setup.get('capture_executed', bool(setup.get('graph_count', 0))):
                    record['includes'].append('cuda_graph_capture')
                if setup.get('mode') == 'compiled_overlap' and setup.get('startup_validation_calls', 0):
                    record['includes'].append('adapter_lazy_compilation')
                record['excludes'].extend(['user_request_inference', 'benchmark_warmup'])
            device = torch.device(provider.model['device'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                record['cuda_synchronized'] = True
            record.update(status='completed', model=provider.model,
                          inference_fingerprint=fingerprint(provider),
                          calibration_sha256=provider.calibration_sha256)
        else:
            record.update(status='unavailable', error=provider.reason or 'Model unavailable.')
        return provider, record
    except Exception as error:
        record.update(status='failed', error=str(error))
        close = getattr(provider, 'close', None)
        if callable(close):
            record['runtime_cleanup_after_load_failure'] = {'attempted': True, 'completed': False}
            try:
                close()
                record['runtime_cleanup_after_load_failure']['completed'] = True
            except Exception as cleanup_error:
                record['runtime_cleanup_after_load_failure']['error'] = str(cleanup_error)
        # Keep the partially measured phase available to the outer evidence writer.
        error.model_load = record
        raise
    finally:
        record['seconds'] = time.perf_counter() - start


def measure(provider, images, input_csv, output, *, allow_cpu=False, model_load=None):
    if not provider.available:
        raise ValueError(provider.reason or 'Model unavailable.')
    import torch
    device = torch.device((provider.model or {}).get('device', 'cpu'))
    gpu = device.type == 'cuda'
    if not gpu and not allow_cpu:
        raise ValueError('GPU measurement requires VEHICLE_DEVICE=cuda. CPU diagnostics require --allow-cpu.')
    if gpu and not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')
    images, input_csv, output = Path(images), Path(input_csv), Path(output)
    event_path = output.with_suffix('.events.jsonl')
    if output.exists() or event_path.exists():
        raise ValueError('Use a new output path; existing benchmark evidence is not overwritten.')
    csv_bytes = input_csv.read_bytes()
    rows = parse_csv(csv_bytes, max_images=1_000_000)
    paths = find_images(images, rows)
    identity = fingerprint(provider)
    input_hash = hashlib.sha256(csv_bytes).hexdigest()
    records = [{**row, 'sha256': sha256(paths[row['image_id']])} for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    def event(name, **details):
        with event_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': now(), 'event': name, **details}, ensure_ascii=False, allow_nan=False) + '\n')
    def sync():
        if gpu:
            torch.cuda.synchronize(device)
    cursor = 0
    def call(batch_size):
        nonlocal cursor
        selected = [rows[(cursor + offset) % len(rows)] for offset in range(batch_size)]
        cursor = (cursor + batch_size) % len(rows)
        sync()
        start = time.perf_counter()
        vectors, _ = infer_batch(provider, selected, paths)
        sync()
        elapsed = time.perf_counter() - start
        if vectors.shape[0] != batch_size:
            raise ValueError('Benchmark inference returned incomplete embeddings.')
        return elapsed
    report = {'schema_version': 1, 'measured': False, 'status': 'running', 'official_measurement': False,
              'scope': 'local_gpu_measurement' if gpu else 'cpu_engineering_only', 'model': provider.model,
              'model_batch_size': getattr(getattr(provider, 'predictor', None), 'model_batch_size', None),
              'runtime_setup': deepcopy(getattr(getattr(provider, 'predictor', None), 'runtime_setup', None)),
              'batch_size_note': 'Throughput batch_size is the external request group; model_batch_size is the actual neural forward size when fixed by the adapter.',
              'inference_fingerprint': identity, 'calibration_sha256': getattr(provider, 'calibration_sha256', None),
              'hardware': {**host_details(gpu), 'platform': platform.platform(), 'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
                           'python': platform.python_version(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                           'device': str(device), 'gpu': torch.cuda.get_device_name(device) if gpu else None,
                           'gpu_total_memory_mb': torch.cuda.get_device_properties(device).total_memory / 1024**2 if gpu else None},
              'input_csv_sha256': input_hash, 'inputs': records,
              'protocol': {'warmup_calls': 50, 'latency_samples': 300, 'latency_batch_size': 1, 'throughput_batches': [1, 8, 16, 32],
                           'throughput_min_seconds': 10, 'includes': ['file_open', 'decode_original_pixels', 'bbox_crop', 'model_preprocessing', 'forward', 'transfer_to_cpu', 'embedding_normalization'],
                           'excludes': ['model_load', 'gallery_search', 'input_hashing', 'weight_inventory', 'report_write'],
                           'os_file_cache': 'not cleared; images cycled in CSV order; no application image or embedding cache', 'cuda_synchronized': gpu},
              'model_load': model_load, 'weights': None, 'total_weights_bytes': None,
              'latency': None, 'throughput': [],
              'notes': ['GPU and CPU values are never interchanged.', 'Local measurements are not an organizer-issued performance score.',
                        'model_load is a separate initialization measurement, never added to per-image latency.']}
    event('benchmark_started', scope=report['scope'])
    try:
        report['weights'] = weight_inventory(provider)
        report['total_weights_bytes'] = report['weights']['total_bytes']
        event('weight_inventory', **report['weights'])
        if gpu:
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(50):
            call(1)
        event('warmup_completed', calls=50)
        samples = [call(1) * 1000 for _ in range(300)]
        report['latency'] = {'median_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95)),
                             'mean_ms': float(np.mean(samples)), 'min_ms': min(samples), 'max_ms': max(samples), 'samples_ms': samples}
        event('latency_completed', median_ms=report['latency']['median_ms'], samples=300)
        for batch_size in (1, 8, 16, 32):
            # First call prepares kernels for this batch size; excluded from measurement.
            call(batch_size)
            calls, elapsed = 0, 0.
            sync()
            start = time.perf_counter()
            while elapsed < 10:
                call(batch_size)
                calls += 1
                elapsed = time.perf_counter() - start
            item = {'batch_size': batch_size, 'duration_seconds': elapsed, 'images': calls * batch_size,
                    'images_per_second': calls * batch_size / elapsed}
            report['throughput'].append(item)
            event('throughput_completed', **item)
        if fingerprint(provider) != identity or sha256(input_csv) != input_hash or any(sha256(paths[row['image_id']]) != row['sha256'] for row in records):
            raise ValueError('Model or input data changed during benchmark.')
        if weight_inventory(provider) != report['weights']:
            raise ValueError('Runtime weight inventory changed during benchmark.')
        report.update(measured=True, status='completed')
        event('benchmark_completed')
    except Exception as error:
        report.update(status='failed', error=str(error))
        if report['weights'] is None:
            report['weights'] = {'status': 'failed', 'total_bytes': None, 'error': str(error)}
        event('benchmark_failed', error=str(error))
        raise
    finally:
        report['memory'] = {'peak_process_ram_mb': peak_rss_mb(),
                            'peak_torch_vram_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2 if gpu else None,
                            'peak_torch_vram_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2 if gpu else None,
                            'note': 'RAM is process-lifetime RSS; VRAM is PyTorch peak, not driver-total memory.'}
        atomic_json(output, report)
    return report


def run_benchmark(images, input_csv, output, *, allow_cpu=False):
    """Keep unavailable-model and input failures, without fabricating latency."""
    output = Path(output)
    if output.exists() or output.with_suffix('.events.jsonl').exists():
        raise ValueError('Use a new output path; existing benchmark evidence is not overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    model_load = None
    try:
        provider, model_load = load_provider_timed()
        return measure(provider, images, input_csv, output, allow_cpu=allow_cpu, model_load=model_load)
    except Exception as error:
        if not output.exists():
            atomic_json(output, {'schema_version': 1, 'status': 'failed', 'measured': False,
                                 'official_measurement': False, 'error': str(error),
                                 'model_load': getattr(error, 'model_load', model_load),
                                 'weights': {'status': 'unavailable', 'total_bytes': None,
                                             'reason': 'Benchmark did not reach a verified weight inventory.'},
                                 'total_weights_bytes': None,
                                 'latency': None, 'throughput': [], 'time': now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New performance.json path')
    parser.add_argument('--allow-cpu', action='store_true', help='Explicit diagnostic run; not GPU evidence')
    args = parser.parse_args()
    try:
        report = run_benchmark(args.images, args.input_csv, args.output, allow_cpu=args.allow_cpu)
    except (OSError, ValueError, RuntimeError, ServiceError) as error:
        parser.exit(2, f'Benchmark failed: {error}\n')
    print(json.dumps({'status': report['status'], 'scope': report['scope'], 'median_ms': report['latency']['median_ms'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
