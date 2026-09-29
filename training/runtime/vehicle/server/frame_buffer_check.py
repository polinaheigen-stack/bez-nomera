"""Complete-CSV A/B check of serial decoding versus one-frame prefetch.

This engineering measurement is separate from the official latency benchmark.
Every request contains one image; no model, preprocessing or ranking is changed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .batch import find_images, fingerprint, infer_images
from .benchmark import host_details, peak_rss_mb, weight_inventory
from .frame_buffer import FrameBuffer, pipeline_settings
from .provider import ProductionProvider, sha256
from .service import ServiceError, atomic_json, now, parse_csv


def identity_snapshot(provider):
    return deepcopy({
        'model': provider.model,
        'inference_fingerprint': fingerprint(provider),
        'calibration_sha256': getattr(provider, 'calibration_sha256', None),
        'threshold': provider.threshold,
        'retrieval': getattr(provider, 'retrieval', None),
        'retrieval_fingerprint': getattr(provider, 'retrieval_fingerprint', None),
        'runtime_setup': getattr(getattr(provider, 'predictor', None), 'runtime_setup', None),
        'declared_model_batch_size': getattr(getattr(provider, 'predictor', None), 'model_batch_size', None),
    })


def complete_pass(provider, items, *, ahead, synchronize=lambda: None,
                  cancelled=None, clock=None):
    """Time enter/read/infer/hash/close/drain and final device completion.

    Partial work never returns a throughput sample. The callback is also useful
    for callers which cancel an engineering run; cleanup drains the reader.
    """
    if not items:
        raise ValueError('A complete pass needs at least one input.')
    clock = clock or time.perf_counter
    initial = identity_snapshot(provider)
    chunks, records = [], []
    synchronize()  # Previous device work is outside this pass.
    started = clock()
    try:
        with FrameBuffer(items, ahead=ahead, cancelled=cancelled, verify_hash=True) as frames:
            for frame in frames:
                if not provider.available or identity_snapshot(provider) != initial:
                    raise ValueError('Model, calibration or retrieval changed during the pass.')
                vectors, inputs = infer_images(provider, [frame.row], [frame.require_image()])
                if vectors.shape != (1, provider.model['dimension']) or len(inputs) != 1:
                    raise ValueError('Model returned an incomplete batch-1 result.')
                frame.verify_unchanged()
                chunks.append(vectors)
                records.append({**inputs[0], 'sha256': frame.sha256})
        values = np.concatenate(chunks) if chunks else np.empty((0, provider.model['dimension']), np.float32)
    finally:
        # Runs after context-manager drain, including on errors/cancellation.
        synchronize()
    elapsed = clock() - started
    if len(records) != len(items) or (cancelled is not None and cancelled()):
        raise RuntimeError('Cancelled or incomplete pass; no throughput sample is valid.')
    expected = [(row['image_id'], list(row['bbox'])) for row in items]
    if [(row['image_id'], row['bbox']) for row in records] != expected:
        raise ValueError('Input order or BBox changed during the pass.')
    if not np.isfinite(values).all() or elapsed <= 0 or not np.isfinite(elapsed):
        raise ValueError('Invalid embeddings or complete-pass duration.')
    if not provider.available or identity_snapshot(provider) != initial:
        raise ValueError('Model, calibration or retrieval changed during the pass.')
    sample = {'prefetch': ahead, 'images': len(items), 'duration_seconds': elapsed,
              'images_per_second': len(items) / elapsed,
              'vectors_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest()}
    return values, records, sample


def _exact(reference, actual, expected_records, actual_records):
    if not np.array_equal(reference, actual) or expected_records != actual_records:
        raise ValueError('Exact embedding/order parity failed against the serial baseline.')


def _source_snapshot(input_csv, items):
    return {'input_csv_sha256': sha256(input_csv),
            'images': {row['image_id']: sha256(row['path']) for row in items}}


def run_check(images, input_csv, output, *, repeats=3, warmup=5,
              allow_cpu=False, provider=None):
    """Use a loaded provider in tests; the CLI constructs ProductionProvider once.

    A new report path is reserved exclusively before loading or touching inputs.
    Successful summaries require exact parity, complete counts and stable inputs.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'status': 'running', 'measured': False,
              'exact_parity': False, 'official_measurement': False, 'time': now(),
              'passes': [], 'summary': None, 'order_probe': None,
              'weights': None, 'total_weights_bytes': None, 'model_load': None}
    try:
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream)
    except FileExistsError:
        raise ValueError('Use a new output path; existing evidence is not overwritten.') from None
    owned_provider = provider is None
    torch = device = None
    gpu = False
    cuda_memory_ready = False
    try:
        if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
            raise ValueError('Repeats must be positive and warmup nonnegative integers.')
        import torch
        if owned_provider:
            tick = time.perf_counter()
            provider = ProductionProvider()
            if provider.available and str(provider.model.get('device', '')).startswith('cuda'):
                torch.cuda.synchronize(torch.device(provider.model['device']))
            report['model_load'] = {'seconds': time.perf_counter() - tick,
                                    'scope': 'constructor through device completion; excluded from A/B timing'}
        if not provider.available:
            raise ValueError(provider.reason or 'Model unavailable.')
        device = torch.device(provider.model.get('device', 'cpu'))
        gpu = device.type == 'cuda'
        if not gpu and not allow_cpu:
            raise ValueError('CPU engineering checks require --allow-cpu; GPU metrics are not substituted.')
        if gpu and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable.')
        initial = identity_snapshot(provider)
        if initial['declared_model_batch_size'] not in (None, 1):
            raise ValueError('This check requires actual model batch size 1.')
        report.update(scope='local_gpu_complete_pass' if gpu else 'cpu_engineering_only',
                      identity_before=initial, runtime_setup=initial['runtime_setup'],
                      model_batch_size=1, requested_batch_size=1,
                      hardware={**host_details(gpu), 'platform': platform.platform(),
                                'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
                                'python': platform.python_version(), 'torch': torch.__version__,
                                'cuda_runtime': torch.version.cuda, 'device': str(device),
                                'gpu': torch.cuda.get_device_name(device) if gpu else None})
        def sync():
            if gpu:
                torch.cuda.synchronize(device)
        csv_bytes = Path(input_csv).read_bytes()
        rows = parse_csv(csv_bytes, max_images=1_000_000)
        paths = find_images(images, rows)
        items = [{**row, 'path': paths[row['image_id']]} for row in rows]
        sources = _source_snapshot(input_csv, items)
        if sources['input_csv_sha256'] != hashlib.sha256(csv_bytes).hexdigest():
            raise ValueError('Input CSV changed while preparing the check.')
        report.update(input_count=len(items), input_order=[row['image_id'] for row in rows],
                      inputs_before=sources, pipeline=[pipeline_settings(0), pipeline_settings(1)])
        report['protocol'] = {
            'repeats_per_mode': repeats, 'warmup_images': warmup,
            'mode_order': [[0, 1] if repeat % 2 == 0 else [1, 0] for repeat in range(repeats)],
            'verify_hash': True, 'cuda_synchronized': gpu,
            'includes': ['buffer_startup', 'file_open', 'original_pixel_decode', 'input_sha256_before_and_after_each_image',
                         'bbox_crop', 'model_preprocessing', 'batch1_forward', 'transfer_to_cpu',
                         'normalization', 'result_assembly', 'frame_close', 'reader_drain', 'final_cuda_completion'],
            'excludes': ['model_load', 'warmup', 'input_inventory', 'weight_inventory',
                         'parity_comparison', 'gallery_search', 'report_write'],
            'os_file_cache': 'not cleared; alternating complete passes; no application embedding cache',
            'timing_semantics': 'finite complete-CSV wall time; includes startup/drain, not steady-state FPS',
        }
        report['weights'] = weight_inventory(provider)
        report['total_weights_bytes'] = report['weights']['total_bytes']
        if report['weights']['status'] != 'verified':
            raise ValueError('Runtime weight inventory must be verified before comparison.')
        if gpu:
            torch.cuda.reset_peak_memory_stats(device)
            cuda_memory_ready = True
        if warmup:
            complete_pass(provider, [items[i % len(items)] for i in range(warmup)], ahead=0, synchronize=sync)
        baseline = baseline_records = None
        for repeat, modes in enumerate(report['protocol']['mode_order']):
            for ahead in modes:
                values, records, sample = complete_pass(provider, items, ahead=ahead, synchronize=sync)
                if baseline is None:
                    baseline, baseline_records = values, records
                    report['baseline'] = {'prefetch': 0, 'repeat': 0, 'vectors_sha256': sample['vectors_sha256'],
                                          'records': baseline_records}
                _exact(baseline, values, baseline_records, records)
                report['passes'].append({**sample, 'repeat': repeat, 'exact_parity': True})
        # The probe changes ordering and repeats IDs intentionally, without
        # relaxing the unique-ID CSV contract or rerunning the entire dataset.
        indices = list(reversed(range(min(8, len(items))))) + [0, min(1, len(items) - 1), 0]
        probe_samples = []
        for ahead in (0, 1):
            values, records, sample = complete_pass(provider, [items[i] for i in indices], ahead=ahead, synchronize=sync)
            _exact(baseline[indices], values, [baseline_records[i] for i in indices], records)
            probe_samples.append({**sample, 'exact_parity': True})
        report['order_probe'] = {'baseline_indices': indices, 'passes': probe_samples,
                                 'scope': 'bounded reverse-and-duplicate parity only; excluded from full-CSV summary'}
        report['identity_after'] = identity_snapshot(provider)
        report['inputs_after'] = _source_snapshot(input_csv, items)
        report['weights_after'] = weight_inventory(provider)
        if report['identity_after'] != initial or report['inputs_after'] != sources or report['weights_after'] != report['weights']:
            raise ValueError('Model, calibration, retrieval, weights or inputs changed during the A/B check.')
        modes = {}
        for ahead in (0, 1):
            samples = [sample for sample in report['passes'] if sample['prefetch'] == ahead]
            if len(samples) != repeats or any(sample['images'] != len(items) for sample in samples):
                raise ValueError('Incomplete full-CSV mode counts.')
            seconds = sum(sample['duration_seconds'] for sample in samples)
            count = sum(sample['images'] for sample in samples)
            modes[str(ahead)] = {'passes': len(samples), 'images': count, 'duration_seconds': seconds,
                                  'complete_pass_images_per_second': count / seconds,
                                  'median_pass_seconds': float(np.median([sample['duration_seconds'] for sample in samples]))}
        report.update(status='completed', measured=True, exact_parity=True,
                      summary={'modes': modes, 'prefetch_vs_serial_throughput_ratio':
                               modes['1']['complete_pass_images_per_second'] / modes['0']['complete_pass_images_per_second'],
                               'note': 'Ratio of completed finite-pass rates on this host; not official latency or steady-state FPS.'})
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                      measured=False, exact_parity=False, summary=None,
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['memory'] = {'peak_process_ram_mb': peak_rss_mb(),
                            'peak_torch_vram_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2 if cuda_memory_ready else None,
                            'peak_torch_vram_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2 if cuda_memory_ready else None,
                            'note': 'RAM is lifetime process peak; PyTorch VRAM covers both modes/warmup/probe, not driver-total memory.'}
        atomic_json(output, report)
        if owned_provider and provider is not None:
            close = getattr(provider, 'close', None)
            if callable(close):
                close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--input-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New comparison JSON path')
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    try:
        report = run_check(args.images, args.input_csv, args.output, repeats=args.repeats,
                           warmup=args.warmup, allow_cpu=args.allow_cpu)
    except (OSError, ValueError, RuntimeError, ServiceError) as error:
        parser.exit(2, f'Frame buffer check failed: {error}\n')
    print(json.dumps({'status': report['status'], 'exact_parity': report['exact_parity'],
                      'scope': report['scope'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
