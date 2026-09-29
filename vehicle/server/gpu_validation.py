"""Reproducible real-model acceptance run on a separate GPU computer.

Reuses the production batch, unchanged official evaluator, process repeatability
and full-cycle benchmark. A successful CPU smoke run never becomes GPU evidence.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .batch import decoded_images, find_images, infer_batch, run_batch
from .benchmark import load_provider_timed, weight_inventory
from .evaluation import evaluate_export, json_finite, validate_ground_truth
from .gpu_check import hardware_info, require_device, selected_runtime
from .provider import sha256
from .repeatability import server_hashes
from .service import atomic_json, now, parse_csv
from .validation import rank_vectors


def spaced_indices(count, limit):
    if count < 1 or limit < 1:
        raise ValueError('Nonempty inputs and positive sample size required.')
    return np.linspace(0, count - 1, min(count, limit), dtype=int).tolist()


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
        for row in rows:
            writer.writerow([row['image_id'], *row['bbox']])


def small_inputs(query_csv, gallery_csv, output):
    """Deterministic real subset; IDs, bounding boxes and image pixels unchanged."""
    query = parse_csv(Path(query_csv).read_bytes(), max_images=1_000_000)
    gallery = parse_csv(Path(gallery_csv).read_bytes(), max_images=1_000_000)
    if len(gallery) < 10:
        raise ValueError('At least ten gallery images are required.')
    qi, gi = spaced_indices(len(query), 2), spaced_indices(len(gallery), 12)
    output = Path(output)
    output.mkdir()
    qpath, gpath = output / 'query.csv', output / 'gallery.csv'
    write_rows(qpath, [query[i] for i in qi])
    write_rows(gpath, [gallery[i] for i in gi])
    return qpath, gpath, qi, gi


def check_independence(provider, images, query_csv, gallery_csv):
    """Compare single, reversed and repeated inputs, including changed opaque IDs."""
    queries = parse_csv(Path(query_csv).read_bytes(), max_images=1_000_000)
    gallery = parse_csv(Path(gallery_csv).read_bytes(), max_images=1_000_000)
    selected = [queries[i] for i in spaced_indices(len(queries), 2)]
    gallery = [gallery[i] for i in spaced_indices(len(gallery), 12)]
    paths = find_images(images, selected + gallery)
    singles = np.concatenate([infer_batch(provider, [row], paths)[0] for row in selected])
    shuffled = list(reversed(selected)) + [gallery[0], selected[0]]
    mixed, _ = infer_batch(provider, shuffled, paths)
    recovered = mixed[:len(selected)][::-1]
    with decoded_images(selected, paths) as pictures:
        renamed = np.asarray(provider.embed_batch(pictures, [r['bbox'] for r in selected],
                                                  image_ids=[f'opaque-{i}' for i in range(len(selected))]), dtype=np.float32)
    gallery_vectors, _ = infer_batch(provider, gallery, paths)
    # CUDA convolution/attention kernels can use a different reduction path
    # for batch=1 and batch>1. Keep this predeclared, conservative numerical
    # gate for the new ensemble; do not relax it after observing failures.
    # BF16 portability for E25 has not yet been measured on the receiving GPU.
    # Rankings and refusal decisions must match regardless of vector tolerance.
    device = getattr(provider, 'device', None)
    if device is None:
        device = getattr(getattr(provider, 'model', {}), 'get', lambda *_: None)('device')
    device_type = getattr(device, 'type', str(device).split(':', 1)[0])
    tolerance = {'atol': 2e-4 if device_type == 'cuda' else 2e-5, 'rtol': 1e-4}
    checks = {'single_vs_reversed_batch': bool(np.allclose(singles, recovered, **tolerance)),
              'duplicate_independence': bool(np.allclose(singles[0], mixed[-1], **tolerance)),
              'opaque_id_independence': bool(np.allclose(singles, renamed, **tolerance))}
    rankings, decisions = [], []
    for single, batched in zip(singles, recovered):
        _, first, accepted_first = rank_vectors(single, gallery_vectors, provider.threshold, getattr(provider, 'retrieval', None))
        _, second, accepted_second = rank_vectors(batched, gallery_vectors, provider.threshold, getattr(provider, 'retrieval', None))
        rankings.append(bool(np.array_equal(first[:10], second[:10])))
        decisions.append(accepted_first == accepted_second)
    checks.update(top10_independence=all(rankings), accepted_pairs_independence=all(decisions))
    return {'passed': all(checks.values()), 'checks': checks, 'tolerance': tolerance,
            'query_count': len(selected), 'gallery_count': len(gallery),
            'max_single_batch_abs_error': float(np.max(np.abs(singles - recovered))),
            'images': [{'image_id': r['image_id'], 'sha256': sha256(paths[r['image_id']])}
                       for r in selected + gallery],
            'scope': 'small real subset, no camera or ground-truth labels supplied to inference',
            'tolerance_scope': 'Predeclared conservative portability gate; not a claimed bound for all GPU kernels or E25 BF16.'}


def reference_comparison(reference_dir, actual_dir, query_count, indices=None, *, threshold=None, retrieval=None):
    """Vector transfer tolerance is fixed; ranking differences remain explicit."""
    reference_dir, actual_dir = Path(reference_dir), Path(actual_dir)
    source = np.load(reference_dir / 'embeddings.npy', allow_pickle=False)
    actual = np.load(actual_dir / 'embeddings.npy', allow_pickle=False)
    if indices is not None:
        qi, gi, source_query_count = indices
        source = source[qi + [source_query_count + i for i in gi]]
    if (source.ndim != 2 or actual.ndim != 2 or source.shape != actual.shape
            or source.shape[1] < 1 or type(query_count) is not int
            or not 0 < query_count < len(source)):
        raise ValueError('Reference embeddings are invalid or their order/shape differs from this run.')
    for name, values in (('Reference', source), ('Actual', actual)):
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f'{name} embeddings must be finite float32 descriptors.')
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
            raise ValueError(f'{name} embeddings must be L2 normalized.')
    minimum_cosine = float(np.min(np.sum(source * actual, axis=1)))
    maximum_error = float(np.max(np.abs(source - actual)))
    expected_scores = source[:query_count] @ source[query_count:].T
    actual_scores = actual[:query_count] @ actual[query_count:].T
    expected_ranks = np.argsort(-expected_scores, axis=1, kind='stable')[:, :10]
    actual_ranks = np.argsort(-actual_scores, axis=1, kind='stable')[:, :10]
    changed = np.flatnonzero(np.any(expected_ranks != actual_ranks, axis=1)).tolist()
    result = {'passed': minimum_cosine >= .9999 and maximum_error <= .002,
            'scope': 'embedding transfer; exact ranking equality is separately reported, not implied',
            'tolerance': {'minimum_cosine': .9999, 'maximum_abs_error': .002},
            'minimum_cosine': minimum_cosine, 'maximum_abs_error': maximum_error,
            'embedding_cosine_top10_exact': not changed,
            'embedding_cosine_top10_changed_query_indices': changed,
            'serving_policy_top10_exact': None, 'serving_policy_submission_exact': None,
            'historical_submission_exact': None, 'accepted_pairs_exact': None,
            'query_refusals_exact': None,
            'reference_embeddings_sha256': sha256(reference_dir / 'embeddings.npy'),
            'actual_embeddings_sha256': sha256(actual_dir / 'embeddings.npy')}
    reference_policy, actual_policy, reference_accept, actual_accept = [], [], [], []
    for expected, measured in zip(source[:query_count], actual[:query_count]):
        _, expected_order, expected_accepted = rank_vectors(expected, source[query_count:], threshold if threshold is not None else 0, retrieval)
        _, actual_order, actual_accepted = rank_vectors(measured, actual[query_count:], threshold if threshold is not None else 0, retrieval)
        reference_policy.append(expected_order[:10].tolist())
        actual_policy.append(actual_order[:10].tolist())
        reference_accept.append(expected_accepted)
        actual_accept.append(actual_accepted)
    changed_policy = [i for i, (a, b) in enumerate(zip(reference_policy, actual_policy)) if a != b]
    result.update(serving_policy_top10_exact=not changed_policy,
                  serving_policy_top10_changed_query_indices=changed_policy)
    if threshold is not None:
        changed_decisions = [i for i, (a, b) in enumerate(zip(reference_accept, actual_accept)) if set(a) != set(b)]
        result.update(threshold_service=threshold, accepted_pairs_exact=not changed_decisions,
                      accepted_pairs_changed_query_indices=changed_decisions,
                      query_refusals_exact=[bool(a) for a in reference_accept] == [bool(a) for a in actual_accept])
    def submission_rows(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.reader(stream))
    actual_submission = actual_dir / 'submission.csv'
    order_file = actual_dir / 'embedding_order.json'
    if actual_submission.is_file() and order_file.is_file():
        order = json.loads(order_file.read_text(encoding='utf-8'))
        if len(order['query']) != query_count or len(order['gallery']) != len(source) - query_count:
            raise ValueError('Actual embedding order differs from reference comparison dimensions.')
        expected_submission = [[qid, *[order['gallery'][i] for i in ranked]]
                               for qid, ranked in zip(order['query'], reference_policy)]
        result['serving_policy_submission_exact'] = submission_rows(actual_submission) == expected_submission
    historical_submission = reference_dir / 'submission.csv'
    if indices is None and historical_submission.is_file() and actual_submission.is_file():
        result['historical_submission_exact'] = submission_rows(historical_submission) == submission_rows(actual_submission)
    else:
        result['historical_submission_note'] = 'Only meaningful for the full original control gallery; unavailable for smoke subsets or absent files.'
    result['top10_exact'] = result['serving_policy_submission_exact'] if result['serving_policy_submission_exact'] is not None else result['serving_policy_top10_exact']
    result['ranking_review_required'] = any(result.get(name) is False for name in
        ('top10_exact', 'historical_submission_exact', 'accepted_pairs_exact', 'query_refusals_exact'))
    return result


def filter_ground_truth(source, output, query_csv, gallery_csv):
    allowed = {'query': {r['image_id'] for r in parse_csv(Path(query_csv).read_bytes())},
               'gallery': {r['image_id'] for r in parse_csv(Path(gallery_csv).read_bytes())}}
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {'image_id', 'vehicle_id', 'camera_id', 'split'} <= set(reader.fieldnames):
            raise ValueError('Invalid control ground truth.')
        records = [r for r in reader if r.get('image_id') in allowed.get(r.get('split'), set())]
    with Path(output).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(records)


def check_control_inputs(directory):
    directory = Path(directory)
    paths = {name: directory / name for name in ('query.csv', 'gallery.csv', 'ground_truth.csv')}
    query = parse_csv(paths['query.csv'].read_bytes(), max_images=1_000_000)
    gallery = parse_csv(paths['gallery.csv'].read_bytes(), max_images=1_000_000)
    if len(gallery) < 10:
        raise ValueError('Control gallery requires at least ten images.')
    validate_ground_truth(paths['ground_truth.csv'], [r['image_id'] for r in query], [r['image_id'] for r in gallery])
    for name in ('embeddings.npy', 'official_report.json', 'source-manifest.json'):
        if (directory / name).is_file():
            paths[name] = directory / name
    if 'source-manifest.json' in paths:
        manifest = json.loads(paths['source-manifest.json'].read_text(encoding='utf-8'))
        for name, expected in manifest.get('files_sha256', {}).items():
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Control manifest may reference only local filenames.')
            if sha256(directory / name) != expected:
                raise ValueError(f'Control data differs from its source manifest: {name}')
    return paths


def child_check(module, arguments, output, *, timeout=14400):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    stdout, stderr = Path(output) / f'{module.rsplit(".", 1)[-1]}.stdout.log', Path(output) / f'{module.rsplit(".", 1)[-1]}.stderr.log'
    with stdout.open('wb') as out, stderr.open('wb') as err:
        completed = subprocess.run(command, stdout=out, stderr=err, timeout=timeout,
                                   cwd=Path(__file__).resolve().parents[2])
    if completed.returncode:
        raise ValueError(f'{module} failed with exit {completed.returncode}; see {stderr.name}.')


def verified_reference_policy(requested, bundle=None):
    """An explicit new-version policy may make old numerics diagnostic only.

    This does not change comparisons, tolerances, current-run repeatability or
    web/CLI parity. Legacy callers retain strict behavior without new fields.
    """
    if requested == 'strict':
        return {'policy': 'strict', 'verified': True, 'source': 'default_or_explicit_strict'}
    if requested != 'diagnostic_only':
        raise ValueError('Historical reference policy must be strict or diagnostic_only.')
    from .model_bundle import inspect_bundle
    selected = bundle if bundle is not None else os.getenv('VEHICLE_MODEL_BUNDLE')
    if not selected:
        raise ValueError('diagnostic_only requires an explicitly selected, verified E25 bundle.')
    verified = inspect_bundle(selected)
    if verified.adapter != 'e25_ensemble':
        raise ValueError('diagnostic_only is supported only by a verified E25 bundle.')
    artifact = verified.manifest['artifacts']['protocol']
    path = (verified.root / artifact['file']).resolve()
    if not path.is_relative_to(verified.root.resolve()):
        raise ValueError('Protocol must remain inside the verified bundle.')
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != artifact['sha256']:
        raise ValueError('Protocol changed after bundle verification.')
    protocol = json.loads(contents)
    from .e25_contract import MODEL_VERSION
    version = verified.manifest.get('model', {}).get('version', '')
    if (protocol.get('revision') != 'E25-R3'
            or version != MODEL_VERSION or '-r3-' not in str(version)
            or verified.config.get('inference', {}).get('runtime', {}).get('cuda_mode')
                not in {'compiled_overlap', 'compiled_precision_preprocess'}):
        raise ValueError('diagnostic_only is limited to the frozen R3 compiled runtime, never legacy or CUDA Graphs.')
    selection = protocol.get('runtime_selection')
    if (not isinstance(selection, dict)
            or selection.get('historical_reference_policy') != 'diagnostic_only'
            or selection.get('selection_scope') != 'threshold_calibration_only'
            or selection.get('control_used_for_selection') is not False
            or not isinstance(selection.get('frozen_at_utc'), str)
            or not selection['frozen_at_utc'].strip()):
        raise ValueError('diagnostic_only requires a frozen calibration-only runtime selection in the verified protocol.')
    try:
        frozen_at = datetime.fromisoformat(selection['frozen_at_utc'].replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Runtime selection freeze timestamp must be ISO8601 UTC.') from error
    if (frozen_at.tzinfo is None or frozen_at.utcoffset() != timedelta(0)
            or frozen_at > datetime.now(timezone.utc)):
        raise ValueError('Runtime selection freeze timestamp must be UTC and not in the future.')
    return {'policy': requested, 'verified': True, 'source': 'frozen_bundle_protocol',
            'protocol_sha256': digest, 'bundle_manifest_sha256': sha256(verified.root / 'bundle.json'),
            'runtime_selection': selection,
            'note': 'Historical numerical differences remain reported; this is not an equivalence waiver for new-version repeatability or web/CLI.'}


def run_validation(images, query_csv, gallery_csv, output, *, bundle=None, device='cuda',
                   batch_size=8, control_dir=None, allow_cpu_smoke=False,
                   historical_reference_policy='strict'):
    import torch
    selected_device = torch.device(device)
    gpu = selected_device.type == 'cuda'
    if selected_device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU smoke or CUDA validation is supported.')
    if not gpu and not allow_cpu_smoke:
        raise ValueError('CPU is allowed only with explicit --allow-cpu-smoke; it cannot produce GPU acceptance evidence.')
    if gpu and allow_cpu_smoke:
        raise ValueError('--allow-cpu-smoke is only valid with --device cpu.')
    if not 1 <= batch_size <= 32:
        raise ValueError('Batch size must be between 1 and 32.')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; evidence is never overwritten.')
    output.mkdir(parents=True)
    report_path = output / 'gpu-validation.json'
    report = {'schema_version': 1, 'status': 'running', 'passed': False,
              'submission_ready': False, 'official_measurement': False,
              'scope': 'gpu_acceptance' if gpu else 'cpu_smoke_only',
              'started_at': now(), 'device_requested': str(selected_device),
              'batch_size': batch_size, 'steps': {}, 'quality_review_required': True,
              'historical_reference_policy': {'policy': historical_reference_policy, 'verified': False},
              'historical_equivalence': None,
              'source': {'revision': os.getenv('VEHICLE_SOURCE_REVISION'),
                         'source_manifest_sha256': os.getenv('VEHICLE_SOURCE_MANIFEST_SHA256'),
                         'dirty': os.getenv('VEHICLE_SOURCE_DIRTY'),
                         'origin': 'Container build metadata declared in environment; verify against image inspection and the handoff manifest.',
                         'server_sha256': server_hashes()},
              'notes': ['This run never trains or tunes the model or threshold.',
                        'Hidden organizer test quality cannot be measured without its labels.',
                        'GPU completion is not a submission-readiness or contest-score certificate.']}
    atomic_json(report_path, report)
    started = time.perf_counter()
    provider = None
    stage = 'hardware'
    def save():
        report['active_step'] = stage
        atomic_json(report_path, report)
    try:
        with selected_runtime(str(selected_device), bundle):
            stage = 'historical_reference_policy'
            report['historical_reference_policy'] = verified_reference_policy(historical_reference_policy, bundle)
            if historical_reference_policy == 'diagnostic_only' and (not gpu or not control_dir):
                raise ValueError('diagnostic_only requires CUDA acceptance with the frozen historical control directory.')
            stage = 'hardware'
            hardware = hardware_info(str(selected_device))
            atomic_json(output / 'hardware.json', hardware)
            require_device(hardware)
            atomic_json(output / 'hardware.json', hardware)
            report['steps']['hardware'] = {'passed': True, 'file': 'hardware.json'}
            stage = 'inputs'
            save()
            source_files = {'query_csv': Path(query_csv), 'gallery_csv': Path(gallery_csv)}
            input_queries = parse_csv(Path(query_csv).read_bytes(), max_images=1_000_000)
            input_gallery = parse_csv(Path(gallery_csv).read_bytes(), max_images=1_000_000)
            if len(input_gallery) < 10:
                raise ValueError('At least ten gallery images are required.')
            find_images(images, input_queries + input_gallery)
            if control_dir:
                source_files.update({f'control/{name}': path for name, path in check_control_inputs(control_dir).items()})
                control_rows = parse_csv((Path(control_dir) / 'query.csv').read_bytes(), max_images=1_000_000) + parse_csv((Path(control_dir) / 'gallery.csv').read_bytes(), max_images=1_000_000)
                find_images(images, control_rows)
            source_hashes = {name: sha256(path) for name, path in source_files.items()}
            report['steps']['inputs'] = {'passed': True, 'source_sha256': source_hashes,
                                         'query_count': len(input_queries), 'gallery_count': len(input_gallery)}
            stage = 'model_load'
            save()
            provider, loading = load_provider_timed()
            atomic_json(output / 'model-load.json', loading)
            if not provider.available:
                raise ValueError(provider.reason or 'Selected real model is unavailable.')
            if torch.device(provider.model['device']).type != selected_device.type:
                raise ValueError('Loaded model device differs from explicitly requested device.')
            report.update(model=provider.model, inference_fingerprint=provider.inference_fingerprint,
                          calibration_sha256=provider.calibration_sha256, threshold_service=provider.threshold,
                          retrieval=getattr(provider, 'retrieval', None), retrieval_fingerprint=getattr(provider, 'retrieval_fingerprint', None))
            bundle_path = os.getenv('VEHICLE_MODEL_BUNDLE')
            if bundle_path and (Path(bundle_path) / 'bundle.json').is_file():
                report['source']['bundle_manifest_sha256'] = sha256(Path(bundle_path) / 'bundle.json')
            report['steps']['model_load'] = {'passed': True, 'file': 'model-load.json',
                                              'weights': weight_inventory(provider)}
            stage = 'independence'
            save()
            independence = check_independence(provider, images, query_csv, gallery_csv)
            atomic_json(output / 'independence.json', independence)
            report['steps']['independence'] = {'passed': independence['passed'], 'file': 'independence.json'}
            if not independence['passed']:
                raise ValueError('Query order/batch/ID independence check failed; inspect independence.json.')
            stage = 'open_test'
            save()
            main_query, main_gallery = Path(query_csv), Path(gallery_csv)
            if not gpu:
                main_query, main_gallery, _, _ = small_inputs(query_csv, gallery_csv, output / 'smoke-inputs')
            proof = run_batch(provider, images, main_query, main_gallery, output / 'open-test', batch_size=batch_size)
            report['steps']['open_test'] = {'passed': True, 'directory': 'open-test',
                                            'full_supplied_inputs': gpu, 'validation': proof['validation']}
            if control_dir:
                stage = 'control'
                save()
                control_dir = Path(control_dir)
                cq, cg = control_dir / 'query.csv', control_dir / 'gallery.csv'
                ground_truth = control_dir / 'ground_truth.csv'
                reference_indices = None
                if not gpu:
                    source_count = len(parse_csv(cq.read_bytes(), max_images=1_000_000))
                    cq, cg, qi, gi = small_inputs(cq, cg, output / 'control-smoke-inputs')
                    reference_indices = (qi, gi, source_count)
                    filtered = output / 'control-smoke-inputs' / 'ground_truth.csv'
                    filter_ground_truth(ground_truth, filtered, cq, cg)
                    ground_truth = filtered
                control_proof = run_batch(provider, images, cq, cg, output / 'control', batch_size=batch_size)
                evaluation = evaluate_export(output / 'control', cq, cg, ground_truth)
                report['steps']['control'] = {'passed': True, 'directory': 'control',
                                               'full_control_inputs': gpu, 'metrics': evaluation['official'],
                                               'historical_metrics': None}
                reference_file = control_dir / 'embeddings.npy'
                if reference_file.is_file():
                    comparison = reference_comparison(control_dir, output / 'control',
                                                      len(control_proof['input_order']['query']), reference_indices,
                                                      threshold=provider.threshold, retrieval=getattr(provider, 'retrieval', None))
                    atomic_json(output / 'control-reference-comparison.json', comparison)
                    report['steps']['control']['reference_comparison'] = 'control-reference-comparison.json'
                    report['steps']['control']['reference_embedding_transfer_passed'] = comparison['passed']
                    report['steps']['control']['reference_top10_exact'] = comparison['top10_exact']
                    report['steps']['control']['historical_submission_exact'] = comparison['historical_submission_exact']
                    report['steps']['control']['reference_accepted_pairs_exact'] = comparison['accepted_pairs_exact']
                    report['steps']['control']['ranking_review_required'] = comparison['ranking_review_required']
                    report['steps']['control']['historical_reference_enforced'] = historical_reference_policy == 'strict'
                    report['historical_equivalence'] = bool(comparison['passed'] and not comparison['ranking_review_required'])
                    if not comparison['passed'] and historical_reference_policy == 'strict':
                        report['steps']['control']['passed'] = False
                        raise ValueError('Control embeddings differ from supplied frozen model reference beyond declared transfer tolerance.')
                elif historical_reference_policy == 'diagnostic_only':
                    raise ValueError('diagnostic_only still requires the unchanged historical reference embeddings.')
                historical = control_dir / 'official_report.json'
                if historical.is_file():
                    report['steps']['control']['historical_metrics'] = {'file': str(historical), 'sha256': sha256(historical),
                        'comparison_scope': 'full reference; do not compare these metrics with a smoke subset',
                        'value': json_finite(json.loads(historical.read_text(encoding='utf-8')))}
            else:
                report['steps']['control'] = {'passed': None, 'reason': 'No labeled control directory supplied; quality and reference transfer are unverified.'}
            # Release the model before fresh child processes, so parent GPU memory
            # cannot skew benchmark memory or cause artificial out-of-memory errors.
            close = getattr(provider, 'close', None)
            if callable(close):
                close()
            provider = None
            gc.collect()
            if gpu:
                torch.cuda.empty_cache()
                stage = 'repeatability'
                save()
                child_check('vehicle.server.repeatability', ['--images', images, '--query-csv', query_csv,
                    '--gallery-csv', gallery_csv, '--output', output / 'repeatability', '--batch-size', batch_size,
                    '--timeout-seconds', '7200'], output, timeout=15000)
                repeat = json.loads((output / 'repeatability' / 'repeatability.json').read_text(encoding='utf-8'))
                report['steps']['repeatability'] = {'passed': repeat['passed'], 'directory': 'repeatability'}
                if not repeat['passed']:
                    raise ValueError('Two independent production runs did not reproduce identical decisions and matching vectors.')
                stage = 'benchmark'
                save()
                child_check('vehicle.server.benchmark', ['--images', images, '--input-csv', query_csv,
                    '--output', output / 'performance.json'], output)
                performance = json.loads((output / 'performance.json').read_text(encoding='utf-8'))
                if performance.get('status') != 'completed' or performance.get('scope') != 'local_gpu_measurement':
                    raise ValueError('A completed GPU benchmark was not produced.')
                report['steps']['benchmark'] = {'passed': True, 'file': 'performance.json'}
            else:
                for skipped in ('repeatability', 'benchmark'):
                    report['steps'][skipped] = {'passed': None, 'reason': 'Intentionally omitted for bounded CPU smoke; run the full CUDA command on the GPU computer.'}
            if any(sha256(path) != source_hashes[name] for name, path in source_files.items()):
                raise ValueError('Source CSV/reference data changed during validation; evidence cannot be accepted.')
            if server_hashes() != report['source']['server_sha256']:
                raise ValueError('Runtime server source changed during validation.')
            if verified_reference_policy(historical_reference_policy, bundle) != report['historical_reference_policy']:
                raise ValueError('Historical reference policy changed during acceptance.')
            control_complete = bool(control_dir and report['steps']['control'].get('reference_embedding_transfer_passed')
                                    and report['steps']['control'].get('ranking_review_required') is False)
            report.update(status=('verified_gpu_run' if control_complete else 'gpu_checks_completed_reference_review_required') if gpu else 'awaiting_gpu',
                          passed=True, all_acceptance_checks_completed=bool(gpu and control_complete))
            if historical_reference_policy == 'diagnostic_only':
                report.update(status='gpu_checks_completed_new_version_reference_review_required',
                              all_acceptance_checks_completed=False, quality_review_required=True,
                              current_version_gpu_checks_completed=True,
                              next_required='Review measured new-version quality and require strict same-version web/CLI parity in run_vm; historical equivalence is not implied.')
            stage = 'finished'
    except Exception as error:
        report['steps'].setdefault(stage, {})['passed'] = False
        report.update(status='failed', passed=False, error=str(error), failed_step=stage,
                      all_acceptance_checks_completed=False)
    except BaseException as error:
        report.update(status='interrupted', passed=False, error=type(error).__name__, failed_step=stage,
                      all_acceptance_checks_completed=False)
        raise
    finally:
        close = getattr(provider, 'close', None)
        if callable(close):
            try:
                close()
            except Exception as error:
                report.update(status='failed', passed=False, cleanup_error=str(error),
                              all_acceptance_checks_completed=False)
        report.update(finished_at=now(), wall_seconds=time.perf_counter() - started)
        report['evidence_sha256'] = {p.relative_to(output).as_posix(): sha256(p)
                                     for p in sorted(output.rglob('*')) if p.is_file() and p != report_path}
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--query-csv', required=True, type=Path)
    parser.add_argument('--gallery-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New evidence directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--control-dir', type=Path, help='Frozen control query.csv, gallery.csv, ground_truth.csv and optional reference embeddings/reports')
    parser.add_argument('--allow-cpu-smoke', action='store_true')
    parser.add_argument('--historical-reference-policy', choices=('strict', 'diagnostic_only'), default='strict',
                        help='diagnostic_only requires an explicit, verified calibration-only policy frozen in the E25 bundle; never changes numerical tolerances')
    args = parser.parse_args()
    try:
        result = run_validation(args.images, args.query_csv, args.gallery_csv, args.output,
                                bundle=args.bundle, device=args.device, batch_size=args.batch_size,
                                control_dir=args.control_dir, allow_cpu_smoke=args.allow_cpu_smoke,
                                historical_reference_policy=args.historical_reference_policy)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'Validation cannot start: {error}\n')
    print(json.dumps({'status': result['status'], 'passed': result['passed'],
                      'submission_ready': False, 'report': str(args.output / 'gpu-validation.json'),
                      'error': result.get('error')}, ensure_ascii=False, allow_nan=False))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
