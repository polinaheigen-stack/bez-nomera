"""Offline, matched compact baseline/KD experiments; no automatic promotion."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

from .batch import decoded_images, run_batch
from .compact_data import preflight, read_teacher_cache, verify_contract
from .evaluation import OFFICIAL_PATH, OFFICIAL_SHA256, evaluate_export, validate_ground_truth
from .model_bundle import digest_json, sha256
from .service import atomic_json
from .student_trial import select_threshold, teacher_cache
from .validation import write_outputs

ROOT = Path(__file__).resolve().parents[2]
TEACHER_SHA = 'ce273cd167a04e10d461b72e295193d3c1ca8dbda8efaf8e10e0214a018bf94d'
REUSED_CACHE_SHA = '519ca7077006a2ae8d4fdc26e650a715042234be28c636cb2b918fcb3e3cac6c'
REUSED_VECTORS_SHA = 'b51271cd8b43f478d590f888a34785cbe9401f62012d926668487bc81fd04498'
PRETRAINED_SHA = '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615'
COMMON = {'epochs': 20, 'seed': 20260929, 'P': 8, 'K': 4, 'lr_backbone': 2e-5,
          'lr_head': 1e-3, 'weight_decay': .01, 'warmup_epochs': 2,
          'kd_temperature': .1, 'label_smoothing': .1, 'inference_batch': 8, 'workers': 1}


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / 'vehicle/server').rglob('*.py'))}
    files['official_evaluator'] = sha256(OFFICIAL_PATH)
    return digest_json(files)


def load_profile(path):
    raw = Path(path).read_bytes()
    profile = json.loads(raw)
    if not isinstance(profile, dict) or type(profile.get('stand_id')) is not int or profile['stand_id'] not in (4, 5):
        raise ValueError('Profile requires stand_id 4 or 5')
    if 'stand' in profile and profile['stand'] != profile['stand_id']:
        raise ValueError('Conflicting stand IDs in profile')
    for key, value in COMMON.items():
        if profile.get(key) != value or type(profile[key]) not in (int, float):
            raise ValueError('Matched experiment requires fixed ' + key + '=' + str(value))
    for key, value in {'model': 'vit_small_plus_patch16_dinov3.lvd1689m', 'image_size': 256,
                       'embedding_dim': 384, 'selection': 'maximum_dev_raw_official_map_at_10',
                       'independent_quality_claim_allowed': False}.items():
        if profile.get(key) != value or type(profile.get(key)) is not type(value):
            raise ValueError('Controlled experiment requires fixed profile field: ' + key)
    expected = 0. if profile['stand_id'] == 4 else 1.
    if profile.get('kd_weight') != expected or type(profile['kd_weight']) not in (int, float):
        raise ValueError('Stand 4 has KD=0; stand 5 has KD=1')
    return profile


def load_evaluator():
    if sha256(OFFICIAL_PATH) != OFFICIAL_SHA256:
        raise ValueError('Official evaluator source differs')
    spec = importlib.util.spec_from_file_location('_compact_official_evaluator', OFFICIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha256(OFFICIAL_PATH) != OFFICIAL_SHA256:
        raise ValueError('Evaluator changed during import')
    return module


def dev_callback(contract, rows, paths, directory):
    """Select using the original dev raw cosine top10, with the official junk rules."""
    directory = Path(directory)
    directory.mkdir()
    evaluator = load_evaluator()
    files = {key: contract['files']['dev/' + key] for key in ('query', 'gallery', 'ground_truth')}
    def evaluate(model, epoch):
        for item in files.values():
            if sha256(item['path']) != item['sha256']:
                raise ValueError('Dev input changed')
        matrices = {}
        for split in ('query', 'gallery'):
            ordered = rows['dev/' + split]
            chunks = []
            for begin in range(0, len(ordered), 8):
                selected = ordered[begin:begin + 8]
                for row in selected:
                    if sha256(paths[row['image_id']]) != contract['images'][row['image_id']]['sha256']:
                        raise ValueError('Dev image changed before decode')
                with decoded_images(selected, paths) as images:
                    chunk = np.asarray(model.embed_batch(images, [r['bbox'] for r in selected]))
                if (chunk.dtype != np.float32 or chunk.shape != (len(selected), 384)
                        or not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk, axis=1), 1., atol=1e-4)):
                    raise ValueError('Invalid normalized compact dev embeddings')
                for row in selected:
                    if sha256(paths[row['image_id']]) != contract['images'][row['image_id']]['sha256']:
                        raise ValueError('Dev image changed during inference')
                chunks.append(chunk)
            matrices[split] = np.concatenate(chunks)
        query, gallery = evaluator.load_gt(files['ground_truth']['path'])
        qids, gids = (contract['order']['dev/' + key] for key in ('query', 'gallery'))
        query, gallery = query.loc[qids], gallery.loc[gids]
        ranked = {qid: [gids[int(i)] for i in np.argsort(-(matrices['gallery'] @ vector), kind='stable')[:10]]
                  for qid, vector in zip(qids, matrices['query'])}
        metrics = evaluator.ranking_metrics(query, gallery, ranked)
        if not math.isfinite(metrics['mAP@10']) or metrics.get('n_scored', 0) < 1:
            raise ValueError('Dev needs finite official mAP@10 on at least one scored query')
        for item in files.values():
            if sha256(item['path']) != item['sha256']:
                raise ValueError('Dev input changed during evaluation')
        if sha256(OFFICIAL_PATH) != OFFICIAL_SHA256:
            raise ValueError('Evaluator changed during dev evaluation')
        atomic_json(directory / f'epoch-{epoch:02d}.json', {'epoch': epoch, 'official_raw_ranking': metrics,
            'selection_metric': 'mAP@10', 'retrieval': 'raw cosine top10, stable ties; official junk filtering without refill',
            'evaluator_sha256': OFFICIAL_SHA256, 'dev_files': files,
            'calibration_or_control_used': False, 'independent_quality_claim_allowed': False})
        return metrics
    return evaluate


def calibrate(provider, dataset, contract, output):
    from .compact_model import source_identity as model_source_identity
    from .retrieval import rank_vectors
    files = {key: Path(contract['files']['calibration/' + key]['path']) for key in ('query', 'gallery', 'ground_truth')}
    collection = Path(output) / 'calibration-collection'
    run_batch(provider, Path(dataset) / 'images', files['query'], files['gallery'], collection, batch_size=8, prefetch=0)
    order = json.loads((collection / 'embedding_order.json').read_text(encoding='utf-8'))
    validate_ground_truth(files['ground_truth'], order['query'], order['gallery'])
    evaluator = load_evaluator()
    query, gallery = evaluator.load_gt(files['ground_truth'])
    query, gallery = query.loc[order['query']], gallery.loc[order['gallery']]
    vectors = np.load(collection / 'embeddings.npy', allow_pickle=False)
    count = len(query)
    tops = {}
    for qid, vector in zip(order['query'], vectors[:count]):
        scores, rank, _ = rank_vectors(vector, vectors[count:], 0., provider.retrieval)
        index = int(rank[0])
        tops[qid] = (order['gallery'][index], float(scores[index]))
    selected, trials = select_threshold(query, gallery, tops, evaluator)
    verify_contract(contract)
    lock = {'schema_version': 1, 'status': 'frozen', 'frozen_at_utc': utc(),
            'checkpoint_sha256': provider.model['sha256'], 'inference_fingerprint': provider.inference_fingerprint,
            'source_sha256': model_source_identity(), 'data_contract_sha256': digest_json(contract),
            'retrieval': deepcopy(provider.retrieval), 'threshold': selected['threshold'],
            'selection': 'max 7*official_query_F1+3*TNR; then TNR; then higher threshold',
            'evaluator_sha256': OFFICIAL_SHA256, 'selected': selected, 'trials': trials,
            'control_used_for_selection': False, 'checkpoint_selected_before_calibration': True}
    lock_path = Path(output) / 'student-calibration.json'
    atomic_json(lock_path, lock)
    provider.bind_calibration(lock_path)
    calibrated = Path(output) / 'calibration'
    write_outputs(calibrated, order['query'], order['gallery'], vectors[:count], vectors[count:],
                  provider.threshold, provider.retrieval)
    quality = evaluate_export(calibrated, files['query'], files['gallery'], files['ground_truth'])
    actual = quality['official']['candidates']
    if not math.isclose(7 * actual['F1'] + 3 * actual['TNR'], selected['points'], abs_tol=1e-12):
        raise ValueError('Frozen threshold differs from official calibration selection')
    return quality, lock_path


def compare_exports(reference, candidate, *, subset=False, tolerance=1e-5):
    """Numerical tolerance is explicit; any changed rank/acceptance is a hard failure."""
    reference, candidate = Path(reference), Path(candidate)
    orders = [json.loads((p / 'embedding_order.json').read_text()) for p in (reference, candidate)]
    if orders[0]['gallery'] != orders[1]['gallery']:
        raise ValueError('Parity gallery order differs')
    if (not subset and orders[0]['query'] != orders[1]['query']) or not set(orders[1]['query']) <= set(orders[0]['query']):
        raise ValueError('Parity query order/set differs')
    matrices = [np.load(p / 'embeddings.npy', allow_pickle=False) for p in (reference, candidate)]
    left_order = orders[0]['query'] + orders[0]['gallery']
    selected_order = orders[1]['query'] + orders[1]['gallery']
    # Query/gallery are disjoint in these held-fixed splits, as checked by preflight.
    lookup = {key: i for i, key in enumerate(left_order)}
    selected = matrices[0][[lookup[key] for key in selected_order]]
    right = matrices[1]
    if selected.shape != right.shape or not np.isfinite(right).all() or not np.allclose(selected, right, atol=tolerance, rtol=0):
        raise ValueError('Batch/query parity embedding tolerance exceeded')
    def predictions(path):
        with (path / 'submission.csv').open(newline='', encoding='utf-8') as stream:
            ranking = {row[0]: row[1:] for row in csv.reader(stream)}
        with (path / 'candidates.csv').open(newline='', encoding='utf-8') as stream:
            accepted = {(r['query_id'], r['gallery_id']): float(r['confidence']) for r in csv.DictReader(stream)}
        return ranking, accepted
    left_rank, left_candidates = predictions(reference)
    right_rank, right_candidates = predictions(candidate)
    expected_rank = {key: left_rank[key] for key in orders[1]['query']}
    expected_candidates = {key: value for key, value in left_candidates.items() if key[0] in set(orders[1]['query'])}
    if right_rank != expected_rank or set(right_candidates) != set(expected_candidates):
        raise ValueError('Batch/query parity changed top10 or accepted candidates')
    if any(abs(right_candidates[key] - value) > tolerance for key, value in expected_candidates.items()):
        raise ValueError('Batch/query parity confidence tolerance exceeded')
    if json.loads((reference / 'retrieval.json').read_text()) != json.loads((candidate / 'retrieval.json').read_text()):
        raise ValueError('Parity retrieval/threshold changed')
    return {'passed': True, 'embedding_max_abs': float(np.max(np.abs(selected - right))),
            'absolute_tolerance': tolerance, 'top10_identical': True, 'accepted_ids_identical': True,
            'exact_file_bytes': {name: sha256(reference / name) == sha256(candidate / name)
                                 for name in ('submission.csv', 'candidates.csv', 'embeddings.npy')}}


def parity(provider, dataset, contract, rows, output):
    query = Path(contract['files']['control/query']['path'])
    gallery = Path(contract['files']['control/gallery']['path'])
    reference = Path(output) / 'control'
    result = {'batch_size_reference': 1, 'batch_checks': {}, 'query_independence': None}
    for size in (8, 16, 32):
        destination = Path(output) / ('parity-batch-' + str(size))
        run_batch(provider, Path(dataset) / 'images', query, gallery, destination, batch_size=size, prefetch=0)
        result['batch_checks'][str(size)] = compare_exports(reference, destination)
    # Remove most other queries, reverse the remaining ones, and change batch placement.
    selected = list(reversed(rows['control/query'][::2]))
    subset_csv = Path(output) / 'query-independence.csv'
    with subset_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
        writer.writerows([[row['image_id'], *row['bbox']] for row in selected])
    destination = Path(output) / 'parity-query-independence'
    run_batch(provider, Path(dataset) / 'images', subset_csv, gallery, destination, batch_size=8, prefetch=0)
    result['query_independence'] = compare_exports(reference, destination, subset=True)
    result['passed'] = True
    atomic_json(Path(output) / 'parity.json', result)
    return result


def run(args):
    output, dataset, inputs = args.output.resolve(), args.dataset.resolve(), args.inputs.resolve()
    if output.exists() or any(output.is_relative_to(p) for p in (ROOT, dataset, inputs, args.pretrained.resolve().parent)):
        raise ValueError('Output must be new and outside runtime, source data and pretrained assets')
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'running', 'stand_id': None, 'phase': 'preflight',
              'training_finished': False, 'measured': False, 'gpu_measured': False,
              'submission_ready': False, 'independent_quality_claim_allowed': False,
              'calibration': None, 'control': None, 'benchmark': None, 'parity': None,
              'notes': ['Historical diagnostic splits; no new independent holdout claim.',
                        'Identical supervised training for stands 4/5; train-only E25 relational KD added only to stand 5.',
                        'No checkpoint selection or hyperparameter tuning on calibration/control.',
                        'No automatic replacement of the frozen production baseline.']}
    provider = None
    def phase(name):
        report.update(phase=name, phase_started_at_utc=utc())
        atomic_json(output / 'trial-report.json', report)
    try:
        phase('preflight')
        profile_sha = sha256(args.profile)
        profile = load_profile(args.profile)
        if sha256(args.profile) != profile_sha:
            raise ValueError('Profile changed while being parsed')
        report['stand_id'] = profile['stand_id']
        if args.pretrained_sha.lower() != PRETRAINED_SHA or sha256(args.pretrained) != PRETRAINED_SHA:
            raise ValueError('Expected the pinned local DINOv3-S+ pretrained safetensors')
        if profile['stand_id'] == 4 and any(p is not None for p in (args.teacher_cache, args.teacher_bundle, args.teacher_cache_receipt)):
            raise ValueError('Stand 4 is supervised-only and must not receive a teacher')
        contract, rows, paths, labels = preflight(dataset, inputs)
        for name, path in (('profile', args.profile), ('pretrained', args.pretrained)):
            contract['files'][name] = {'path': str(path.resolve()),
                                       'sha256': profile_sha if name == 'profile' else PRETRAINED_SHA}
        atomic_json(output / 'data-contract.json', contract)
        contract_sha = digest_json(contract)
        report.update(data_contract_sha256=contract_sha, identity_disjointness_proved=True,
                      profile=profile, source_sha256=source_identity(),
                      pretrained={'sha256': PRETRAINED_SHA, 'path': str(args.pretrained.resolve())})
        vectors, cache_proof = None, None
        if profile['stand_id'] == 5:
            if args.teacher_cache is not None:
                metadata_path = args.teacher_cache / 'cache.json' if args.teacher_cache.is_dir() else args.teacher_cache
                if sha256(metadata_path) != REUSED_CACHE_SHA:
                    raise ValueError('Reused cache metadata must match the reviewed E25 receipt')
                if args.teacher_cache_receipt is None:
                    raise ValueError('Reused teacher cache requires its pinned receipt')
                vectors, cache_proof = read_teacher_cache(args.teacher_cache, rows['train'], contract, TEACHER_SHA,
                                                         receipt=args.teacher_cache_receipt)
                if cache_proof['embeddings_sha256'] != REUSED_VECTORS_SHA:
                    raise ValueError('Reused cache descriptor checksum differs from pinned evidence')
                report['teacher_cache'] = cache_proof
                for name, path in (('teacher_cache_metadata', metadata_path),
                                   ('teacher_cache_vectors', metadata_path.with_name('embeddings.npy')),
                                   ('teacher_cache_receipt', args.teacher_cache_receipt)):
                    contract['files'][name] = {'path': str(path.resolve()), 'sha256': sha256(path)}
            elif args.teacher_bundle is not None:
                from .model_bundle import inspect_bundle
                teacher = inspect_bundle(args.teacher_bundle)
                if teacher.adapter != 'e25_ensemble' or teacher.weights_sha256 != TEACHER_SHA:
                    raise ValueError('Only the pinned owned E25 teacher bundle is permitted')
                report['teacher'] = {'sha256': teacher.weights_sha256, 'cache': 'to_be_created_on_GPU_train_only'}
            else:
                raise ValueError('Stand 5 requires the reviewed train cache or the pinned E25 bundle')
        verify_contract(contract)
        contract_sha = digest_json(contract)
        report['data_contract_sha256'] = contract_sha
        atomic_json(output / 'data-contract.json', contract)
        if args.preflight_only:
            report['status'] = 'prepared_not_trained_not_measured'
            return report
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA is required for actual trials; CPU is preflight/tests only')
        from .compact_model import CompactProvider
        from .compact_learning import train_model
        if profile['stand_id'] == 5 and vectors is None:
            phase('teacher_train_cache')
            from .provider import ProductionProvider
            previous = os.environ.get('VEHICLE_MODEL_BUNDLE')
            os.environ['VEHICLE_MODEL_BUNDLE'] = str(args.teacher_bundle.resolve())
            try:
                provider = ProductionProvider(device='cuda')
                if not provider.available or provider.model['sha256'] != TEACHER_SHA:
                    raise ValueError('Owned E25 teacher unavailable or changed')
                cache_path = teacher_cache(provider, rows['train'], paths, output / 'teacher-train-cache', contract_sha)
                vectors, cache_proof = read_teacher_cache(cache_path, rows['train'], contract, TEACHER_SHA)
            finally:
                if provider is not None:
                    provider.close()
                    provider = None
                if previous is None:
                    os.environ.pop('VEHICLE_MODEL_BUNDLE', None)
                else:
                    os.environ['VEHICLE_MODEL_BUNDLE'] = previous
            report['teacher_cache'] = cache_proof
            gc.collect()
            torch.cuda.empty_cache()
        verify_contract(contract)
        phase('training')
        training_profile = deepcopy(profile)
        training_profile['stand'] = profile['stand_id']
        training_profile['provenance'] = {'data_contract': contract, 'data_contract_sha256': contract_sha,
                                          'teacher_cache': cache_proof, 'trial_source_sha256': report['source_sha256']}
        checkpoint = Path(train_model(training_profile, rows['train'], paths, labels,
                                      args.pretrained, PRETRAINED_SHA, output / 'training',
                                      dev_callback(contract, rows, paths, output / 'dev-epochs'),
                                      teacher_vectors=vectors, device='cuda'))
        verify_contract(contract)
        report['training_finished'] = True
        report['student_checkpoint'] = {'path': str(checkpoint), 'sha256': sha256(checkpoint), 'bytes': checkpoint.stat().st_size}
        phase('checkpoint_freeze')
        atomic_json(output / 'student-selection-lock.json', {'checkpoint': report['student_checkpoint'],
            'frozen_at_utc': utc(), 'selection': 'maximum dev official raw cosine mAP@10',
            'calibration_or_control_used_for_selection': False, 'data_contract_sha256': contract_sha,
            'source_sha256': report['source_sha256']})
        provider = CompactProvider(checkpoint, device='cuda')
        phase('calibration')
        report['calibration'], lock_path = calibrate(provider, dataset, contract, output)
        lock_sha = sha256(lock_path)
        phase('control')
        controls = {key: Path(contract['files']['control/' + key]['path']) for key in ('query', 'gallery', 'ground_truth')}
        run_batch(provider, dataset / 'images', controls['query'], controls['gallery'], output / 'control', batch_size=1, prefetch=1)
        report['control'] = evaluate_export(output / 'control', controls['query'], controls['gallery'], controls['ground_truth'])
        if sha256(lock_path) != lock_sha:
            raise ValueError('Calibration changed during held-fixed control evaluation')
        phase('batch_and_query_parity')
        report['parity'] = parity(provider, dataset, contract, rows, output)
        phase('benchmark')
        from .benchmark import measure
        speed_csv = dataset / 'test_query.csv'
        speed_record = contract['files'].get('official_test/query')
        if speed_record is None:
            speed_csv = controls['query']
            speed_record = contract['files']['control/query']
        if sha256(speed_csv) != speed_record['sha256']:
            raise ValueError('Performance query CSV changed since preflight')
        report['performance_input'] = deepcopy(speed_record)
        report['benchmark'] = measure(provider, dataset / 'images', speed_csv, output / 'performance.json')
        verify_contract(contract)
        if (sha256(lock_path) != lock_sha or sha256(checkpoint) != report['student_checkpoint']['sha256']
                or source_identity() != report['source_sha256']):
            raise ValueError('Frozen checkpoint/calibration/runtime changed during the experiment')
        report.update(status='completed', phase='completed', measured=True, gpu_measured=True,
                      student_calibration_sha256=lock_sha, student_inference_ready=True)
        return report
    except BaseException as error:
        report.update(status='failed', failed_stage=report.get('phase'), error=f'{type(error).__name__}: {error}',
                      measured=False, gpu_measured=False)
        raise
    finally:
        try:
            if provider is not None:
                provider.close()
        finally:
            atomic_json(output / 'trial-report.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'dataset', 'inputs', 'pretrained', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pretrained-sha', required=True)
    parser.add_argument('--teacher-bundle', type=Path)
    parser.add_argument('--teacher-cache', type=Path)
    parser.add_argument('--teacher-cache-receipt', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'Compact trial failed: {error}\n')
    print(json.dumps({key: result[key] for key in ('stand_id', 'status', 'training_finished', 'measured', 'gpu_measured')}))


if __name__ == '__main__':
    main()
