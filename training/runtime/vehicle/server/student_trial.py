"""Offline P1+P4: own E25 teacher cache, ViT-B student, calibration, held-fixed control.

Preparation never trains. This CLI runs on a subsequently supplied GPU; outputs
are new-only, failures are explicit, and historical control is never called an
independent holdout. No downloads, competitor models or confidence-policy P2.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random

import numpy as np

from .batch import find_images, infer_images, run_batch
from .frame_buffer import FrameBuffer
from .model_bundle import digest_json, sha256
from .service import atomic_json, parse_csv

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path, expected_hash=None):
    """Allow identity metadata in training CSV, never forward it to a model."""
    raw = Path(path).read_bytes()
    if expected_hash is not None and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('CSV changed before parsing: ' + str(path))
    source = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
    for row in source:
        writer.writerow([row[key] for key in ('image_id', 'x', 'y', 'w', 'h')])
    return parse_csv(stream.getvalue().encode(), max_images=10_000_000)


def preflight(dataset, train_csv, val_csv, *, identity_csv=None, checkpoint=None):
    dataset = Path(dataset).resolve()
    paths, rows, csv_hashes = {}, {}, {}
    for split in ('calibration', 'control'):
        supplied = dataset / split
        directory = supplied if supplied.is_dir() else ROOT / split
        for kind in ('query', 'gallery', 'ground_truth'):
            key = f'{split}/{kind}'
            paths[key] = directory / (kind + '.csv')
            if not paths[key].is_file():
                raise ValueError('Missing split input: ' + str(paths[key]))
            csv_hashes[key] = sha256(paths[key])
            if kind != 'ground_truth':
                rows[key] = read_rows(paths[key], csv_hashes[key])
    if checkpoint is None:
        if train_csv is None or val_csv is None:
            raise ValueError('Training requires explicit --train-csv and --val-csv')
        for key, path in (('train', train_csv), ('val', val_csv)):
            paths[key] = Path(path).resolve()
            csv_hashes[key] = sha256(paths[key])
            rows[key] = read_rows(path, csv_hashes[key])
    elif train_csv is not None or val_csv is not None:
        raise ValueError('Evaluation-only checkpoint cannot be combined with train/val CSVs')
    protected, protected_rows = set(), []
    # Built-in IDs are excluded even when explicit alternative evaluation splits are used.
    for split in ('calibration', 'control'):
        for kind in ('query', 'gallery'):
            key = f'builtin/{split}/{kind}'
            paths[key] = ROOT / split / (kind + '.csv')
            csv_hashes[key] = sha256(paths[key])
            builtin_rows = read_rows(paths[key], csv_hashes[key])
            protected_rows.extend(builtin_rows)
            protected.update(r['image_id'] for r in builtin_rows)
    groups = {key: {r['image_id'] for r in value} for key, value in rows.items()}
    evaluation_ids = set().union(*(ids for key, ids in groups.items() if '/' in key))
    protected |= evaluation_ids
    identity_protected = set(protected)
    # Closed-test labels are unavailable; exclude its images without inventing identities.
    for kind in ('query', 'gallery'):
        official_path = dataset / ('test_' + kind + '.csv')
        if official_path.is_file():
            key = 'official_test/' + kind
            paths[key], csv_hashes[key] = official_path, sha256(official_path)
            official_rows = read_rows(official_path, csv_hashes[key])
            protected_rows.extend(official_rows)
            protected.update(row['image_id'] for row in official_rows)
    if checkpoint is None:
        if groups['train'] & groups['val'] or (groups['train'] | groups['val']) & protected:
            raise ValueError('Training/validation overlap each other or protected evaluation images')
    calibration_ids = groups['calibration/query'] | groups['calibration/gallery']
    control_ids = groups['control/query'] | groups['control/gallery']
    if calibration_ids & control_ids:
        raise ValueError('Calibration and control image IDs overlap')
    image_paths = find_images(dataset / 'images', [r for items in rows.values() for r in items] + protected_rows)
    image_hashes = {key: sha256(value) for key, value in image_paths.items()}
    group_hashes = {key: {image_hashes[i] for i in ids} for key, ids in groups.items()}
    evaluation_hashes = {image_hashes[key] for key in protected}
    if checkpoint is None and (group_hashes['train'] & group_hashes['val']
            or (group_hashes['train'] | group_hashes['val']) & evaluation_hashes):
        raise ValueError('Training/validation duplicate evaluation pixels or each other under renamed IDs')
    if ((group_hashes['calibration/query'] | group_hashes['calibration/gallery'])
            & (group_hashes['control/query'] | group_hashes['control/gallery'])):
        raise ValueError('Calibration and control contain duplicate image bytes')
    identity_proved = False
    if identity_csv is not None:
        paths['identity_mapping'] = Path(identity_csv).resolve()
        csv_hashes['identity_mapping'] = sha256(paths['identity_mapping'])
        with paths['identity_mapping'].open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['image_id', 'identity_id']:
                raise ValueError('Identity map requires exactly image_id,identity_id in one global namespace')
            mapping = {}
            for row in reader:
                if not row['image_id'] or not row['identity_id'] or row['image_id'] in mapping:
                    raise ValueError('Invalid/duplicate identity mapping')
                mapping[row['image_id']] = row['identity_id']
        needed = set().union(*groups.values()) | identity_protected
        if not needed <= set(mapping):
            raise ValueError('Identity map must cover train, val, selected and built-in protected IDs')
        if checkpoint is None:
            tr, va, pr = ({mapping[i] for i in ids} for ids in (groups['train'], groups['val'], identity_protected))
            if tr & va or (tr | va) & pr:
                raise ValueError('Training/validation identities overlap each other or protected evaluation')
        if {mapping[i] for i in calibration_ids} & {mapping[i] for i in control_ids}:
            raise ValueError('Calibration and control identities overlap')
        identity_proved = True
    contract = {'schema_version': 1, 'files': {key: {'path': str(path), 'sha256': csv_hashes[key]} for key, path in paths.items()},
                'images': {key: {'path': str(image_paths[key]), 'sha256': value} for key, value in image_hashes.items()},
                'order': {key: [r['image_id'] for r in value] for key, value in rows.items()},
                'bboxes': {key: [r['bbox'] for r in value] for key, value in rows.items()},
                'counts': {key: len(value) for key, value in rows.items()},
                'train_val_evaluation_image_disjoint': True,
                'identity_disjointness_proved': identity_proved,
                'independent_quality_claim_allowed': False,
                'notes': ['Optional identity map is an operator assertion in one global namespace.',
                          'Without an identity map only image disjointness is verified.',
                          'Historical E25 teacher/control provenance prevents a new independent-holdout claim.',
                          'Official test CSVs, when present, are excluded by image ID and bytes; their unknown identities are not proved disjoint.',
                          'Control labels are hashed here, not used for model or threshold selection.']}
    verify_contract(contract)
    return contract, rows, image_paths


def verify_contract(contract):
    for entry in [*contract['files'].values(), *contract['images'].values()]:
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('Input changed: ' + entry['path'])


def teacher_cache(provider, rows, paths, output, contract_sha):
    output = Path(output)
    output.mkdir()
    model = json.loads(json.dumps(provider.model))
    chunks, records = [], []
    with FrameBuffer([{**row, 'path': paths[row['image_id']]} for row in rows], ahead=1, verify_hash=True) as frames:
        for frame in frames:
            vectors, _ = infer_images(provider, [frame.row], [frame.require_image()])
            if vectors.shape != (1, 1024) or provider.model != model:
                raise ValueError('Teacher must remain the same normalized 1024D E25 model')
            frame.verify_unchanged()
            records.append({'image_id': frame.row['image_id'], 'bbox': frame.row['bbox'], 'sha256': frame.sha256})
            chunks.append(vectors)
    if len(records) != len(rows):
        raise ValueError('Teacher cache is incomplete')
    vectors = np.concatenate(chunks)
    np.save(output / 'embeddings.npy', vectors, allow_pickle=False)
    atomic_json(output / 'cache.json', {'schema_version': 1, 'status': 'completed', 'teacher': model,
        'teacher_calibration_sha256': provider.calibration_sha256,
        'teacher_retrieval_used_for_targets': False, 'data_contract_sha256': contract_sha,
        'ordered_inputs': records, 'shape': list(vectors.shape), 'dtype': 'float32',
        'embeddings_sha256': sha256(output / 'embeddings.npy'),
        'pipeline': {'cpu_frames_ahead': 1, 'actual_teacher_model_batch_size': 1, 'model_execution': 'consumer_only'}})
    return output / 'cache.json'


def read_cache(path, rows, contract_sha):
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    vector_path = path.with_name('embeddings.npy')
    if (record.get('status') != 'completed' or record.get('data_contract_sha256') != contract_sha
            or record.get('embeddings_sha256') != sha256(vector_path)
            or [r['image_id'] for r in record['ordered_inputs']] != [r['image_id'] for r in rows]
            or [tuple(r['bbox']) for r in record['ordered_inputs']] != [tuple(r['bbox']) for r in rows]):
        raise ValueError('Cache does not match ordered training inputs')
    vectors = np.load(vector_path, allow_pickle=False)
    if (vectors.shape != (len(rows), 1024) or vectors.dtype != np.float32 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError('Invalid teacher descriptors')
    return vectors


def train_student(rows, paths, train_cache, val_cache, output, contract, args):
    import torch
    from torch.nn import functional as F
    from .student_model import StudentModel, CONFIG, amp_context, dependencies, source_identity
    from .reidkit_adapter import image_tensor
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    contract_sha = digest_json(contract)
    targets = {'train': read_cache(train_cache, rows['train'], contract_sha),
               'val': read_cache(val_cache, rows['val'], contract_sha)}
    init_hash = sha256(args.init_backbone) if args.init_backbone else None
    model = StudentModel(initial_weights=args.init_backbone).to('cuda')
    if args.init_backbone and sha256(args.init_backbone) != init_hash:
        raise ValueError('Initial backbone changed during loading')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    provenance = {'source_sha256': source_identity(), 'dependencies': dependencies(),
                  'train_cache_sha256': sha256(train_cache), 'val_cache_sha256': sha256(val_cache),
                  'teacher': json.loads(Path(train_cache).read_text())['teacher'],
                  'data_contract_sha256': contract_sha, 'data_contract': contract,
                  'initial_backbone_sha256': init_hash, 'initialization': 'explicit_local_native_timm' if init_hash else 'random_fresh',
                  'seed': args.seed, 'epochs_requested': args.epochs, 'batch_size': args.batch_size,
                  'learning_rate': args.learning_rate, 'weight_decay': .01,
                  'loss': 'mean(1-cosine(student,teacher)); fixed original bbox preprocessing; no augmentation',
                  'selection_basis': 'minimum_validation_cosine_loss', 'control_used_for_selection': False,
                  'identity_disjointness_proved': contract['identity_disjointness_proved'],
                  'independent_quality_claim_allowed': False}
    history, best, best_epoch = [], float('inf'), None
    best_path = output / 'selected-state.pt'
    for epoch in range(1, args.epochs + 1):
        metrics = {'epoch': epoch}
        for split in ('train', 'val'):
            training = split == 'train'
            model.train(training)
            order = np.random.default_rng(args.seed + epoch).permutation(len(rows[split])) if training else np.arange(len(rows[split]))
            total, count, pending = 0., 0, []
            def step(batch):
                nonlocal total, count
                indices, tensors = zip(*batch)
                tensor = torch.stack(tensors).to('cuda')
                target = torch.from_numpy(targets[split][list(indices)]).to('cuda')
                with torch.set_grad_enabled(training), amp_context('cuda'):
                    vectors = model(tensor)
                    loss = (1 - F.cosine_similarity(vectors.float(), target.float(), dim=1)).mean()
                if not torch.isfinite(loss).item():
                    raise ValueError('Nonfinite distillation loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                total += float(loss.detach()) * len(batch)
                count += len(batch)
            selected = [{**rows[split][int(i)], 'path': paths[rows[split][int(i)]['image_id']], 'target_index': int(i)} for i in order]
            with FrameBuffer(selected, ahead=1, verify_hash=True) as frames:
                for frame in frames:
                    if frame.sha256 != contract['images'][frame.row['image_id']]['sha256']:
                        raise ValueError('Training image differs from frozen teacher input')
                    pending.append((frame.row['target_index'], image_tensor(frame.require_image(), frame.row['bbox'], CONFIG['image_size'])))
                    frame.verify_unchanged()
                    if len(pending) == args.batch_size:
                        step(pending)
                        pending = []
                if pending:
                    step(pending)
            if count != len(rows[split]):
                raise ValueError('Training/validation pass is incomplete')
            metrics[split + '_cosine_loss'] = total / count
        history.append(metrics)
        if metrics['val_cosine_loss'] < best:
            best, best_epoch = metrics['val_cosine_loss'], epoch
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, best_path)
        atomic_json(output / 'training-history.json', {'status': 'running', 'epochs': history, 'selected_epoch': best_epoch})
        print(json.dumps(metrics), flush=True)
    verify_contract(contract)
    if sha256(train_cache) != provenance['train_cache_sha256'] or sha256(val_cache) != provenance['val_cache_sha256']:
        raise ValueError('Teacher cache changed during training')
    checkpoint = output / 'student.pt'
    torch.save({'kind': 'owned_e25_distilled_student', 'format_version': 1, 'training_finished': True,
                'config': CONFIG, 'selected_epoch': best_epoch, 'selected_validation_cosine_loss': best,
                'provenance': provenance, 'model_state': torch.load(best_path, map_location='cpu', weights_only=True)}, checkpoint)
    best_path.unlink()  # Only our temporary state; retained complete checkpoint contains identical tensors.
    atomic_json(output / 'training-history.json', {'status': 'completed', 'epochs': history, 'selected_epoch': best_epoch})
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint


def select_threshold(query, gallery, tops, evaluator):
    boundaries = sorted(set([0., 1., *(pair[1] for pair in tops.values())]))
    thresholds = [0., 1.] + [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:]) if a < b]
    trials = []
    for threshold in thresholds:
        metrics = evaluator.candidate_metrics(query, gallery, {qid: [pair] for qid, pair in tops.items() if pair[1] >= threshold})
        if not all(math.isfinite(metrics[key]) for key in ('F1', 'TNR')):
            raise ValueError('Calibration needs both positive and open-set queries')
        trials.append({'threshold': threshold, 'F1': metrics['F1'], 'TNR': metrics['TNR'],
                       'points': 7 * metrics['F1'] + 3 * metrics['TNR']})
    selected = max(trials, key=lambda row: (row['points'], row['TNR'], row['threshold']))
    reject = evaluator.candidate_metrics(query, gallery, {})
    if not all(math.isfinite(reject[key]) for key in ('F1', 'TNR')):
        raise ValueError('Reject-all calibration metrics are undefined')
    # threshold=1 cannot reject an exact score of 1. Never silently omit a better
    # official decision state just because this serving contract cannot encode it.
    if (7 * reject['F1'] + 3 * reject['TNR'], reject['TNR']) > (selected['points'], selected['TNR']):
        raise ValueError('Optimal reject-all calibration is not representable by the [0,1] threshold contract')
    return selected, trials


def calibrate(provider, dataset, contract, output):
    from .evaluation import OFFICIAL_PATH, OFFICIAL_SHA256, evaluate_export
    from .retrieval import rank_vectors
    from .student_model import source_identity
    from .validation import write_outputs
    def path(key):
        return Path(contract['files']['calibration/' + key]['path'])
    export = output / 'calibration-collection'
    run_batch(provider, dataset / 'images', path('query'), path('gallery'), export, batch_size=1, prefetch=1)
    if sha256(OFFICIAL_PATH) != OFFICIAL_SHA256:
        raise ValueError('Official evaluator source differs')
    spec = importlib.util.spec_from_file_location('student_official_calibration', OFFICIAL_PATH)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    from .evaluation import validate_ground_truth
    order = json.loads((export / 'embedding_order.json').read_text())
    validate_ground_truth(path('ground_truth'), order['query'], order['gallery'])
    query, gallery = evaluator.load_gt(path('ground_truth'))
    query, gallery = query.loc[order['query']], gallery.loc[order['gallery']]
    vectors = np.load(export / 'embeddings.npy', allow_pickle=False)
    count, tops = len(query), {}
    for qid, vector in zip(order['query'], vectors[:len(query)]):
        scores, rank, _ = rank_vectors(vector, vectors[count:], 0., provider.retrieval)
        first = int(rank[0])
        tops[qid] = (order['gallery'][first], float(scores[first]))
    selected, trials = select_threshold(query, gallery, tops, evaluator)
    verify_contract(contract)
    lock = {'schema_version': 1, 'status': 'frozen', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256': provider.model['sha256'], 'inference_fingerprint': provider.inference_fingerprint,
            'source_sha256': source_identity(), 'data_contract_sha256': digest_json(contract),
            'retrieval': provider.retrieval, 'threshold': selected['threshold'],
            'selection': 'max 7*official_query_F1+3*TNR; then TNR; then higher threshold',
            'evaluator_sha256': OFFICIAL_SHA256, 'selected': selected, 'trials': trials,
            'control_used_for_selection': False}
    lock_path = output / 'student-calibration.json'
    atomic_json(lock_path, lock)
    provider.bind_calibration(lock_path)
    calibrated = output / 'calibration'
    write_outputs(calibrated, order['query'], order['gallery'], vectors[:count], vectors[count:], provider.threshold, provider.retrieval)
    quality = evaluate_export(calibrated, path('query'), path('gallery'), path('ground_truth'))
    actual = quality['official']['candidates']
    if not math.isclose(7 * actual['F1'] + 3 * actual['TNR'], selected['points'], abs_tol=1e-12):
        raise ValueError('Calibrated export differs from official selection metrics')
    return quality, lock_path


def run(args):
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.dataset.resolve()) or output.is_relative_to(ROOT):
        raise ValueError('Output must be new and outside the immutable stand and dataset')
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'stand': 'P1+P4_distillation', 'status': 'running',
              'training_finished': False, 'measured': False, 'gpu_measured': False,
              'submission_ready': False, 'independent_quality_claim_allowed': False,
              'calibration': None, 'control': None, 'benchmark': None,
              'notes': ['No P2 policy; fixed P1 gallery-only single-query retrieval.',
                        'Historical labeled control is diagnostic, not a new independent test.']}
    provider = None
    def phase(name):
        report['phase'] = name
        report['phase_started_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / 'trial-report.json', report)
    try:
        phase('preflight')
        contract, rows, paths = preflight(args.dataset, args.train_csv, args.val_csv,
                                        identity_csv=args.identity_csv, checkpoint=args.student_checkpoint)
        atomic_json(output / 'data-contract.json', contract)
        report['data_contract_sha256'] = digest_json(contract)
        report['identity_disjointness_proved'] = contract['identity_disjointness_proved']
        for label, selected in (('initial_backbone', args.init_backbone), ('student_checkpoint', args.student_checkpoint)):
            if selected is not None:
                report[label + '_input'] = {'path': str(selected.resolve()), 'sha256': sha256(selected), 'bytes': selected.stat().st_size}
        if args.preflight_only:
            report['status'] = 'prepared_not_trained_not_measured'
            return report
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        if not torch.cuda.is_available():
            raise ValueError('A CUDA GPU is required; real CPU training/inference is disabled')
        from .student_model import StudentProvider, source_identity, load_checkpoint
        report['source_sha256'] = source_identity()
        checkpoint = args.student_checkpoint
        if checkpoint is None:
            phase('teacher_load')
            from .provider import ProductionProvider
            from .benchmark import weight_inventory
            os.environ['VEHICLE_MODEL_BUNDLE'] = str(args.teacher_bundle.resolve())
            provider = ProductionProvider(device='cuda')
            if (not provider.available or provider.model.get('dimension') != 1024
                    or not getattr(provider, '_bundle_mode', False)):
                raise ValueError('The owned E25 teacher could not be loaded: ' + str(provider.reason))
            from .model_bundle import inspect_bundle
            teacher = inspect_bundle(args.teacher_bundle)
            if teacher.adapter != 'e25_ensemble' or teacher.config.get('fusion') != 'normalized_sqrt_weight_concatenation':
                raise ValueError('Only the owned E25 DINO/Swin teacher is permitted')
            report['teacher'] = {'model': provider.model, 'weights': weight_inventory(provider),
                                 'config': teacher.config, 'manifest_sha256': sha256(args.teacher_bundle / 'bundle.json')}
            phase('teacher_cache')
            train_cache = teacher_cache(provider, rows['train'], paths, output / 'teacher-train-cache', digest_json(contract))
            val_cache = teacher_cache(provider, rows['val'], paths, output / 'teacher-val-cache', digest_json(contract))
            if (weight_inventory(provider) != report['teacher']['weights']
                    or sha256(args.teacher_bundle / 'bundle.json') != report['teacher']['manifest_sha256']):
                raise ValueError('Teacher weights/bundle changed during cache extraction')
            provider.close()
            provider = None
            gc.collect()
            torch.cuda.empty_cache()
            verify_contract(contract)
            phase('training')
            checkpoint = train_student(rows, paths, train_cache, val_cache, output, contract, args)
        else:
            phase('student_checkpoint_validation')
            # Reusing a trained student still requires image/identity exclusion against its recorded training inputs.
            temporary, saved = load_checkpoint(checkpoint)
            del temporary
            old = saved['provenance']['data_contract']
            training_ids = set(old['order']['train']) | set(old['order']['val'])
            evaluation_ids = set(paths)
            train_hashes = {old['images'][key]['sha256'] for key in training_ids}
            if training_ids & evaluation_ids or train_hashes & {item['sha256'] for item in contract['images'].values()}:
                raise ValueError('Evaluation overlaps the recorded student training/validation inputs')
            report['identity_disjointness_proved'] = False
            report['notes'].append('Eval-only reuse checks image hashes; cross-run identity disjointness is not newly established.')
        report['training_finished'] = True
        report['student_checkpoint'] = {'path': str(checkpoint), 'sha256': sha256(checkpoint), 'bytes': Path(checkpoint).stat().st_size}
        phase('checkpoint_freeze')
        atomic_json(output / 'student-selection-lock.json', {'checkpoint_sha256': sha256(checkpoint),
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'control_used_for_selection': False,
            'source_sha256': source_identity(), 'data_contract_sha256': digest_json(contract)})
        provider = StudentProvider(checkpoint, device='cuda')
        phase('calibration')
        report['calibration'], lock_path = calibrate(provider, args.dataset, contract, output)
        lock_sha = sha256(lock_path)
        phase('control')
        control = {key: Path(contract['files']['control/' + key]['path']) for key in ('query', 'gallery', 'ground_truth')}
        run_batch(provider, args.dataset / 'images', control['query'], control['gallery'], output / 'control', batch_size=1, prefetch=1)
        from .evaluation import evaluate_export
        report['control'] = evaluate_export(output / 'control', control['query'], control['gallery'], control['ground_truth'])
        if sha256(lock_path) != lock_sha:
            raise ValueError('Threshold lock changed during control')
        from .benchmark import measure
        phase('benchmark')
        speed_csv = args.dataset / 'test_query.csv'
        if not speed_csv.is_file():
            speed_csv = control['query']
        report['performance_input'] = {'path': str(speed_csv), 'sha256': sha256(speed_csv),
            'scope': 'same original query CSV as stand 1 when available; control fallback is explicitly recorded'}
        report['benchmark'] = measure(provider, args.dataset / 'images', speed_csv, output / 'performance.json')
        from .frame_buffer_check import run_check
        phase('frame_buffer_parity')
        report['frame_buffer'] = run_check(args.dataset / 'images', speed_csv, output / 'frame-buffer-comparison.json',
                                           repeats=3, warmup=5, provider=provider)
        verify_contract(contract)
        if source_identity() != report['source_sha256']:
            raise ValueError('Runtime source changed during the trial')
        if sha256(lock_path) != lock_sha or sha256(checkpoint) != report['student_checkpoint']['sha256']:
            raise ValueError('Frozen student/calibration changed during measurement')
        report.update(status='completed', measured=True, gpu_measured=True,
                      phase='completed', student_calibration_sha256=lock_sha, student_inference_ready=True)
        return report
    except BaseException as error:
        report.update(status='failed', failed_stage=report.get('phase'), error=f'{type(error).__name__}: {error}', measured=False, gpu_measured=False)
        raise
    finally:
        if provider is not None:
            provider.close()
        atomic_json(output / 'trial-report.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Original images/ and optional calibration/ + control/ CSV directories')
    parser.add_argument('--train-csv', type=Path)
    parser.add_argument('--val-csv', type=Path)
    parser.add_argument('--identity-csv', type=Path, help='Optional global image_id,identity_id mapping covering train/val and all protected IDs')
    parser.add_argument('--student-checkpoint', type=Path, help='Eval-only completed owned checkpoint; mutually exclusive with train/val')
    parser.add_argument('--teacher-bundle', type=Path, default=ROOT / 'models/e25')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch only; all evaluation model forwards stay batch1')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--init-backbone', type=Path, help='Explicit local native timm ViT-B checkpoint; no automatic downloads')
    parser.add_argument('--preflight-only', action='store_true', help='Hash inputs and check split exclusions without importing/loading any model')
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 1 <= args.batch_size <= 128 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Positive epochs/learning rate and training batch size 1..128 required')
    if args.student_checkpoint is not None and args.init_backbone is not None:
        parser.error('Eval-only checkpoint cannot use an initial backbone')
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'Student trial failed: {error}\n')
    print(json.dumps({'status': result['status'], 'training_finished': result['training_finished'], 'measured': result['measured']}))


if __name__ == '__main__':
    main()
