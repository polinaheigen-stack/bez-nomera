"""Frozen input and train-only teacher-cache contracts for compact trials."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .evaluation import OFFICIAL_PATH, OFFICIAL_SHA256, validate_ground_truth
from .model_bundle import digest_json, sha256
from .student_trial import preflight as base_preflight, read_rows, verify_contract


def csv_records(path, required):
    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if reader.fieldnames is None or not set(required) <= set(reader.fieldnames):
        raise ValueError('Missing required CSV columns: ' + str(path))
    rows = list(reader)
    if not rows or any(None in row or any(not row.get(k, '').strip() for k in required) for row in rows):
        raise ValueError('Empty or invalid CSV: ' + str(path))
    return rows, hashlib.sha256(raw).hexdigest()


def preflight(dataset, inputs):
    """Keep P3 pixel/identity exclusions, prove labels and dev partition against originals."""
    dataset, inputs = Path(dataset).resolve(), Path(inputs).resolve()
    identity_path = inputs / 'identity-map.csv'
    if not identity_path.is_file():
        raise ValueError('A complete identity-map.csv is mandatory')
    contract, rows, paths = base_preflight(dataset, inputs / 'train.csv', inputs / 'val.csv',
                                          identity_csv=identity_path)
    source_path = dataset / 'train.csv'
    original, original_sha = csv_records(source_path, ('image_id', 'x', 'y', 'w', 'h', 'vehicle_id', 'camera_id'))
    originals = {}
    for record in original:
        key = record['image_id']
        if key in originals:
            raise ValueError('Duplicate original train image ID')
        originals[key] = record
    identities, identity_sha = csv_records(identity_path, ('image_id', 'identity_id'))
    labels = {r['image_id']: r['identity_id'] for r in identities}
    if sha256(identity_path) != identity_sha or identity_sha != contract['files']['identity_mapping']['sha256']:
        raise ValueError('Identity map changed during preflight')
    needed = set().union(*(set(ids) for ids in contract['order'].values()))
    if not needed <= set(originals):
        raise ValueError('Train, dev, calibration and control IDs must come from original train.csv')
    for key, identity in labels.items():
        if key not in originals or identity != originals[key]['vehicle_id']:
            raise ValueError('Identity mapping disagrees with original vehicle_id: ' + key)
    for group in rows.values():
        for row in group:
            source_bbox = tuple(int(originals[row['image_id']][k]) for k in ('x', 'y', 'w', 'h'))
            if tuple(row['bbox']) != source_bbox:
                raise ValueError('Split bbox differs from original train.csv: ' + row['image_id'])
    contract['files']['original_training_labels'] = {'path': str(source_path), 'sha256': original_sha}
    dev = {}
    for kind in ('query', 'gallery'):
        path = inputs / ('dev_' + kind + '.csv')
        digest = sha256(path)
        dev[kind] = read_rows(path, digest)
        contract['files']['dev/' + kind] = {'path': str(path), 'sha256': digest}
    qids, gids = ({r['image_id'] for r in dev[k]} for k in ('query', 'gallery'))
    if qids & gids or qids | gids != set(contract['order']['val']):
        raise ValueError('Dev query/gallery must be a disjoint exact partition of val.csv')
    if len(dev['gallery']) < 10:
        raise ValueError('Dev gallery requires at least ten images')
    val_bboxes = {r['image_id']: tuple(r['bbox']) for r in rows['val']}
    for group in dev.values():
        for row in group:
            if tuple(row['bbox']) != val_bboxes[row['image_id']]:
                raise ValueError('Dev bbox differs from val.csv')
    gt_path = inputs / 'dev_ground_truth.csv'
    gt, gt_sha = csv_records(gt_path, ('image_id', 'vehicle_id', 'camera_id', 'split'))
    validate_ground_truth(gt_path, qids, gids)
    for item in gt:
        expected = originals[item['image_id']]
        if item['vehicle_id'] != expected['vehicle_id'] or item['camera_id'] != expected['camera_id']:
            raise ValueError('Dev ground truth disagrees with original labels: ' + item['image_id'])
    contract['files']['dev/ground_truth'] = {'path': str(gt_path), 'sha256': gt_sha}
    for kind in ('query', 'gallery'):
        key = 'dev/' + kind
        rows[key] = dev[kind]
        contract['order'][key] = [r['image_id'] for r in dev[kind]]
        contract['bboxes'][key] = [r['bbox'] for r in dev[kind]]
        contract['counts'][key] = len(dev[kind])
    if sha256(OFFICIAL_PATH) != OFFICIAL_SHA256:
        raise ValueError('Official evaluator source differs')
    contract['files']['official_evaluator'] = {'path': str(OFFICIAL_PATH), 'sha256': OFFICIAL_SHA256}
    contract.update(schema_version=2, identity_mapping_verified_against_original=True,
                    independent_quality_claim_allowed=False,
                    dev_selection='official raw cosine mAP@10; no re-ranking; only original dev split',
                    training_labels={key: {'vehicle_id': originals[key]['vehicle_id'],
                                          'camera_id': originals[key]['camera_id']} for key in contract['order']['train']})
    contract['notes'] = ['Original train.csv verifies identity IDs, camera labels and bboxes.',
                         'Student training uses train only; checkpoints are selected on dev raw mAP@10.',
                         'Calibration/control are excluded by identity, image ID and original file bytes.',
                         'Historical E25 and dev/control provenance do not establish an independent quality holdout.']
    verify_contract(contract)
    return contract, rows, paths, {key: originals[key]['vehicle_id'] for key in contract['order']['train']}


def validate_vectors(vectors, count):
    if (vectors.dtype != np.float32 or vectors.shape != (count, 1024)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4)):
        raise ValueError('Teacher cache must contain finite normalized float32 [train_count,1024]')
    return vectors


def read_teacher_cache(cache, rows, contract, expected_teacher_sha, receipt=None):
    """Reuse P3 flat caches by content equivalence, never by stale absolute paths."""
    cache = Path(cache)
    metadata_path = cache / 'cache.json' if cache.is_dir() else cache
    metadata_sha = sha256(metadata_path)
    record = json.loads(metadata_path.read_text(encoding='utf-8'))
    if record.get('status') != 'completed' or (record.get('teacher') or {}).get('sha256') != expected_teacher_sha:
        raise ValueError('Teacher cache status/model hash differs from the selected E25 bundle')
    ordered = record.get('ordered_inputs', [])
    if len(ordered) != len(rows):
        raise ValueError('Teacher cache must contain exactly train rows, no dev/evaluation targets')
    expected = {row['image_id']: row for row in rows}
    if len(expected) != len(rows) or len({item.get('image_id') for item in ordered}) != len(rows):
        raise ValueError('Duplicate teacher cache or training IDs')
    indices = {}
    for index, item in enumerate(ordered):
        key = item.get('image_id')
        if (key not in expected or tuple(item.get('bbox', [])) != tuple(expected[key]['bbox'])
                or item.get('sha256') != contract['images'][key]['sha256']):
            raise ValueError('Teacher cache input ID/bbox/bytes differs: ' + str(key))
        indices[key] = index
    path = metadata_path.with_name('embeddings.npy')
    if sha256(path) != record.get('embeddings_sha256'):
        raise ValueError('Teacher descriptor file checksum differs')
    vectors = validate_vectors(np.load(path, allow_pickle=False), len(rows))
    if sha256(metadata_path) != metadata_sha or sha256(path) != record['embeddings_sha256']:
        raise ValueError('Teacher cache changed during reading')
    result = np.ascontiguousarray(vectors[[indices[row['image_id']] for row in rows]])
    proof = {'metadata_sha256': metadata_sha, 'embeddings_sha256': record['embeddings_sha256'],
             'teacher_sha256': expected_teacher_sha, 'training_order_sha256': digest_json([r['image_id'] for r in rows]),
             'path_relocation_allowed': True, 'validation': 'exact IDs, bboxes, image bytes, model hash and normalized descriptors',
             'teacher_train_only': True}
    if receipt is not None:
        receipt_sha = sha256(receipt)
        attestation = json.loads(Path(receipt).read_text(encoding='utf-8'))
        if (attestation.get('teacher_sha256') != expected_teacher_sha
                or attestation.get('cache_sha256') != metadata_sha
                or attestation.get('embeddings_sha256') != record['embeddings_sha256']):
            raise ValueError('Teacher cache receipt disagrees with actual cached files/model')
        if sha256(receipt) != receipt_sha:
            raise ValueError('Teacher cache receipt changed during reading')
        proof['receipt_sha256'] = receipt_sha
    return result, proof
