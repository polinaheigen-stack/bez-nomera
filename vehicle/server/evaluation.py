"""Run the unchanged organizer evaluator on actual exported files and supplied labels."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .provider import sha256
from .service import atomic_json, parse_csv, ServiceError
from .validation import validate_export

OFFICIAL_SHA256 = '655c71db8c2e4d2cd7680c40c768afacfdffff360401111c1a46df921551ffa3'
OFFICIAL_PATH = Path(__file__).resolve().parents[2] / 'docs' / 'sources' / 'falcon-evaluation-2026-09-21' / 'evaluate.py'


def validate_ground_truth(path, query_ids, gallery_ids):
    contents = path if isinstance(path, bytes) else Path(path).read_bytes()
    with io.StringIO(contents.decode('utf-8-sig'), newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image_id', 'vehicle_id', 'camera_id', 'split'}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError('Ground truth requires image_id,vehicle_id,camera_id,split.')
        ids = {'query': set(), 'gallery': set()}
        for row in reader:
            if None in row or any(not row.get(key, '').strip() for key in required) or row['split'] not in ids:
                raise ValueError('Ground truth has invalid or missing labels.')
            split = row['split']
            if row['image_id'] in ids[split]:
                raise ValueError('Duplicate ground-truth image ID in split.')
            ids[split].add(row['image_id'])
        if ids != {'query': set(query_ids), 'gallery': set(gallery_ids)}:
            raise ValueError('Ground truth IDs must exactly match the exported query/gallery sets.')


def json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    return value


def evaluate_export(output, query_csv, gallery_csv, ground_truth, *, evaluator=OFFICIAL_PATH, current_run=False, timeout_seconds=120):
    output, query_csv, gallery_csv, ground_truth, evaluator = map(Path, (output, query_csv, gallery_csv, ground_truth, evaluator))
    input_paths = {'query': query_csv, 'gallery': gallery_csv, 'ground_truth': ground_truth, 'evaluator': evaluator}
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    snapshots = {name: hashlib.sha256(contents).hexdigest() for name, contents in input_bytes.items()}
    if snapshots['evaluator'] != OFFICIAL_SHA256:
        raise ValueError('Organizer evaluate.py hash differs from the preserved original.')
    for name in ('submission.csv', 'candidates.csv', 'embeddings.npy'):
        input_paths[name] = output / name
        snapshots[name] = sha256(input_paths[name])
    provenance = output / 'provenance.json'
    provenance_bytes = provenance.read_bytes() if provenance.is_file() else None
    if provenance_bytes is not None:
        input_paths['provenance'] = provenance
        snapshots['provenance'] = hashlib.sha256(provenance_bytes).hexdigest()
    order_path = output / 'embedding_order.json'
    if order_path.is_file():
        input_paths['embedding_order'] = order_path
        snapshots['embedding_order'] = sha256(order_path)
    retrieval_path = output / 'retrieval.json'
    if retrieval_path.is_file():
        input_paths['retrieval'] = retrieval_path
        snapshots['retrieval'] = sha256(retrieval_path)
    def verify_inputs():
        for name, path in input_paths.items():
            if not path.is_file() or sha256(path) != snapshots[name]:
                raise ValueError(f'Evaluation inputs changed during evaluation: {name}')
        if provenance_bytes is None and provenance.exists():
            raise ValueError('Evaluation provenance appeared during evaluation.')
    query = parse_csv(input_bytes['query'], max_images=1_000_000)
    gallery = parse_csv(input_bytes['gallery'], max_images=1_000_000)
    qids, gids = [r['image_id'] for r in query], [r['image_id'] for r in gallery]
    validation = validate_export(output, qids, gids)
    validate_ground_truth(input_bytes['ground_truth'], qids, gids)
    verify_inputs()
    # No full-gallery re-ranking/refill: pass the exact submission.csv to the official script.
    with tempfile.TemporaryDirectory(prefix='.evaluation-', dir=output) as temporary:
        raw = Path(temporary) / 'metrics.json'
        command = [sys.executable, str(evaluator), '--gt', str(ground_truth), '--submission', str(output / 'submission.csv'),
                   '--candidates', str(output / 'candidates.csv'), '--embeddings', str(output / 'embeddings.npy'),
                   '--query', str(query_csv), '--gallery', str(gallery_csv), '--json', str(raw)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            (output / 'evaluation.log').write_text(f'Official evaluator exceeded {timeout_seconds} seconds.\n', encoding='utf-8')
            raise RuntimeError('Official evaluation timed out; see evaluation.log.') from error
        (output / 'evaluation.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        verify_inputs()
        if result.returncode or not raw.is_file():
            raise RuntimeError('Official evaluation failed; see evaluation.log. Install the evaluation dependencies, including pandas.')
        official = json.loads(raw.read_text(encoding='utf-8'))
    report = {'measured': True, 'scope': 'provided_labeled_split', 'evaluator_sha256': OFFICIAL_SHA256,
              'ground_truth_sha256': snapshots['ground_truth'], 'input_csv_sha256': {'query': snapshots['query'], 'gallery': snapshots['gallery']},
              'export_sha256': {name: snapshots[name] for name in ('submission.csv', 'candidates.csv', 'embeddings.npy')},
              'validation': validation, 'official': json_finite(official),
              'notes': ['Official ranking removes same-vehicle/same-camera entries only from the submitted top-10; it does not refill from embeddings.',
                        'Official candidate correctness compares vehicle ID for the top candidate once a cross-camera positive exists; that top candidate may be same-camera.',
                        'PR-AUC is calculated from returned candidates and can change with the rejection threshold. Undefined metrics are null.',
                        'This result applies only to the supplied labels, not to the undisclosed organizer test set.',
                        'Optional full_mAP and mINP are computed from raw embeddings; they are not reranked top-10 metrics.']}
    ui = None
    if provenance_bytes is not None:
        proof = json.loads(provenance_bytes.decode('utf-8'))
        if proof.get('status') != 'completed':
            raise ValueError('Provenance does not describe a completed inference run.')
        for name, digest in report['export_sha256'].items():
            if proof.get('files_sha256', {}).get(name) != digest:
                raise ValueError('Export has changed since inference; refusing to bind metrics to model provenance.')
        if proof.get('input_csv_sha256') and proof['input_csv_sha256'] != report['input_csv_sha256']:
            raise ValueError('Evaluation input CSV differs from the inference CSV.')
        if proof.get('input_order') != {'query': qids, 'gallery': gids}:
            raise ValueError('Provenance input order differs from evaluation inputs.')
        model = proof.get('model') or {}
        identity = model.get('inference_fingerprint') or proof.get('inference_fingerprint')
        if not re.fullmatch('[0-9a-f]{64}', str(model.get('sha256', ''))) or not proof.get('run_id'):
            raise ValueError('Provenance must identify the model weights and inference run.')
        if identity and not re.fullmatch('[0-9a-f]{64}', str(identity)):
            raise ValueError('Invalid inference fingerprint in provenance.')
        if model.get('inference_fingerprint') and proof.get('inference_fingerprint') and model['inference_fingerprint'] != proof['inference_fingerprint']:
            raise ValueError('Conflicting inference fingerprints in provenance.')
        ranking, candidates = report['official']['ranking'], report['official']['candidates']
        report.update(run_id=proof['run_id'], model=model, calibration_sha256=proof.get('calibration_sha256'),
                      threshold=proof.get('threshold'), inference_fingerprint=identity,
                      retrieval=proof.get('retrieval'), retrieval_fingerprint=proof.get('retrieval_fingerprint'))
        ui = {'mode': proof.get('mode', 'prod'), 'measured': True, 'model': proof.get('model'),
              'calibration_sha256': proof.get('calibration_sha256'), 'run_id': proof.get('run_id'),
              'ground_truth_sha256': report['ground_truth_sha256'],
              'dataset': f'Размеченная контрольная выборка: {len(qids)} запросов, {len(gids)} снимков галереи. SHA-256: ' + report['ground_truth_sha256'],
              'hardware': model.get('device', 'не указано') + '; скорость GPU этой проверкой не измерялась',
              'metrics': {'map_at_10': ranking['mAP@10'], 'rank_1': ranking['Rank-1'], 'rank_5': ranking['Rank-5'], 'micro_f1': candidates['F1'], 'tnr': candidates['TNR']},
              'notes': [
                  'Показатели рассчитаны неизменённым официальным оценщиком на отдельной размеченной выборке. Это не оценка последнего пользовательского пакета и не закрытого теста организаторов.',
                  'Если выборка участвовала в выборе модели, результат нельзя считать независимой проверкой качества.',
                  'Rank-1 и mAP исключают пары одной машины с одной камеры из поданной десятки без её дополнения. F1 использует отдельное правило официального оценщика.',
                  f'Запросов без искомой машины в галерее: {candidates["n_openset_queries"]}. Для них показатель TNR отражает долю правильных отказов.',
                  'Замеры скорости, оперативной памяти и памяти GPU выполняются отдельно; отсутствующие измерения здесь не подставляются.']}
        if current_run:
            report['scope'] = 'exact_web_run_with_supplied_labels'
            ui['dataset'] = f'Этот запуск: {len(qids)} запросов, {len(gids)} снимков галереи. SHA-256 разметки: ' + report['ground_truth_sha256']
            ui['notes'][0] = 'Качество рассчитано неизменённым официальным оценщиком по экспорту именно этого запуска и загруженным правильным ответам. Метки не передавались модели и не меняли поиск.'
            report['notes'].append('Labels are evaluation-only. This report is bound to this exact web run and its unchanged exported predictions.')
    # No measured=true artifact is changed until all export/provenance checks pass.
    verify_inputs()
    atomic_json(output / 'metrics.json', report)
    if ui is not None:
        atomic_json(output / 'evaluation-report.json', ui)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed batch export directory')
    parser.add_argument('--query-csv', type=Path, required=True)
    parser.add_argument('--gallery-csv', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True, help='Labels are evaluation-only; never supplied to inference')
    args = parser.parse_args()
    try:
        report = evaluate_export(args.output, args.query_csv, args.gallery_csv, args.gt)
    except (OSError, ValueError, RuntimeError, ServiceError) as error:
        parser.exit(2, f'Evaluation failed: {error}\n')
    print(json.dumps(report['official'], ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
