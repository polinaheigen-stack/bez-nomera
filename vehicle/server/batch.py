"""Offline original-image and bbox inference, without HTTP, database or archive.

python -m vehicle.server.batch --images /data/images --query-csv /data/query.csv \
    --gallery-csv /data/gallery.csv --output /out --batch-size 1
"""
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from PIL import Image

from .provider import ProductionProvider, sha256
from .service import ServiceError, atomic_json, normalized, now, parse_csv
from .validation import write_outputs
from .retrieval import score_definition, validate_retrieval


def find_images(directory, rows):
    """Resolve original images once; flat directory and one file per requested ID."""
    by_id = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            by_id.setdefault(path.stem, []).append(path)
    paths = {}
    for row in rows:
        matches = by_id.get(row['image_id'], [])
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one JPEG/PNG for {row['image_id']}")
        paths[row['image_id']] = matches[0]
    return paths


@contextmanager
def decoded_images(rows, paths):
    """Read original pixels. EXIF orientation is not applied silently."""
    with ExitStack() as stack:
        images = []
        for row in rows:
            image = stack.enter_context(Image.open(paths[row['image_id']]))
            if image.format not in ('JPEG', 'PNG') or getattr(image, 'n_frames', 1) != 1:
                raise ValueError(f"Expected a single JPEG/PNG frame: {row['image_id']}")
            width, height = image.size
            x, y, w, h = row['bbox']
            if width * height > 40_000_000 or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError(f"Invalid image dimensions or bbox: {row['image_id']}")
            image.load()
            images.append(image)
        yield images


def infer_batch(provider, rows, paths):
    if getattr(provider, 'supports_path_extraction', False):
        # FastExtractor owns original decode and parallel preprocessing. Header
        # reads here record dimensions only; no second full decode is performed.
        inputs = []
        for row in rows:
            with Image.open(paths[row['image_id']]) as image:
                inputs.append({'image_id': row['image_id'], 'bbox': list(row['bbox']),
                               'width': image.width, 'height': image.height})
        vectors = provider.embed_paths([paths[r['image_id']] for r in rows],
                                       [r['bbox'] for r in rows], image_ids=[r['image_id'] for r in rows])
        if len(vectors) != len(rows):
            raise ValueError('Model returned a different number of embeddings than inputs.')
        dimension = (provider.model or {}).get('dimension')
        return np.stack([normalized(v, dimension) for v in vectors]).astype(np.float32), inputs
    with decoded_images(rows, paths) as images:
        if callable(getattr(provider, 'embed_batch', None)):
            vectors = provider.embed_batch(images, [row['bbox'] for row in rows], image_ids=[row['image_id'] for row in rows])
        else:
            vectors = [provider.embed(image, row['bbox'], image_id=row['image_id']) for image, row in zip(images, rows)]
        if len(vectors) != len(rows):
            raise ValueError('Model returned a different number of embeddings than inputs.')
        dimension = (provider.model or {}).get('dimension')
        values = np.stack([normalized(vector, dimension) for vector in vectors]).astype(np.float32)
        inputs = [{'image_id': row['image_id'], 'bbox': list(row['bbox']), 'width': image.width, 'height': image.height} for row, image in zip(rows, images)]
        return values, inputs


def fingerprint(provider):
    model = provider.model or {}
    return getattr(provider, 'inference_fingerprint', None) or model.get('inference_fingerprint') or model.get('sha256')


def run_batch(provider, images, query_csv, gallery_csv, output, *, batch_size=1):
    """Direct paths avoid HTTP upload limits and copying several GiB of source images."""
    if not provider.available:
        raise ValueError(provider.reason or 'Model is unavailable.')
    if not 1 <= batch_size <= 32:
        raise ValueError('Batch size must be between 1 and 32.')
    if provider.threshold is None or not math.isfinite(provider.threshold) or not 0 <= provider.threshold <= 1:
        raise ValueError('A calibrated [0, 1] threshold is required.')
    images, query_csv, gallery_csv, output = map(Path, (images, query_csv, gallery_csv, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be empty.')
    query_bytes, gallery_bytes = query_csv.read_bytes(), gallery_csv.read_bytes()
    query = parse_csv(query_bytes, max_images=1_000_000)
    gallery = parse_csv(gallery_bytes, max_images=1_000_000)
    csv_hashes = {'query': hashlib.sha256(query_bytes).hexdigest(), 'gallery': hashlib.sha256(gallery_bytes).hexdigest()}
    if len(gallery) < 10:
        raise ValueError('At least ten gallery images are required.')
    paths = find_images(images, query + gallery)
    output.mkdir(parents=True, exist_ok=True)
    events = []
    def event(name, **details):
        entry = {'time': now(), 'event': name, **details}
        events.append(entry)
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
    identity, calibration = fingerprint(provider), getattr(provider, 'calibration_sha256', None)
    model, threshold = deepcopy(provider.model), float(provider.threshold)
    retrieval = validate_retrieval(getattr(provider, 'retrieval', None))
    retrieval_fingerprint = getattr(provider, 'retrieval_fingerprint', None)
    proof = {'contract_version': '1', 'mode': 'prod', 'run_id': uuid.uuid4().hex, 'status': 'running', 'model': model,
             'inference_fingerprint': identity, 'calibration_sha256': calibration, 'threshold': threshold,
             'score_definition': score_definition(retrieval), 'retrieval': retrieval, 'retrieval_fingerprint': retrieval_fingerprint, 'input_order': {'query': [r['image_id'] for r in query], 'gallery': [r['image_id'] for r in gallery]},
             'input_csv_sha256': csv_hashes, 'inputs': {}, 'files_sha256': {},
             'batch_size': batch_size, 'metrics_measured': False,
             'notes': ['Ranking uses only each query and its frozen gallery.', 'Timing includes file reads and inference but is not an official GPU benchmark.']}
    started = time.perf_counter()
    event('run_started', query_count=len(query), gallery_count=len(gallery), batch_size=batch_size)
    try:
        matrices = {}
        for split, rows in (('gallery', gallery), ('query', query)):
            chunks, records = [], []
            for begin in range(0, len(rows), batch_size):
                if fingerprint(provider) != identity or getattr(provider, 'calibration_sha256', None) != calibration or provider.threshold != threshold or getattr(provider, 'retrieval', None) != retrieval or getattr(provider, 'retrieval_fingerprint', None) != retrieval_fingerprint:
                    raise ValueError('Model or calibration changed during the run.')
                selected = rows[begin:begin + batch_size]
                before = {row['image_id']: sha256(paths[row['image_id']]) for row in selected}
                tick = time.perf_counter()
                vectors, inputs = infer_batch(provider, selected, paths)
                elapsed = time.perf_counter() - tick
                for record in inputs:
                    image_id = record['image_id']
                    if sha256(paths[image_id]) != before[image_id]:
                        raise ValueError(f'Input image changed during inference: {image_id}')
                    record['sha256'] = before[image_id]
                chunks.append(vectors)
                records.extend(inputs)
                event('batch_completed', split=split, processed=begin + len(selected), duration_ms=round(elapsed * 1000, 3))
            matrices[split] = np.concatenate(chunks)
            proof['inputs'][split] = records
        def verify_source_csvs():
            if sha256(query_csv) != csv_hashes['query'] or sha256(gallery_csv) != csv_hashes['gallery']:
                raise ValueError('Input CSV changed during inference; export is not published.')
        verify_source_csvs()
        with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as temporary:
            validation = write_outputs(temporary, proof['input_order']['query'], proof['input_order']['gallery'], matrices['query'], matrices['gallery'], threshold, retrieval=retrieval)
            verify_source_csvs()
            for name in ('submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json'):
                (Path(temporary) / name).replace(output / name)
                proof['files_sha256'][name] = sha256(output / name)
        proof.update(status='completed', duration_seconds=round(time.perf_counter() - started, 6), validation=validation)
        event('run_completed', duration_seconds=proof['duration_seconds'])
        proof['events'] = events
        atomic_json(output / 'provenance.json', proof)
        atomic_json(output / 'validation.json', validation)
        return proof
    except Exception as error:
        event('run_failed', error=str(error))
        proof.update(status='failed', error=str(error), events=events)
        atomic_json(output / 'provenance.json', proof)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--query-csv', required=True, type=Path)
    parser.add_argument('--gallery-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', default=1, type=int)
    args = parser.parse_args()
    try:
        result = run_batch(ProductionProvider(), args.images, args.query_csv, args.gallery_csv, args.output, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError, ServiceError) as error:
        parser.exit(2, f'Inference failed: {error}\n')
    print(json.dumps({'status': result['status'], 'query_count': len(result['input_order']['query']), 'gallery_count': len(result['input_order']['gallery']), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
