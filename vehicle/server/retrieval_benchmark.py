"""Measure frozen-gallery retrieval only; never label this as model/GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .retrieval import rank_vectors, validate_retrieval


def measure(export, output):
    export, output = Path(export), Path(output)
    if output.exists():
        raise ValueError('Use a new output report.')
    policy = json.loads((export / 'retrieval.json').read_text(encoding='utf-8'))
    order = json.loads((export / 'embedding_order.json').read_text(encoding='utf-8'))
    vectors = np.load(export / 'embeddings.npy', allow_pickle=False)
    recipe = validate_retrieval(policy['retrieval'])
    if recipe is None or vectors.dtype != np.float32 or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('E25 policy and finite float32 vectors are required.')
    threshold = float(policy['threshold'])
    queries, gallery = vectors[:len(order['query'])], vectors[len(order['query']):]
    if not len(queries) or len(gallery) != len(order['gallery']):
        raise ValueError('Invalid embedding order.')
    for index in range(10):
        rank_vectors(queries[index % len(queries)], gallery, threshold, recipe)
    samples = []
    for query in queries:
        start = time.perf_counter()
        rank_vectors(query, gallery, threshold, recipe)
        samples.append((time.perf_counter() - start) * 1000)
    report = {'status': 'completed', 'scope': 'CPU retrieval over existing embeddings; excludes image decode, neural networks and I/O',
              'official_measurement': False, 'gallery_count': len(gallery), 'query_count': len(queries),
              'warmup_calls': 10, 'median_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95)),
              'samples_ms': samples, 'platform': platform.platform(), 'processor': platform.processor(),
              'retrieval': recipe, 'threshold': threshold,
              'embedding_sha256': hashlib.sha256((export / 'embeddings.npy').read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.export, args.output)))
