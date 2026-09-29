"""Official three-file export: formatting and integrity, without ground-truth access."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


from .retrieval import rank_vectors, retrieval_record, resolve_export_policy, validate_retrieval


def _check_ids(query_ids, gallery_ids):
    if not query_ids or len(gallery_ids) < 10:
        raise ValueError('Required: at least one query and ten gallery images.')
    for values in (query_ids, gallery_ids):
        if any(not isinstance(value, str) or not value or value.strip() != value for value in values):
            raise ValueError('Image IDs must be nonempty strings without surrounding whitespace.')
        if len(set(values)) != len(values):
            raise ValueError('Duplicate image IDs in input order.')


def write_outputs(directory, query_ids, gallery_ids, query_vectors, gallery_vectors, threshold, retrieval=None):
    """Export the exact serving search policy; raw embeddings remain unchanged."""
    directory = Path(directory)
    retrieval = validate_retrieval(retrieval)
    _check_ids(query_ids, gallery_ids)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1].')
    queries = np.asarray(query_vectors, dtype=np.float32)
    gallery = np.asarray(gallery_vectors, dtype=np.float32)
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[0] != len(query_ids) or gallery.shape[0] != len(gallery_ids) or queries.shape[1] != gallery.shape[1]:
        raise ValueError('Embedding dimensions do not match the input order.')
    vectors = np.concatenate([queries, gallery]).astype(np.float32)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Expected finite, nonzero L2-normalized embeddings.')
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'embeddings.npy', vectors, allow_pickle=False)
    with (directory / 'submission.csv').open('w', encoding='utf-8', newline='') as sub, (directory / 'candidates.csv').open('w', encoding='utf-8', newline='') as can:
        submission, candidates = csv.writer(sub), csv.writer(can)
        # Organizer evaluate.py and example_submission.zip require NO header here.
        candidates.writerow(['query_id', 'gallery_id', 'confidence'])
        for query_id, vector in zip(query_ids, queries):
            scores, order, accepted = rank_vectors(vector, gallery, threshold, retrieval)
            submission.writerow([query_id] + [gallery_ids[int(i)] for i in order[:10]])
            for index in accepted:
                # Round-trip float32 scores: fixed decimal rounding could change
                # ties or move a candidate below the calibrated threshold.
                candidates.writerow([query_id, gallery_ids[index], repr(float(scores[index]))])
    (directory / 'embedding_order.json').write_text(json.dumps({'query': query_ids, 'gallery': gallery_ids}, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / 'retrieval.json').write_text(json.dumps(retrieval_record(retrieval, threshold), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return validate_export(directory, query_ids, gallery_ids, threshold=threshold, check_ranking=True, retrieval=retrieval)


def validate_export(directory, query_ids, gallery_ids, *, threshold=None, check_ranking=False, retrieval=None):
    """Reject malformed outputs and replay the declared policy without identity labels."""
    directory = Path(directory)
    retrieval, threshold = resolve_export_policy(directory, retrieval, threshold)
    _check_ids(query_ids, gallery_ids)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Threshold must be in [0, 1].')
    with (directory / 'submission.csv').open(encoding='utf-8-sig', newline='') as stream:
        submission = list(csv.reader(stream))
    if len(submission) != len(query_ids) or [row[0] if row else None for row in submission] != query_ids:
        raise ValueError('submission.csv must have no header and one row per query in input order.')
    gallery_set, query_set = set(gallery_ids), set(query_ids)
    for row in submission:
        if len(row) != 11 or len(set(row[1:])) != 10 or not set(row[1:]) <= gallery_set:
            raise ValueError('Each submission row must contain ten distinct valid gallery IDs.')
    embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False, mmap_mode='r')
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids) + len(gallery_ids) or not 1 <= embeddings.shape[1] <= 16384:
        raise ValueError('embeddings.npy must be float32 [query_count + gallery_count, D].')
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise ValueError('Embeddings contain nonfinite or zero vectors.')
    order_path = directory / 'embedding_order.json'
    if order_path.exists() and json.loads(order_path.read_text(encoding='utf-8')) != {'query': query_ids, 'gallery': gallery_ids}:
        raise ValueError('Embedding order metadata differs from input CSV order.')
    grouped = {qid: [] for qid in query_ids}
    with (directory / 'candidates.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ['query_id', 'gallery_id', 'confidence']:
            raise ValueError('candidates.csv must have the official three-column header.')
        seen = set()
        for row in reader:
            if len(row) != 3 or row[0] not in query_set or row[1] not in gallery_set:
                raise ValueError('Unknown candidate IDs or wrong column count.')
            score = float(row[2])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Candidate confidence must be finite and in [0, 1].')
            if (row[0], row[1]) in seen:
                raise ValueError('Duplicate query-gallery candidate pair.')
            seen.add((row[0], row[1]))
            if threshold is not None and score < threshold:
                raise ValueError('Candidate below calibrated threshold.')
            grouped[row[0]].append((row[1], score))
    for index, pairs in enumerate(grouped.values()):
        if retrieval and (len(pairs) > 1 or (pairs and pairs[0][0] != submission[index][1])):
            raise ValueError('E25 candidates must contain only the accepted final top1.')
        if any(a[1] < b[1] for a, b in zip(pairs, pairs[1:])):
            raise ValueError('Candidates must be sorted by nonincreasing confidence per query.')
    if check_ranking:
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
            raise ValueError('Search export parity requires L2-normalized vectors.')
        gallery = embeddings[len(query_ids):]
        for index, qid in enumerate(query_ids):
            scores, order, accepted = rank_vectors(embeddings[index], gallery, threshold if threshold is not None else 0, retrieval)
            if submission[index][1:] != [gallery_ids[int(i)] for i in order[:10]]:
                raise ValueError('Submission ranking differs from the declared retrieval policy.')
            if threshold is not None:
                expected = [(gallery_ids[i], float(scores[i])) for i in accepted]
                if grouped[qid] != expected:
                    raise ValueError('Candidates differ from calibrated search decisions.')
    return {'valid': True, 'query_count': len(query_ids), 'gallery_count': len(gallery_ids), 'dimension': embeddings.shape[1], 'candidate_count': sum(map(len, grouped.values())), 'rejected_query_count': sum(not pairs for pairs in grouped.values()), 'metrics_measured': False, 'retrieval': retrieval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-csv', type=Path, required=True)
    parser.add_argument('--gallery-csv', type=Path, required=True)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--check-cosine-ranking', action='store_true', help='Legacy: explicitly require cosine search')
    parser.add_argument('--check-ranking', action='store_true', help='Replay the saved retrieval.json policy')
    parser.add_argument('--retrieval-config', type=Path, help='Optional JSON with an explicit retrieval recipe')
    args = parser.parse_args()
    from .service import parse_csv
    try:
        q = parse_csv(args.query_csv.read_bytes(), max_images=1_000_000)
        g = parse_csv(args.gallery_csv.read_bytes(), max_images=1_000_000)
        retrieval = json.loads(args.retrieval_config.read_text(encoding='utf-8')) if args.retrieval_config else None
        if args.check_cosine_ranking and resolve_export_policy(args.output, retrieval, args.threshold)[0] is not None:
            raise ValueError('Export uses reranking; use --check-ranking instead of --check-cosine-ranking.')
        print(json.dumps(validate_export(args.output, [r['image_id'] for r in q], [r['image_id'] for r in g], threshold=args.threshold, check_ranking=args.check_cosine_ranking or args.check_ranking, retrieval=retrieval), ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'Validation failed: {error}\n')


if __name__ == '__main__':
    main()
