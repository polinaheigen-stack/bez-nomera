"""Compare a reference export with the integrated model; no training or label access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .provider import sha256
from .service import parse_csv, ServiceError
from .validation import validate_export


def compare_exports(reference, actual, query_ids, gallery_ids, *, atol=1e-5, rtol=1e-4):
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError('Tolerances must be finite and nonnegative.')
    reference, actual = Path(reference), Path(actual)
    names = ('submission.csv', 'candidates.csv', 'embeddings.npy')
    original_hashes = {str(directory): {n: sha256(directory / n) for n in names} for directory in (reference, actual)}
    for directory in (reference, actual):
        validate_export(directory, query_ids, gallery_ids)
    expected = np.load(reference / 'embeddings.npy', allow_pickle=False)
    observed = np.load(actual / 'embeddings.npy', allow_pickle=False)
    same_shape = expected.shape == observed.shape
    checks = {'shape': same_shape,
              'embeddings': bool(same_shape and np.allclose(expected, observed, atol=atol, rtol=rtol))}
    max_difference = float(np.max(np.abs(expected - observed))) if same_shape else None

    def rows(directory, name):
        with (directory / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.reader(stream))

    policies = [json.loads((d / 'retrieval.json').read_text(encoding='utf-8')) if (d / 'retrieval.json').is_file() else None for d in (reference, actual)]
    if all(policy is not None for policy in policies):
        checks['retrieval_policy'] = policies[0] == policies[1]
    checks['ranking'] = rows(reference, 'submission.csv') == rows(actual, 'submission.csv')
    ref_candidates = rows(reference, 'candidates.csv')[1:]
    actual_candidates = rows(actual, 'candidates.csv')[1:]
    checks['accepted_candidates_and_order'] = [r[:2] for r in ref_candidates] == [r[:2] for r in actual_candidates]
    checks['refusals'] = {r[0] for r in ref_candidates} == {r[0] for r in actual_candidates}
    checks['scores'] = bool(checks['accepted_candidates_and_order'] and np.allclose(
        [float(r[2]) for r in ref_candidates], [float(r[2]) for r in actual_candidates], atol=atol, rtol=rtol))
    for directory in (reference, actual):
        if {n: sha256(directory / n) for n in names} != original_hashes[str(directory)]:
            raise ValueError('An export changed during comparison; rerun on immutable inputs.')
    return {'schema_version': 1, 'passed': all(checks.values()), 'quality_claim': False,
            'checks': checks, 'atol': atol, 'rtol': rtol, 'max_embedding_absolute_difference': max_difference,
            'reference_sha256': original_hashes[str(reference)],
            'actual_sha256': original_hashes[str(actual)],
            'note': 'Numerical portability check only. Ranking/refusal changes require review, including close ties.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--query-csv', type=Path, required=True)
    parser.add_argument('--gallery-csv', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Use a new report path; existing evidence is not overwritten.')
    try:
        query_bytes, gallery_bytes = args.query_csv.read_bytes(), args.gallery_csv.read_bytes()
        hashes = {'query_csv': hashlib.sha256(query_bytes).hexdigest(), 'gallery_csv': hashlib.sha256(gallery_bytes).hexdigest()}
        query = [r['image_id'] for r in parse_csv(query_bytes, max_images=1_000_000)]
        gallery = [r['image_id'] for r in parse_csv(gallery_bytes, max_images=1_000_000)]
        report = compare_exports(args.reference, args.actual, query, gallery, atol=args.atol, rtol=args.rtol)
        if hashes != {'query_csv': sha256(args.query_csv), 'gallery_csv': sha256(args.gallery_csv)}:
            raise ValueError('An input CSV changed during comparison.')
        report['inputs_sha256'] = hashes
    except (ValueError, OSError, ServiceError) as error:
        report = {'passed': False, 'quality_claim': False, 'error': str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
