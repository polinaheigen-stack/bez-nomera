"""Two complete production batch runs in separate processes, then export parity.

This checks repeatability on the supplied inputs, not model quality or official
GPU latency. Missing weights or a failed run never produce a successful report.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from .batch import find_images
from .parity import compare_exports
from .provider import sha256
from .service import atomic_json, now, parse_csv


def runtime_versions():
    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for package in ('torch', 'torchvision', 'numpy', 'Pillow', 'timm', 'transformers', 'safetensors'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def server_hashes():
    root = Path(__file__).resolve().parent
    return {path.name: sha256(path) for path in sorted(root.glob('*.py'))}


def verify_run(directory, proof, expected_csv, expected_inputs, batch_size):
    if proof.get('status') != 'completed' or proof.get('mode') != 'prod':
        raise ValueError('Batch did not report a completed production run.')
    if proof.get('batch_size') != batch_size or proof.get('input_csv_sha256') != expected_csv:
        raise ValueError('Batch size or input CSV identity differs from the requested run.')
    for split, records in expected_inputs.items():
        if proof.get('input_order', {}).get(split) != [r['image_id'] for r in records]:
            raise ValueError('Batch input order differs from the input snapshot.')
        actual = [{key: record.get(key) for key in ('image_id', 'bbox', 'sha256')}
                  for record in proof.get('inputs', {}).get(split, [])]
        if actual != records:
            raise ValueError('Batch image/bbox identity differs from the input snapshot.')
    for name in ('submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json'):
        if sha256(directory / name) != proof.get('files_sha256', {}).get(name):
            raise ValueError(f'Export does not match its provenance: {name}')
    for key in ('inference_fingerprint', 'calibration_sha256'):
        if not proof.get(key):
            raise ValueError(f'Production provenance is missing {key}.')
    if not proof.get('model', {}).get('sha256'):
        raise ValueError('Production provenance is missing the weight hash.')


def run_repeatability(images, query_csv, gallery_csv, output, *, batch_size=1,
                      atol=1e-5, rtol=1e-4, timeout_seconds=3600):
    """Always launch the existing batch CLI; no alternative inference path."""
    if not 1 <= batch_size <= 32:
        raise ValueError('Batch size must be between 1 and 32.')
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError('Tolerances must be finite and nonnegative.')
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('Timeout must be finite and positive.')
    images, query_csv, gallery_csv, output = [Path(p).resolve() for p in (images, query_csv, gallery_csv, output)]
    if output.exists():
        raise ValueError('Use a new output directory; previous evidence is never overwritten.')
    output.mkdir(parents=True)
    report_path = output / 'repeatability.json'
    report = {'schema_version': 1, 'status': 'running', 'passed': False, 'quality_claim': False,
              'official_measurement': False, 'started_at': now(), 'batch_size': batch_size,
              'versions': runtime_versions(), 'server_sha256': server_hashes(),
              'inputs': None, 'input_csv_sha256': None, 'runs': [], 'parity': None,
              'protocol': {'runs': 2, 'new_process_each_run': True, 'same_batch_size': True,
                           'atol': atol, 'rtol': rtol, 'timeout_seconds_per_run': timeout_seconds,
                           'walltime_includes': ['process_start', 'imports', 'model_load', 'input_validation',
                                                 'image_reads', 'inference', 'gallery_search', 'export', 'evidence_write'],
                           'walltime_excludes': ['parent_input_hashing', 'parent_parity_comparison'],
                           'os_file_cache': 'not cleared; second run may benefit from OS file cache'},
              'note': 'Two fresh production processes on identical supplied inputs; no hidden-test quality claim.'}
    atomic_json(report_path, report)
    started = time.perf_counter()
    try:
        csv_bytes = {'query': query_csv.read_bytes(), 'gallery': gallery_csv.read_bytes()}
        rows = {split: parse_csv(value, max_images=1_000_000) for split, value in csv_bytes.items()}
        csv_hashes = {split: hashlib.sha256(value).hexdigest() for split, value in csv_bytes.items()}
        paths = find_images(images, rows['query'] + rows['gallery'])
        expected = {split: [{'image_id': row['image_id'], 'bbox': list(row['bbox']),
                             'sha256': sha256(paths[row['image_id']])} for row in records]
                    for split, records in rows.items()}
        report.update(inputs=expected, input_csv_sha256=csv_hashes)

        def verify_sources():
            if {'query': sha256(query_csv), 'gallery': sha256(gallery_csv)} != csv_hashes:
                raise ValueError('An input CSV changed between or during the two runs.')
            if any(sha256(paths[r['image_id']]) != r['sha256'] for records in expected.values() for r in records):
                raise ValueError('An input image changed between or during the two runs.')
            if server_hashes() != report['server_sha256']:
                raise ValueError('Server code changed during the repeatability check.')

        proofs = []
        for number in (1, 2):
            verify_sources()
            directory = output / f'run-{number}'
            command = [sys.executable, '-m', 'vehicle.server.batch', '--images', str(images),
                       '--query-csv', str(query_csv), '--gallery-csv', str(gallery_csv),
                       '--output', str(directory), '--batch-size', str(batch_size)]
            run = {'number': number, 'status': 'running', 'command': command, 'started_at': now(),
                   'pid': None, 'returncode': None, 'wall_seconds': None,
                   'stdout': f'run-{number}.stdout.log', 'stderr': f'run-{number}.stderr.log'}
            report['runs'].append(run)
            atomic_json(report_path, report)
            tick = time.perf_counter()
            try:
                with (output / run['stdout']).open('wb') as stdout, (output / run['stderr']).open('wb') as stderr:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                               cwd=Path(__file__).resolve().parents[2])
                    run['pid'] = process.pid
                    try:
                        run['returncode'] = process.wait(timeout=timeout_seconds)
                    except BaseException:
                        process.kill()
                        process.wait()
                        run['returncode'] = process.returncode
                        raise
                run['wall_seconds'] = time.perf_counter() - tick
                if run['returncode'] != 0:
                    raise ValueError(f'Run {number} exited with code {run["returncode"]}; see {run["stderr"]}.')
                proof_path = directory / 'provenance.json'
                proof_bytes = proof_path.read_bytes()
                proof = json.loads(proof_bytes)
                verify_run(directory, proof, csv_hashes, expected, batch_size)
                verify_sources()
                run.update(status='completed', provenance_sha256=hashlib.sha256(proof_bytes).hexdigest(),
                           model=proof['model'], inference_fingerprint=proof['inference_fingerprint'],
                           calibration_sha256=proof['calibration_sha256'], threshold=proof['threshold'],
                           files_sha256=proof['files_sha256'], batch_duration_seconds=proof.get('duration_seconds'))
                proofs.append(proof)
            except BaseException as error:
                run.update(status='failed', error=str(error) or type(error).__name__)
                raise
            finally:
                if run['wall_seconds'] is None:
                    run['wall_seconds'] = time.perf_counter() - tick
                run['finished_at'] = now()
                atomic_json(report_path, report)
        for key in ('model', 'inference_fingerprint', 'calibration_sha256', 'threshold', 'score_definition', 'retrieval', 'retrieval_fingerprint'):
            if proofs[0].get(key) != proofs[1].get(key):
                raise ValueError(f'The two runs used different {key}.')
        parity = compare_exports(output / 'run-1', output / 'run-2',
                                 [r['image_id'] for r in rows['query']], [r['image_id'] for r in rows['gallery']],
                                 atol=atol, rtol=rtol)
        verify_sources()
        for number, proof in enumerate(proofs, 1):
            directory = output / f'run-{number}'
            if sha256(directory / 'provenance.json') != report['runs'][number - 1]['provenance_sha256']:
                raise ValueError('Batch provenance changed during comparison.')
            verify_run(directory, proof, csv_hashes, expected, batch_size)
        atomic_json(output / 'parity.json', parity)
        report['parity'] = {'file': 'parity.json', 'sha256': sha256(output / 'parity.json'),
                            'passed': parity['passed'], 'checks': parity['checks']}
        report.update(status='completed' if parity['passed'] else 'failed', passed=parity['passed'])
        if not parity['passed']:
            report['error'] = 'The two production exports differ; inspect parity.json.'
    except Exception as error:
        report.update(status='failed', error=str(error))
    except BaseException as error:
        report.update(status='interrupted', error=type(error).__name__)
        raise
    finally:
        report.update(finished_at=now(), total_wall_seconds=time.perf_counter() - started)
        atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--query-csv', type=Path, required=True)
    parser.add_argument('--gallery-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for both runs and evidence')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--timeout-seconds', type=float, default=3600, help='Maximum wall time of each child process')
    args = parser.parse_args()
    try:
        report = run_repeatability(args.images, args.query_csv, args.gallery_csv, args.output,
                                   batch_size=args.batch_size, atol=args.atol, rtol=args.rtol,
                                   timeout_seconds=args.timeout_seconds)
    except (ValueError, OSError) as error:
        parser.exit(2, f'Repeatability check failed: {error}\n')
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'runs': len(report['runs']),
                      'report': str(args.output / 'repeatability.json'), 'error': report.get('error')}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
