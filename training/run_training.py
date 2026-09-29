"""Explicit, single-profile reproduction of the selected stand-5 training trial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PRETRAINED_SHA = '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615'


def verify_copied_sources():
    manifest = json.loads((ROOT / 'COPIED_TRAINING_SOURCE_SHA256.json').read_text(encoding='utf-8'))
    records = manifest.get('files', manifest.get('sha256', {}))
    if not records:
        raise ValueError('Missing training source hashes')
    for name, digest in records.items():
        file = (ROOT / name).resolve()
        if not file.is_relative_to(ROOT) or not file.is_file():
            raise ValueError('Missing or unsafe source: ' + name)
        expected = digest.get('sha256') if isinstance(digest, dict) else digest
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Training source changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, default=ROOT / 'assets/dinov3-splus/model.safetensors')
    parser.add_argument('--teacher-cache', type=Path, default=ROOT / 'assets/teacher-train-cache')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(args.dataset.resolve()):
        parser.error('Use a new output directory outside training source and dataset')
    verify_copied_sources()
    profile = json.loads((ROOT / 'profiles/stand-5.json').read_text(encoding='utf-8'))
    if profile.get('stand_id') != 5 or profile.get('epochs') != 20:
        parser.error('The frozen stand-5 training profile is required')
    sys.path.insert(0, str(ROOT / 'runtime'))
    from vehicle.server.compact_trial import main as run_trial
    command = ['--profile', str(ROOT / 'profiles/stand-5.json'), '--dataset', str(args.dataset),
               '--inputs', str(ROOT / 'inputs'), '--pretrained', str(args.pretrained),
               '--pretrained-sha', PRETRAINED_SHA, '--output', str(output),
               '--teacher-cache', str(args.teacher_cache),
               '--teacher-cache-receipt', str(args.teacher_cache / 'receipt.json')]
    if args.preflight_only:
        command.append('--preflight-only')
    run_trial(command)


if __name__ == '__main__':
    main()
