"""Import the single matching E27 checkpoint without downloading or replacing files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_context import safe_path, WEIGHT_BYTES, WEIGHT_SHA256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def import_model(source, *, root=ROOT, check_only=False):
    root, source = Path(root).resolve(), Path(source).resolve()
    destination = root / 'models/e27'
    matches = [p for p in (source, source / 'models/e27', source / 'e27') if (p / 'bundle.json').is_file()]
    if len(matches) != 1:
        raise ValueError('Point --source to the matching unpacked Модель_E27 folder or its models/e27 directory.')
    source = matches[0]
    bundle_bytes = (destination / 'bundle.json').read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle.get('adapter') != 'e27_compact' or (source / 'bundle.json').read_bytes() != bundle_bytes:
        raise ValueError('Model metadata differs from this repository. Use the matching E27 delivery.')
    artifacts = bundle['artifacts']
    if artifacts.get('weights') != {'file': 'model.pth', 'sha256': WEIGHT_SHA256}:
        raise ValueError('Expected the single frozen E27 checkpoint.')
    expected = {'bundle.json': sha256(destination / 'bundle.json')}
    for item in artifacts.values():
        name, digest = item['file'], item['sha256']
        if name in expected:
            raise ValueError('Duplicate model artifact: ' + name)
        expected[name] = digest
    if [name for name in expected if Path(name).suffix in ('.pth', '.pt', '.safetensors')] != ['model.pth']:
        raise ValueError('Expected exactly one checkpoint.')
    for name, digest in expected.items():
        incoming, existing = safe_path(source, name), safe_path(destination, name)
        if not incoming.is_file() or sha256(incoming) != digest:
            raise ValueError('Missing or changed incoming model file: ' + name)
        if existing.exists() and (not existing.is_file() or sha256(existing) != digest):
            raise ValueError('Different existing repository file; refusing overwrite: ' + name)
        if name != 'model.pth' and not existing.is_file():
            raise ValueError('Repository model metadata is missing: ' + name)
    if (source / 'model.pth').stat().st_size != WEIGHT_BYTES:
        raise ValueError('Checkpoint size differs from the frozen E27 model.')
    target, copied = destination / 'model.pth', []
    if not check_only and not target.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix='.import-', suffix='.tmp', dir=destination, delete=False) as stream:
                temporary = Path(stream.name)
                with (source / 'model.pth').open('rb') as incoming:
                    shutil.copyfileobj(incoming, stream, 8 * 1024 * 1024)
            if sha256(temporary) != WEIGHT_SHA256 or temporary.stat().st_size != WEIGHT_BYTES:
                raise ValueError('Checkpoint changed while copying.')
            # link() is atomic and fails rather than replacing an existing destination.
            import os
            os.link(temporary, target)
            copied.append('models/e27/model.pth')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {'status': 'verified' if check_only else 'imported', 'model_version': bundle['model']['version'],
            'verified_model_files': len(expected), 'weights_bytes': WEIGHT_BYTES, 'copied': copied,
            'next': 'python verify_context.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(import_model(args.source, check_only=args.check_only), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
