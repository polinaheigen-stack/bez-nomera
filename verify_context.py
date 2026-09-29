"""Verify the exact non-circular source manifest and the single E27 checkpoint."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parent
WEIGHT_PATH = 'models/e27/model.pth'
WEIGHT_SHA256 = '300b1c0ba907fa90ceacc38693713789cfb2a8b83248fd585fadb0d059381fdd'
WEIGHT_BYTES = 118036485


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(relative):
    parts = Path(relative).parts
    return (parts[0] in {'input', 'results'}
            or any(p in {'__pycache__', '.pytest_cache', 'node_modules', 'dist', 'dist-demo', '.git'}
                   or p.startswith('.venv') for p in parts)
            or Path(relative).suffix == '.pyc' or Path(relative).name == '.env')


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or str(PurePosixPath(relative)) != relative
            or any(p in ('', '.', '..') for p in relative.split('/'))):
        raise ValueError('Unsafe manifest path: ' + str(relative))
    path = root
    for part in relative.split('/'):
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink in build input: ' + relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Escaping build input: ' + relative)
    return path


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest_path = root / 'SOURCE_SHA256.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest.get('files')
    if manifest.get('release') != 'E27' or not isinstance(entries, list) or not entries:
        raise ValueError('Expected the final E27 source manifest.')
    listed = []
    for item in entries:
        name = item['path']
        safe_path(root, name)
        if (name == 'SOURCE_SHA256.json' or ignored(name) or type(item.get('bytes')) is not int
                or item['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', str(item.get('sha256', '')))):
            raise ValueError('Invalid or circular manifest entry: ' + name)
        listed.append(name)
    if len(listed) != len(set(listed)):
        raise ValueError('Duplicate source manifest paths.')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.relative_to(root).as_posix() != 'SOURCE_SHA256.json'
              and not ignored(p.relative_to(root))}
    if actual != set(listed):
        raise ValueError('Source inventory differs: ' + json.dumps({'extra': sorted(actual-set(listed)), 'missing': sorted(set(listed)-actual)}))
    weight_entries = [e for e in entries if Path(e['path']).suffix.lower() in ('.pth', '.pt', '.safetensors')]
    if weight_entries != [{'path': WEIGHT_PATH, 'bytes': WEIGHT_BYTES, 'sha256': WEIGHT_SHA256}]:
        raise ValueError('Expected exactly the single frozen E27 runtime checkpoint.')
    for item in entries:
        path = safe_path(root, item['path'])
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Changed build input: ' + item['path'])
    return {'status': 'verified', 'release': 'E27', 'files': len(entries),
            'source_manifest_sha256': sha256(manifest_path), 'weights_bytes': WEIGHT_BYTES}


if __name__ == '__main__':
    try:
        print(json.dumps(verify()))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error))
