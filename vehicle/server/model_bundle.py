"""Validate one operator-supplied model package. No downloads or model execution.

Run ``python -m vehicle.server.model_bundle check --bundle /models`` to inspect
metadata; add --load to verify the checkpoint and installed runtime as well.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys


# Add only the selected model's reviewed adapter at integration time.
ADAPTERS = frozenset({'e22_ensemble', 'reidkit_single', 'e25_ensemble', 'e27_compact'})
MAX_WEIGHTS_BYTES = 2_000_000_000
WEIGHT_SUFFIXES = ('.pt', '.pth', '.bin', '.onnx', '.engine', '.plan', '.safetensors',
                   '.ckpt', '.trt', '.pb', '.tflite', '.npz', '.onnx.data')
HASH_PATTERN = re.compile(r'^[a-f0-9]{64}$')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def digest_json(value, *, compact=True):
    options = {'separators': (',', ':')} if compact else {}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, **options).encode('utf-8')).hexdigest()


def read_object(path, limit=1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError(f'JSON превышает допустимый размер: {path.name}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Повторяющийся ключ JSON: {key}')
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Неконечное число JSON: {value}')))
    if not isinstance(value, dict):
        raise ValueError(f'Ожидается JSON-объект: {path.name}')
    return value


def require_hash(value, name):
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value) or value == '0' * 64:
        raise ValueError(f'{name}: требуется настоящий SHA256 в нижнем регистре.')
    return value


def checked_cosine_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Порог cosine должен быть конечным числом от -1 до 1.')
    return float(value)


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    manifest: dict
    config: dict
    calibration: dict
    weights_path: Path | None
    inference_fingerprint: str
    threshold_cosine: float | None
    member_bundles: tuple = ()
    retrieval: dict | None = None
    threshold_score: float | None = None

    @property
    def weights_sha256(self):
        if self.adapter == 'e25_ensemble':
            return self.manifest['weights_sha256']
        return self.manifest['artifacts']['weights']['sha256']

    @property
    def retrieval_fingerprint(self):
        if self.retrieval is None:
            return None
        return digest_json({'retrieval': self.retrieval, 'threshold': self.threshold_score,
                            'calibration_sha256': self.calibration_sha256})

    @property
    def adapter(self):
        return self.manifest['adapter']

    @property
    def calibration_sha256(self):
        return self.manifest['artifacts']['calibration']['sha256']


def _artifact(root, artifacts, name, limit):
    item = artifacts.get(name)
    if not isinstance(item, dict) or set(item) != {'file', 'sha256'}:
        raise ValueError(f'artifacts.{name}: нужны file и sha256.')
    expected = require_hash(item['sha256'], f'artifacts.{name}.sha256')
    relative = item['file']
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError(f'artifacts.{name}.file: требуется относительный путь с /.')
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'artifacts.{name}: файл отсутствует или находится вне пакета.')
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f'artifacts.{name}: недопустимый размер файла.')
    if sha256(path) != expected:
        raise ValueError(f'artifacts.{name}: SHA256 не совпадает.')
    return path


def adapter_code_hash():
    """Bind generated vectors to the server-side bridge, not only backbone code."""
    root = Path(__file__).parent
    files = [root / name for name in ('model_bundle.py', 'model_adapters.py', 'provider.py',
                                      'e22_adapter.py', 'e22_contract.py')]
    files += sorted((root / 'e22_source').glob('*.py'))
    files += [root / 'e22_source' / 'SOURCE.json']
    files += [root / name for name in ('reidkit_adapter.py', 'reidkit_contract.py')]
    files += sorted((root / 'reidkit_source').glob('*.py'))
    files += [root / 'reidkit_source' / 'SOURCE.json']
    files += [root / name for name in ('e25_adapter.py', 'e25_contract.py', 'e25_runtime.py', 'e25_preprocess.py')]
    return digest_json({path.relative_to(root).as_posix(): sha256(path) for path in files})


def inspect_bundle(directory):
    root = Path(directory).resolve()
    manifest = read_object(root / 'bundle.json')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Поддерживается только bundle schema_version=1.')
    if manifest.get('status') != 'ready':
        raise ValueError('Пакет не готов: status должен быть ready. Шаблон не является моделью.')
    if manifest.get('adapter') not in ADAPTERS:
        raise ValueError('Неизвестный адаптер модели.')
    if manifest['adapter'] == 'e27_compact':
        from .e27_contract import inspect_e27_bundle
        return inspect_e27_bundle(root, manifest)
    if manifest['adapter'] == 'e25_ensemble':
        from .e25_contract import inspect_e25_bundle
        return inspect_e25_bundle(root, manifest)
    model = manifest.get('model', {})
    if not isinstance(model, dict) or any(not isinstance(model.get(k), str) or not model[k].strip() for k in ('name', 'version')):
        raise ValueError('Нужны непустые model.name и model.version.')
    if type(model.get('dimension')) is not int or not 1 <= model['dimension'] <= 16384:
        raise ValueError('Некорректная model.dimension.')
    provenance = manifest.get('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('Не задано происхождение модели.')
    for name in ('inference_code_sha256', 'handoff_metadata_sha256'):
        require_hash(provenance.get(name), f'provenance.{name}')
    inference = manifest.get('inference', {})
    if (not isinstance(inference, dict) or inference.get('adapter_revision') != 1
            or type(inference.get('adapter_revision')) is not int
            or inference.get('preprocessing') != 'native_checkpoint'
            or inference.get('score') != 'cosine_similarity'):
        raise ValueError('Неподдерживаемый inference-контракт.')
    versions = inference.get('dependency_versions')
    if (not isinstance(versions, dict) or not {'torch', 'numpy', 'Pillow'} <= versions.keys()
            or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in versions.items())):
        raise ValueError('Требуются точные dependency_versions, включая torch, numpy и Pillow.')
    artifacts = manifest.get('artifacts', {})
    if not isinstance(artifacts, dict) or set(artifacts) != {'weights', 'config', 'calibration'}:
        raise ValueError('Нужны ровно три артефакта: weights, config, calibration.')
    weights = _artifact(root, artifacts, 'weights', MAX_WEIGHTS_BYTES)
    for candidate in root.rglob('*'):
        if candidate.is_file() and candidate.name.lower().endswith(WEIGHT_SUFFIXES):
            if candidate.is_symlink() or candidate != weights:
                raise ValueError(f'Лишний файл весов в пакете одной модели: {candidate.relative_to(root)}')
    config_path = _artifact(root, artifacts, 'config', 1024 * 1024)
    calibration_path = _artifact(root, artifacts, 'calibration', 1024 * 1024)
    if len({weights, config_path, calibration_path}) != 3:
        raise ValueError('Файлы weights, config и calibration должны быть разными.')
    config, calibration = read_object(config_path), read_object(calibration_path)
    if config.get('debug_only'):
        raise ValueError('Отладочная конфигурация не допускается в PROD.')
    threshold = checked_cosine_threshold(calibration.get('threshold_cosine'))
    if manifest['adapter'] == 'e22_ensemble':
        from .e22_contract import validate_bundle_fields
    else:
        from .reidkit_contract import validate_bundle_fields
    validate_bundle_fields(manifest, config, calibration)
    fingerprint = digest_json({'adapter': manifest['adapter'], 'inference': inference,
                              'weights_sha256': artifacts['weights']['sha256'],
                              'config_sha256': artifacts['config']['sha256'],
                              'inference_code_sha256': provenance['inference_code_sha256'],
                              'adapter_code_sha256': adapter_code_hash(), 'dimension': model['dimension']})
    return ModelBundle(root, manifest, config, calibration, weights, fingerprint, threshold)


def verify_dependencies(bundle):
    for name, expected in bundle.manifest['inference']['dependency_versions'].items():
        actual = importlib.metadata.version(name)
        compatible_build = name in ('torch', 'torchvision') and '+' not in expected and actual.split('+')[0] == expected
        if actual != expected and not compatible_build:
            raise ValueError(f'Версия {name}: установлена {actual}, пакет требует {expected}.')


def runtime_fingerprint(bundle, device):
    actual_versions = {name: importlib.metadata.version(name)
                       for name in bundle.manifest['inference']['dependency_versions']}
    return digest_json({'bundle_inference_fingerprint': bundle.inference_fingerprint,
                        'device': str(device), 'actual_dependency_versions': actual_versions})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check')
    check.add_argument('--bundle', type=Path, required=True)
    check.add_argument('--load', action='store_true', help='Verify installed runtime and strict checkpoint loading; no quality claim')
    check.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    try:
        bundle = inspect_bundle(args.bundle)
        result = {'status': 'metadata_only', 'model': bundle.manifest['model'],
                  'bundle_inference_fingerprint': bundle.inference_fingerprint, 'inference_fingerprint': None,
                  'calibration_sha256': bundle.calibration_sha256, 'inference_executed': False}
        if bundle.retrieval is not None:
            result.update(retrieval=bundle.retrieval, retrieval_fingerprint=bundle.retrieval_fingerprint,
                          threshold=bundle.threshold_score, weights_sha256=bundle.weights_sha256,
                          weights_bytes=bundle.manifest['weights_bytes'])
        if args.load:
            from .model_adapters import load_adapter
            adapter = load_adapter(bundle, args.device)
            try:
                setup = getattr(adapter, 'runtime_metadata', None)
                result.update(status='checkpoint_loaded', device=str(adapter.device),
                              inference_fingerprint=runtime_fingerprint(bundle, adapter.device),
                              runtime_setup=setup, quality_measured=False,
                              inference_executed=bool(setup and setup.get('forward_executed', False)))
            finally:
                close = getattr(adapter, 'close', None)
                if callable(close):
                    close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'rejected', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
