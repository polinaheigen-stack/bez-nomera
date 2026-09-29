"""Frozen E25 ensemble contract. No downloads, fallback or configurable tuning."""
from pathlib import Path, PurePosixPath

from .model_bundle import (MAX_WEIGHTS_BYTES, WEIGHT_SUFFIXES, ModelBundle, adapter_code_hash,
                           digest_json, inspect_bundle, read_object, require_hash, sha256)
from .reidkit_contract import DEPENDENCIES, source_hash as member_source_hash

MEMBERS = (
    ('dino-large', 'cea1da9a1791ae27ac5bf854076f204a538eef0babb100507ab8a9af1c2a609e'),
    ('swin-base', 'b5a9e5158ca49b8507fc9a5517facb80c26428c2fb7f9f735b37f979dbaf16f1'),
)
RECIPE_SHA256 = '88787ae7bbff5098822fa7afbd20a32992f3e025805ae085212c08a11b571877'
PROTOCOL_SHA256 = 'f53ddfa77ae9b55e3e44aeb3367c19fcd45b94a824904b47bca7c0c16f38e808'
THRESHOLD = 0.6966247613864339
MODEL_VERSION = 'e25-dino-large-swin-base-5050-kreciprocal-r3-cuda_graphs_preprocess-t1'
ADAPTER_REVISION = 2
RUNTIME = {'cuda_mode': 'cuda_graphs_preprocess', 'cpu_threads': 1, 'interop_threads': 1, 'cpu_mode': 'eager'}
CALIBRATION_SELECTION = 'r2_calibration_threshold_retained_for_r3_runtime_validation'
RETRIEVAL = {'type': 'k_reciprocal', 'top_k': 50, 'k1': 5, 'lambda_value': 0.6,
             'implementation_sha256': '6eef2ebd048e257737a67b0218d7e1e9658269c985125c55b23d9da5eff73620',
             'score_domain': '1-final_distance', 'query_expansion': False}


def source_hash():
    root = Path(__file__).parent
    return digest_json({'reidkit_runtime': member_source_hash(),
                        **{name: sha256(root / name) for name in ('e25_adapter.py', 'e25_contract.py', 'e25_runtime.py', 'e25_preprocess.py')}})


def frozen_config():
    return {'schema_version': 1, 'name': 'E25', 'embedding_dim': 1024,
            'fusion': 'normalized_sqrt_weight_concatenation', 'flip_tta': False, 'model_batch_size': 1,
            'members': [{'name': name, 'weight': 0.5, 'dimension': 512} for name, _ in MEMBERS],
            'inference': {'runtime': dict(RUNTIME)},
            'retrieval': dict(RETRIEVAL)}


def composite_weights_hash():
    return digest_json([{'name': name, 'sha256': value} for name, value in MEMBERS])


def validate_config(config):
    # Canonical JSON equality also rejects booleans substituted for 0/1 integers.
    if digest_json(config) != digest_json(frozen_config()):
        raise ValueError('E25 weights, preprocessing, fusion or retrieval configuration changed')


def safe_file(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or any(p in ('..', '.') for p in relative.split('/'))):
        raise ValueError('E25 requires a safe relative POSIX path')
    raw = root / relative
    for part in (raw, *raw.parents):
        if part == root:
            break
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('Links are forbidden inside an E25 bundle')
    path = raw.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('E25 artifact missing or outside bundle')
    return path


def artifact(root, entry, name, limit=1024 * 1024):
    if not isinstance(entry, dict) or set(entry) != {'file', 'sha256'}:
        raise ValueError(f'E25 {name}: expected file and sha256')
    expected = require_hash(entry['sha256'], name)
    path = safe_file(root, entry['file'])
    if not 0 < path.stat().st_size <= limit or sha256(path) != expected:
        raise ValueError(f'E25 {name}: size or SHA256 mismatch')
    return path


def inspect_e25_bundle(root, manifest):
    expected_inference = {'adapter_revision': ADAPTER_REVISION, 'preprocessing': 'native_checkpoint',
                          'score': '1-final_distance', 'dependency_versions': DEPENDENCIES}
    if digest_json(manifest.get('inference')) != digest_json(expected_inference):
        raise ValueError('E25 inference contract differs from the frozen runtime')
    if digest_json(manifest.get('model')) != digest_json({'name': 'E25', 'version': MODEL_VERSION, 'dimension': 1024}):
        raise ValueError('Unsupported E25 identity or dimension')
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict) or set(artifacts) != {'config', 'calibration', 'recipe', 'protocol'}:
        raise ValueError('E25 needs config, calibration, frozen recipe and protocol')
    paths = {name: artifact(root, entry, name) for name, entry in artifacts.items()}
    if len(set(paths.values())) != 4:
        raise ValueError('E25 artifacts must be distinct files')
    if (artifacts['recipe']['sha256'] != RECIPE_SHA256
            or artifacts['protocol']['sha256'] != PROTOCOL_SHA256):
        raise ValueError('E25 recipe/protocol differs from the frozen experiment')
    config, calibration = read_object(paths['config']), read_object(paths['calibration'])
    validate_config(config)
    recipe = read_object(paths['recipe'])
    if (recipe.get('threshold') != THRESHOLD or recipe.get('control_used_for_selection') is not False
            or recipe.get('protocol_sha256') != PROTOCOL_SHA256):
        raise ValueError('Invalid frozen E25 threshold provenance')
    provenance = manifest.get('provenance', {})
    if digest_json(provenance) != digest_json({'inference_code_sha256': source_hash(), 'frozen_recipe_sha256': RECIPE_SHA256,
                                              'frozen_protocol_sha256': PROTOCOL_SHA256, 'control_used_for_selection': False}):
        raise ValueError('E25 runtime or selection provenance mismatch')
    expected_calibration = {'schema_version': 1, 'threshold_score': THRESHOLD,
                            'score_domain': RETRIEVAL['score_domain'], 'selection': CALIBRATION_SELECTION,
                            'model_sha256': composite_weights_hash(),
                            'inference_config_sha256': artifacts['config']['sha256'],
                            'frozen_recipe_sha256': RECIPE_SHA256, 'frozen_protocol_sha256': PROTOCOL_SHA256,
                            'control_used_for_selection': False}
    if digest_json(calibration) != digest_json(expected_calibration):
        raise ValueError('E25 calibration is not bound to the frozen model/retrieval')
    entries = manifest.get('members')
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError('E25 requires exactly two reviewed member bundles')
    bundles = []
    declared_files = {root / 'bundle.json', *paths.values()}
    for entry, (name, weight_hash) in zip(entries, MEMBERS):
        if not isinstance(entry, dict) or set(entry) != {'name', 'directory', 'files'} or entry['name'] != name:
            raise ValueError('E25 member order/identity mismatch')
        if entry['directory'] != f'members/{name}' or not isinstance(entry['files'], dict):
            raise ValueError('Invalid E25 member path/files')
        member_root = (root / entry['directory']).resolve()
        if set(entry['files']) != {'bundle.json', 'model.pth', 'config.json', 'calibration.json', 'metadata.json', 'frozen-selection.json'}:
            raise ValueError('Unexpected E25 member files')
        for file, expected in entry['files'].items():
            path = artifact(root, {'file': entry['directory'] + '/' + file, 'sha256': expected},
                            name + '/' + file, MAX_WEIGHTS_BYTES if file == 'model.pth' else 1024 * 1024)
            declared_files.add(path)
        child_manifest = read_object(member_root / 'bundle.json')
        if child_manifest.get('adapter') != 'reidkit_single':
            raise ValueError('Nested ensembles and unknown member adapters are forbidden')
        child = inspect_bundle(member_root)
        if (child.weights_sha256 != weight_hash or child.config.get('name') != name
                or child.config.get('flip_tta', False) is not False
                or child.manifest['model']['dimension'] != 512):
            raise ValueError('Wrong E25 checkpoint or member inference recipe')
        if (entry['files']['metadata.json'] != child_manifest['provenance']['handoff_metadata_sha256']
                or entry['files']['frozen-selection.json'] != child_manifest['provenance']['frozen_selection_file_sha256']):
            raise ValueError('Member metadata or freeze provenance mismatch')
        bundles.append(child)
    all_entries = list(root.rglob('*'))
    if any(path.is_symlink() or getattr(path, 'is_junction', lambda: False)() for path in all_entries):
        raise ValueError('Links are forbidden inside an E25 bundle')
    actual_files = {path for path in all_entries if path.is_file()}
    if actual_files != declared_files:
        raise ValueError('Extra or missing files in E25 model bundle')
    weights = [path for path in actual_files if path.name.lower().endswith(WEIGHT_SUFFIXES)]
    total = sum(path.stat().st_size for path in weights)
    if (len(weights) != 2 or not 0 < total <= MAX_WEIGHTS_BYTES
            or type(manifest.get('weights_bytes')) is not int or manifest['weights_bytes'] != total):
        raise ValueError('E25 must contain exactly two weights files, at most 2 GB total')
    if manifest.get('weights_sha256') != composite_weights_hash():
        raise ValueError('E25 composite weights fingerprint mismatch')
    fingerprint = digest_json({'adapter': 'e25_ensemble', 'config_sha256': artifacts['config']['sha256'],
                               'members': [child.inference_fingerprint for child in bundles],
                               'inference_code_sha256': source_hash(), 'adapter_code_sha256': adapter_code_hash(),
                               'dimension': 1024})
    return ModelBundle(root, manifest, config, calibration, None, fingerprint, None,
                       tuple(bundles), dict(RETRIEVAL), THRESHOLD)
