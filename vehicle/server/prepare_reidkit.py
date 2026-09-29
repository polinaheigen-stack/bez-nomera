"""Package a completed or explicitly frozen trained model and separate calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .model_bundle import read_object, sha256, inspect_bundle
from .reidkit_contract import validate_config, validate_frozen_selection, source_hash, DEPENDENCIES, PROTOCOL, STOP_REASONS


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def snapshot_frozen_json(path):
    contents = Path(path).read_bytes()
    if len(contents) > 1024 * 1024:
        raise ValueError('Frozen selection JSON is too large')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate frozen selection field: ' + key)
            result[key] = value
        return result
    value = json.loads(contents.decode('utf-8-sig'), object_pairs_hook=unique)
    return contents, hashlib.sha256(contents).hexdigest(), value


def prepare(run, calibration_dir, output, frozen_selection_json=None):
    run, calibration_dir, output = map(Path, (run, calibration_dir, output))
    if output.exists():
        raise ValueError('Output must be a new directory; existing bundles are never overwritten')
    summary = read_object(run / 'run.json')
    if summary.get('control_and_official_test_used') is not False:
        raise ValueError('Control/official-test data may not be used for model selection')
    training_complete = summary.get('status') == 'complete'
    if training_complete == (frozen_selection_json is not None):
        raise ValueError('Incomplete runs require --frozen-selection-json; completed runs must omit it')
    source = run / 'bundle'
    config = read_object(source / 'config.json')
    metadata = read_object(source / 'metadata.json')
    raw = read_object(calibration_dir / 'calibration.json')
    proof = read_object(calibration_dir / 'provenance.json')
    validate_config(config)
    weights_hash = sha256(source / 'model.pth')
    frozen_selection = frozen_bytes = frozen_hash = None
    if frozen_selection_json is not None:
        frozen_bytes, frozen_hash, frozen_selection = snapshot_frozen_json(frozen_selection_json)
        validate_frozen_selection(frozen_selection, weights_hash)
    config_hash = sha256(source / 'config.json')
    if metadata.get('kind') != 'trained_vehicle_reid' or metadata.get('checkpoint_sha256') != weights_hash:
        raise ValueError('Training bundle metadata does not identify its checkpoint')
    training = metadata['provenance']
    if training != summary.get('provenance') or training.get('config_sha256') != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError('Run/config/provenance mismatch')
    code_hash = source_hash()
    threshold = raw.get('selected', {}).get('threshold')
    if type(threshold) not in (int, float):
        raise ValueError('Separately calibrated numerical threshold required')
    calibration = {'calibration_protocol': PROTOCOL, 'model_sha256': weights_hash,
                   'inference_config_sha256': config_hash, 'code_hash': code_hash,
                   'threshold_cosine': 2 * threshold - 1, 'original': raw, 'calibration_provenance': proof}
    if frozen_selection is not None:
        calibration['selection_lock'] = read_object(calibration_dir.parent / 'selection-lock.json')
        calibration['source_calibration_sha256'] = sha256(calibration_dir / 'calibration.json')
    manifest = {'schema_version': 1, 'status': 'ready', 'adapter': 'reidkit_single',
                'model': {'name': config['name'], 'version': f"trained-epoch{metadata['epoch']}-{weights_hash[:12]}", 'dimension': 512},
                'artifacts': {'weights': {'file': 'model.pth', 'sha256': weights_hash},
                              'config': {'file': 'config.json', 'sha256': config_hash}},
                'provenance': {'inference_code_sha256': code_hash, 'handoff_metadata_sha256': sha256(source / 'metadata.json'),
                               'training': training, 'training_complete': training_complete},
                'inference': {'adapter_revision': 1, 'preprocessing': 'native_checkpoint', 'score': 'cosine_similarity', 'dependency_versions': DEPENDENCIES}}
    if frozen_selection is not None:
        manifest['provenance'].update(frozen_selection=frozen_selection,
                                     frozen_selection_file_sha256=frozen_hash,
                                     training_stop_reason=frozen_selection['reason'])
        manifest['model']['version'] += '-' + STOP_REASONS[frozen_selection['reason']]
    # Validate before copying a large checkpoint. The generated bundle still uses
    # the existing server validator after writing its concrete artifact hashes.
    from .reidkit_contract import validate_bundle_fields
    validate_bundle_fields(manifest, config, calibration)
    output.mkdir(parents=True)
    for name in ('model.pth', 'config.json', 'metadata.json'):
        shutil.copyfile(source / name, output / name)
    if frozen_selection is not None:
        (output / 'frozen-selection.json').write_bytes(frozen_bytes)
    write_json(output / 'calibration.json', calibration)
    manifest['artifacts']['calibration'] = {'file': 'calibration.json', 'sha256': sha256(output / 'calibration.json')}
    write_json(output / 'bundle.json', manifest)
    inspected = inspect_bundle(output)
    return {'status': 'metadata_checked', 'bundle': str(output.resolve()), 'model': inspected.manifest['model'],
            'weights_sha256': weights_hash, 'threshold_cosine': inspected.threshold_cosine,
            'checkpoint_loaded': False, 'inference_executed': False, 'training_complete': training_complete,
            'training_stop_reason': frozen_selection['reason'] if frozen_selection is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-selection-json', type=Path,
                        help='For an actually stopped budget-limited or user-stopped run; requires evaluation selection-lock.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.training_run, args.calibration_dir, args.output, args.frozen_selection_json), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
