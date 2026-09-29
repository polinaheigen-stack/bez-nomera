"""Import the supplied composite E22 weights and JSON into a checked local bundle."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

from .e22_contract import checkpoint_config, source_hash, SOURCE_COMMIT, PROTOCOL
from .model_bundle import read_object, sha256, inspect_bundle, checked_cosine_threshold, MAX_WEIGHTS_BYTES


def prepare(checkpoint, metadata, output):
    import torch
    from .model_adapters import load_adapter
    checkpoint, metadata, output = map(lambda p: Path(p).resolve(), (checkpoint, metadata, output))
    if output.exists():
        raise ValueError('Каталог пакета уже существует. Укажите новый каталог; готовые пакеты не перезаписываются.')
    declared = read_object(metadata)
    if not 0 < checkpoint.stat().st_size <= MAX_WEIGHTS_BYTES:
        raise ValueError('Размер весов E22 должен быть от 1 байта до 2 ГБ.')
    weight_hash = sha256(checkpoint)
    if declared.get('checkpoint_sha256') != weight_hash or declared.get('checkpoint_bytes') != checkpoint.stat().st_size:
        raise ValueError('Весовой файл E22 не совпадает с переданным JSON.')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint_config(state)
    for field in ('schema', 'experiment', 'variant', 'embedding_dim', 'member_weights', 'calibration'):
        if declared.get(field) != state.get(field):
            raise ValueError(f'JSON не совпадает с checkpoint: {field}')
    expected_members = [{'name': m['name'], 'source_sha256': m['source_sha256'],
                         'architecture': m['config']['architecture']} for m in state['members']]
    if [{k: m.get(k) for k in ('name', 'source_sha256', 'architecture')}
        for m in declared.get('members', [])] != expected_members:
        raise ValueError('JSON описывает другие исходные модели E22.')
    original = state['calibration']
    if not isinstance(original, dict):
        raise ValueError('В checkpoint E22 нет калибровки.')
    threshold = checked_cosine_threshold(original.get('selected', {}).get('threshold'))
    del state
    code_hash = source_hash()
    output.parent.mkdir(parents=True, exist_ok=True)
    # This new staging directory is confined to the requested output parent.
    with tempfile.TemporaryDirectory(prefix='.e22-import-', dir=output.parent) as temporary:
        root = Path(temporary)
        def write(name, value):
            (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        shutil.copyfile(checkpoint, root / 'e22.pth')
        if sha256(root / 'e22.pth') != weight_hash:
            raise ValueError('Веса изменились при копировании.')
        write('config.json', config)
        write('calibration.json', {'calibration_protocol': PROTOCOL, 'model_sha256': weight_hash,
              'inference_config_sha256': sha256(root / 'config.json'), 'code_hash': code_hash,
              'threshold_cosine': threshold, 'original': original,
              'note': 'Imported threshold unchanged; original diagnostic statistics are not official evaluation results.'})
        versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'Pillow')}
        versions['torch'] = versions['torch'].split('+')[0]
        write('bundle.json', {
            'schema_version': 1, 'status': 'ready', 'adapter': 'e22_ensemble',
            'model': {'name': 'E22 · ConvNeXt-Tiny + DINOv2-B/14',
                      'version': 'research2030-seed2026-' + weight_hash[:12], 'dimension': 512},
            'artifacts': {key: {'file': name, 'sha256': sha256(root/name)} for key, name in
                          (('weights', 'e22.pth'), ('config', 'config.json'), ('calibration', 'calibration.json'))},
            'provenance': {'source_commit': SOURCE_COMMIT, 'source_role': 'supplied_reproduction_source',
                           'inference_code_sha256': code_hash, 'handoff_metadata_sha256': sha256(metadata),
                           'note': 'This records supplied inference source; it does not claim an exact historical training commit or a verified training split manifest.'},
            'inference': {'adapter_revision': 1, 'preprocessing': 'native_checkpoint',
                          'score': 'cosine_similarity', 'dependency_versions': versions}})
        bundle = inspect_bundle(root)
        load_adapter(bundle, 'cpu')  # Strictly validate both encoders before publishing.
        # Copy into an absent destination only; no existing user package is changed.
        shutil.copytree(root, output)
    return {'status': 'checkpoint_loaded', 'bundle': str(output), 'weights_sha256': weight_hash,
            'threshold_cosine': threshold, 'threshold_service': (threshold+1)/2,
            'dimension': 512, 'quality_measured': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkpoint, args.metadata, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
