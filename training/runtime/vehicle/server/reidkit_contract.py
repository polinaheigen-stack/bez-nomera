"""Reviewed three-model contract. Calibration is separate from fitting and control."""
from .model_bundle import digest_json, read_object, require_hash, sha256
from pathlib import Path
import math
from datetime import datetime, timezone

RECIPES = {
    'dino-large': ('vit_large_patch14_dinov2.lvd142m', [336, 336], 'cls_plus_mean_patches'),
    'convnext-base': ('convnext_base.fb_in22k_ft_in1k', [384, 384], 'native_average'),
    'swin-base': ('swin_base_patch4_window12_384.ms_in22k_ft_in1k', [384, 384], 'native_average'),
}
DEPENDENCIES = {'torch': '2.8.0', 'torchvision': '0.23.0', 'timm': '1.0.20',
                'numpy': '2.2.6', 'Pillow': '11.3.0', 'safetensors': '0.6.2'}
PROTOCOL = 'reidkit_calibration_split_v1'
STOP_REASONS = {'budget_limit': 'budget-stopped', 'user_requested_early_stop': 'user-stopped'}


def validate_frozen_selection(selection, checkpoint_hash):
    """Same freeze contract as post_training_evaluate; no claim about live processes."""
    if (not isinstance(selection, dict) or selection.get('schema_version') != 1
            or selection.get('reason') not in STOP_REASONS or selection.get('training_complete') is not False
            or selection.get('checkpoint_sha256') != checkpoint_hash
            or selection.get('selection_basis') != 'best_internal_dev_mAP@10'
            or selection.get('control_used_for_selection') is not False):
        raise ValueError('Invalid frozen checkpoint selection or unsupported stop reason')
    frozen_at = datetime.fromisoformat(selection.get('frozen_at_utc', ''))
    if (frozen_at.tzinfo is None or frozen_at.utcoffset().total_seconds() != 0
            or frozen_at > datetime.now(timezone.utc)):
        raise ValueError('Freeze timestamp must be UTC and not in the future')


def source_hash():
    root = Path(__file__).parent
    source = read_object(root / 'reidkit_source' / 'SOURCE.json')
    if source.get('schema') != 'reidkit-reviewed-runtime-v1':
        raise ValueError('Unknown reidkit runtime snapshot')
    if set(source['files']) != {'__init__.py', 'model.py', 'data.py'}:
        raise ValueError('Unexpected reidkit runtime files')
    files = [root / 'reidkit_adapter.py', root / 'reidkit_contract.py', root / 'reidkit_source' / 'SOURCE.json']
    for name, entry in source['files'].items():
        path = root / 'reidkit_source' / name
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Reviewed runtime changed: {name}')
        files.append(path)
    return digest_json({p.relative_to(root).as_posix(): sha256(p) for p in files})


def validate_config(config):
    recipe = RECIPES.get(config.get('name'))
    if not recipe or (config.get('timm_model'), config.get('image_size'), config.get('pooling')) != recipe:
        raise ValueError('Unsupported architecture, image size or pooling')
    if config.get('embedding_dim') != 512 or config.get('debug_only'):
        raise ValueError('A trained 512-dimensional reidkit model is required')
    if (config.get('normalization_mean') != [0.485, 0.456, 0.406]
            or config.get('normalization_std') != [0.229, 0.224, 0.225]
            or config.get('interpolation') != 'bicubic'):
        raise ValueError('Preprocessing differs from the reviewed training runtime')
    if type(config.get('flip_tta', False)) is not bool:
        raise ValueError('flip_tta must be boolean')


def validate_bundle_fields(manifest, config, calibration):
    validate_config(config)
    provenance = manifest['provenance']
    if manifest['model']['dimension'] != 512 or provenance.get('inference_code_sha256') != source_hash():
        raise ValueError('Runtime source or embedding dimension mismatch')
    if manifest['inference']['dependency_versions'] != DEPENDENCIES:
        raise ValueError('Reidkit requires the reviewed exact dependency versions')
    training = provenance.get('training')
    if not isinstance(training, dict):
        raise ValueError('Missing training provenance')
    if type(provenance.get('training_complete')) is not bool:
        raise ValueError('Explicit training_complete status is required')
    if not provenance['training_complete']:
        selection = provenance.get('frozen_selection')
        validate_frozen_selection(selection, manifest['artifacts']['weights']['sha256'])
        if provenance.get('training_stop_reason') != selection['reason']:
            raise ValueError('Training stop reason differs from frozen selection')
        freeze_hash = require_hash(provenance.get('frozen_selection_file_sha256'), 'frozen_selection_file_sha256')
        lock = calibration.get('selection_lock', {})
        if (lock.get('checkpoint_sha256') != manifest['artifacts']['weights']['sha256']
                or lock.get('training_complete') is not False or lock.get('control_used_for_selection') is not False
                or lock.get('frozen_selection_file_sha256') != freeze_hash
                or lock.get('calibration_json_sha256') != calibration.get('source_calibration_sha256')
                or lock.get('threshold') != calibration.get('original', {}).get('selected', {}).get('threshold')):
            raise ValueError('Stopped model requires an unchanged calibration/selection lock')
    require_hash(training.get('split_manifest_sha256'), 'training.split_manifest_sha256')
    source = read_object(Path(__file__).parent / 'reidkit_source' / 'SOURCE.json')
    for name, entry in source['files'].items():
        if training.get('source_sha256', {}).get('reidkit/' + name) != entry['sha256']:
            raise ValueError(f'Training runtime differs from reviewed runtime: {name}')
    raw, proof = calibration.get('original', {}), calibration.get('calibration_provenance', {})
    threshold = raw.get('selected', {}).get('threshold')
    if (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Calibrated score threshold must be in [0,1]; reject-all sentinel is unsupported')
    if (calibration.get('calibration_protocol') != PROTOCOL
            or calibration.get('model_sha256') != manifest['artifacts']['weights']['sha256']
            or calibration.get('inference_config_sha256') != manifest['artifacts']['config']['sha256']
            or calibration.get('code_hash') != provenance['inference_code_sha256']
            or raw.get('checkpoint_sha256') != manifest['artifacts']['weights']['sha256']
            or raw.get('calibration_only') is not True or raw.get('score_domain') != '(cosine + 1) / 2'
            or raw.get('selection') != 'max official F1, then TNR, then higher threshold'
            or calibration.get('threshold_cosine') != 2 * threshold - 1):
        raise ValueError('Calibration does not bind to this checkpoint/config/runtime')
    if (proof.get('stage') != 'calibration' or proof.get('threshold_selected_on') != 'calibration'
            or proof.get('checkpoint_sha256') != raw['checkpoint_sha256']
            or proof.get('threshold') != threshold or proof.get('training') != training):
        raise ValueError('Threshold must come from this model and its isolated calibration split')
