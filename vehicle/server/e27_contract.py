"""Immutable E27 model and calibrated retrieval contract; no model execution."""
from copy import deepcopy
from pathlib import Path, PurePosixPath

from .model_bundle import (MAX_WEIGHTS_BYTES, ModelBundle, digest_json, read_object,
                           require_hash, sha256)
from .e27_retrieval import POLICY, THRESHOLD

WEIGHTS_SHA256 = '300b1c0ba907fa90ceacc38693713789cfb2a8b83248fd585fadb0d059381fdd'
WEIGHTS_BYTES = 118036485
MODEL_VERSION = 'e27-dinov3-smallplus256-fast32-gallery4nn-p2'
DEPENDENCIES = {'torch': '2.8.0', 'torchvision': '0.23.0', 'timm': '1.0.20',
                'numpy': '2.2.6', 'Pillow': '11.3.0', 'safetensors': '0.6.2'}
PREPROCESSING_SHA256 = '4bf999485d2ff0ec338bad5981ecf163a69c317f0dd7d40c6294978126ce7afa'


def source_hash():
    root = Path(__file__).parent
    files = ('e27_model.py', 'e27_adapter.py', 'e27_contract.py', 'e27_retrieval.py',
             'reidkit_adapter.py', 'reidkit_source/model.py', 'reidkit_source/data.py',
             'e25_rerank.py', 'retrieval.py', 'model_adapters.py', 'provider.py',
             'model_bundle.py', 'compact_model.py', 'fast_extract.py', 'service.py', 'batch.py')
    records = {name: sha256(root / name) for name in files}
    records.update({'stand5_native/' + p.relative_to(root / 'stand5_native').as_posix(): sha256(p)
                    for p in (root / 'stand5_native').rglob('*.py')})
    return digest_json(records)


def verify_native_sources():
    root = Path(__file__).parent
    for name, expected in CHECKPOINT_PROVENANCE['source_sha256'].items():
        if sha256(root / 'stand5_native' / name) != expected:
            raise ValueError('Native checkpoint source changed: ' + name)
    if sha256(root / 'fast_extract.py') != '4d030490ebabec5e054c38bb398669b791cb015922ecea8a8eb815d955288645':
        raise ValueError('Selected FP32 FastExtractor source changed')


def artifact(root, entry, name, limit):
    if not isinstance(entry, dict) or set(entry) != {'file', 'sha256'}:
        raise ValueError('E27 artifact requires file and sha256: ' + name)
    require_hash(entry['sha256'], name)
    relative = entry['file']
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('E27 artifact needs a safe relative path')
    raw = root / relative
    for part in (raw, *raw.parents):
        if part == root:
            break
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('Links are forbidden inside an E27 bundle')
    path = raw.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing E27 artifact: ' + name)
    if not 0 < path.stat().st_size <= limit or sha256(path) != entry['sha256']:
        raise ValueError('E27 artifact size/hash mismatch: ' + name)
    return path


def inspect_e27_bundle(root, manifest):
    verify_native_sources()
    expected_model = {'name': 'E27', 'version': MODEL_VERSION, 'dimension': 384}
    inference = {'adapter_revision': 1, 'preprocessing': 'native_stand5_checkpoint',
                 'score': POLICY['score_domain'], 'dependency_versions': DEPENDENCIES}
    if (digest_json(manifest.get('model')) != digest_json(expected_model)
            or digest_json(manifest.get('inference')) != digest_json(inference)):
        raise ValueError('E27 model/inference identity changed')
    artifacts = manifest.get('artifacts', {})
    if set(artifacts) != {'weights', 'config', 'calibration', 'recipe', 'protocol'}:
        raise ValueError('E27 needs one checkpoint, config, calibration, recipe and protocol')
    paths = {name: artifact(root, entry, name, MAX_WEIGHTS_BYTES if name == 'weights' else 1024 * 1024)
             for name, entry in artifacts.items()}
    if len(set(paths.values())) != 5:
        raise ValueError('E27 artifacts must be distinct')
    actual = {p for p in root.rglob('*') if p.is_file()}
    if actual != {root / 'bundle.json', *paths.values()}:
        raise ValueError('Extra or missing files in E27 model bundle')
    if (artifacts['weights'] != {'file': 'model.pth', 'sha256': WEIGHTS_SHA256}
            or paths['weights'].stat().st_size != WEIGHTS_BYTES
            or type(manifest.get('weights_bytes')) is not int or manifest['weights_bytes'] != WEIGHTS_BYTES):
        raise ValueError('E27 requires the exact selected stand-5 checkpoint')
    config, calibration, recipe, protocol = [read_object(paths[n]) for n in ('config', 'calibration', 'recipe', 'protocol')]
    if digest_json(config) != digest_json(frozen_config()):
        raise ValueError('E27 encoder/retrieval configuration changed')
    if digest_json(recipe) != digest_json(frozen_recipe()) or digest_json(protocol) != digest_json(frozen_protocol()):
        raise ValueError('E27 frozen selection or historical evidence changed')
    code_hash = source_hash()
    if (manifest.get('provenance') != {'inference_code_sha256': code_hash,
                                     'original_checkpoint_source_sha256': digest_json(CHECKPOINT_PROVENANCE['source_sha256']),
                                     'control_used_for_selection': False}
            or sha256(Path(__file__).with_name('reidkit_adapter.py')) != PREPROCESSING_SHA256):
        raise ValueError('E27 runtime/preprocessing provenance mismatch')
    expected_calibration = {'schema_version': 1, 'threshold_score': THRESHOLD,
        'score_domain': POLICY['score_domain'], 'model_sha256': WEIGHTS_SHA256,
        'inference_config_sha256': artifacts['config']['sha256'],
        'recipe_sha256': artifacts['recipe']['sha256'], 'protocol_sha256': artifacts['protocol']['sha256'],
        'source_sha256': code_hash, 'control_used_for_selection': False,
        'native_rejection_lock_sha256': NATIVE_REJECTION_LOCK_SHA256}
    if digest_json(calibration) != digest_json(expected_calibration):
        raise ValueError('E27 threshold is not bound to the frozen encoder and postprocessing')
    fingerprint = digest_json({'adapter': 'e27_compact', 'weights_sha256': WEIGHTS_SHA256,
        'config_sha256': artifacts['config']['sha256'], 'source_sha256': code_hash, 'dimension': 384})
    return ModelBundle(root, manifest, config, calibration, paths['weights'], fingerprint, None,
                       (), deepcopy(POLICY), THRESHOLD)


def frozen_config():
    return {'schema_version': 1, 'name': 'E27', 'encoder': deepcopy(ENCODER_CONFIG),
            'checkpoint_provenance': deepcopy(CHECKPOINT_PROVENANCE),
            'model_batch_size': 'requested_batch_max32', 'extraction': {'mode': 'fast', 'workers': 4,
                'precision': 'fp32', 'pin_memory': True, 'batch_size': 32, 'torch_cpu_threads': 2,
                'tf32': False, 'source_sha256': '4d030490ebabec5e054c38bb398669b791cb015922ecea8a8eb815d955288645'},
            'retrieval': deepcopy(POLICY)}


def frozen_recipe():
    return {'schema_version': 1, 'release': 'E27', 'selected_stand': 5, 'selected_epoch': 14,
            'checkpoint_sha256': WEIGHTS_SHA256, 'extraction': frozen_config()['extraction'], 'retrieval': deepcopy(POLICY), 'threshold': THRESHOLD,
            'native_rejection_lock_sha256': NATIVE_REJECTION_LOCK_SHA256,
            'native_study_result_sha256': NATIVE_STUDY_SHA256,
            'fast_selection_sha256': '89836703d9272af04d3be98da2f50fd36654dbdf80c55a6cc7e8d6995bb3c135',
            'fast_gpu_acceptance_sha256': '2307cd2cd5cdb56c35067a2246fb65dc0c87e780e0576fd8bf7cc9fce5d72790',
            'selection': 'dev ranking recipe; calibration maximum F1 subject to TNR>=0.95',
            'control_used_for_selection': False, 'historical_control_is_independent': False}


def frozen_protocol():
    return {'schema_version': 1, 'release': 'E27', 'original_stand5_gpu_completed': True,
            'new_training_performed': False, 'gpu_encoder_benchmark_reused': True,
            'selected_fast_extractor_gpu_benchmark_completed': True,
            'selected_fast_extractor_gpu_vectors_cpu_acceptance_completed': True,
            'integrated_e27_gpu_acceptance_performed': False,
            'native_GPU_vs_fast_GPU_feature_parity_established': True,
            'CPU_inference_vs_GPU_numerical_parity_established': False,
            'parity_scope': 'native saved GPU vs selected fast GPU vectors; cal/control batch/single; max_abs<=1e-5',
            'quality_scope': 'Official quality of selected FastExtractor GPU vectors; frozen postprocessing; historical control',
            'raw_embedding_export': 'unchanged float32 L2-normalized; query CSV then gallery CSV',
            'gallery_mixing': 'gallery only; simultaneous original descriptors; stable index ties; self excluded',
            'support': '3 neighbours of final top1 from original gallery; no other query input',
            'control_is_independent_holdout': False}

# Facts extracted from the immutable native checkpoint and selected CPU study.

ENCODER_CONFIG = {'schema_version': 1, 'timm_model': 'vit_small_plus_patch16_dinov3.lvd1689m', 'image_size': [256, 256], 'native_dimension': 384, 'embedding_dim': 384, 'pooling': 'native_pre_logits', 'projection': 'linear384_layernorm_l2', 'preprocessing': 'owned_reidkit_bicubic_letterbox_imagenet', 'flip_tta': False, 'training_amp': 'cuda_bfloat16_if_supported_else_float32', 'inference_precision': 'float32_autocast_off_tf32_off', 'pretrained_sha256': '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615'}
CHECKPOINT_PROVENANCE = {'source_sha256': {'compact_model.py': 'c3d9cc90c3e8af53612f77a9c76e443711c3efd377d2ee340d6d618904b97350', 'compact_learning.py': '8d1dc67d9487fe9883b2b1a3216e8440a983934bb623eb0b1cf3a43b58da37c9', 'reidkit_adapter.py': '4bf999485d2ff0ec338bad5981ecf163a69c317f0dd7d40c6294978126ce7afa', 'reidkit_source/model.py': '0050bceed5d093acebca038fff7c1c0c6ce3dc8547869ae8bc99df9c0d8cf686', 'frame_buffer.py': 'c6f4783f8449290a92e46a4e365a09fba8e2db800699a391e359ece32d843f1a', 'batch.py': '90a0a4e66b4ec54e9263e62c06760001e950f8ab3b399f41bdde046a1def5da3', 'retrieval.py': '7e4b86ecbb5d8e2c9a0a4d6979052571faaa2a19080a0ac553b2aeb1305832c9', 'e25_rerank.py': '6eef2ebd048e257737a67b0218d7e1e9658269c985125c55b23d9da5eff73620', 'validation.py': '39cc26c4f270be1d6bdb0a1f3ef2f5b2c8ae08d717b686055ec30b61723f402b', 'evaluation.py': 'eaf3ab458d8f58f12916a0ae7faf3f810d8f79160cdbe640602c23a60c347071'}, 'pretrained_sha256': '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615', 'initial_model_state_sha256': 'b278cbf71bf2ca0a3f998ebb7a7bc918dce56643af631b0b18fa46cab3c1a267', 'selection_basis': 'dev_raw_retrieval_mAP@10', 'control_used_for_selection': False}
NATIVE_REJECTION_LOCK_SHA256 = '9e560082161446f39ff1487f52b1dbc9b37ccba3a0e0026fc5099392980bc245'
NATIVE_STUDY_SHA256 = '0cff3322acd0fd2aa71295f652a857344645bffb06d5f50ebfe51833f84c9df0'
