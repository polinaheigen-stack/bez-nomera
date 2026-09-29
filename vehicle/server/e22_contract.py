"""Validated inference recipe for the supplied composite E22 checkpoint."""
from copy import deepcopy
import math
from pathlib import Path

from .model_bundle import digest_json, read_object, require_hash, sha256

SCHEMA = 'vehicle-reid-ensemble-checkpoint-v1'
AGGREGATION = 'L2-normalize each member embedding, weighted mean, L2-normalize'
SOURCE_COMMIT = '1025da3dd24d6325b4885fff388d86391b79140e'
PROTOCOL = 'imported_e22_research2030_threshold_v1'
MEMBERS = (('e15_convnext_tiny', 'convnext_tiny', 'ConvNeXt-Tiny'),
           ('e20_dinov2_vitb14', 'dinov2_vitb14', 'DINOv2-ViT-B/14'))


def source_hash():
    root = Path(__file__).parent
    source = read_object(root / 'e22_source' / 'SOURCE.json')
    if source.get('commit') != SOURCE_COMMIT:
        raise ValueError('Неизвестная версия исходного кода E22.')
    for name, record in source['files'].items():
        if sha256(root / 'e22_source' / name) != record['sha256']:
            raise ValueError(f'Исходный файл E22 изменён: {name}')
    paths = [root / 'e22_adapter.py', root / 'e22_contract.py']
    paths += sorted((root / 'e22_source').glob('*.py'))
    paths += [root / 'e22_source' / 'SOURCE.json']
    return digest_json({p.relative_to(root).as_posix(): sha256(p) for p in paths})


def validate_config(config):
    if (config.get('schema') != SCHEMA or config.get('embedding_dim') != 512
            or config.get('aggregation') != AGGREGATION or config.get('flip_tta') is not True):
        raise ValueError('Неподдерживаемый рецепт E22: нужны 512 признаков и flip-TTA.')
    members = config.get('members')
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError('E22 требует ровно два участника ансамбля.')
    for member, (name, backbone, architecture) in zip(members, MEMBERS):
        if not isinstance(member, dict) or member.get('name') != name:
            raise ValueError('Неверный состав или порядок моделей E22.')
        require_hash(member.get('source_sha256'), 'member.source_sha256')
        weight = member.get('weight')
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight != 1:
            raise ValueError('Проверенный рецепт E22 использует равные веса 1/1.')
        c = member.get('config')
        if not isinstance(c, dict) or c.get('debug_only'):
            raise ValueError('Нет рабочей конфигурации модели E22.')
        required = {'backbone': backbone, 'architecture': architecture, 'embedding_dim': 512,
                    'pooling': 'avg', 'bnneck_mode': 'shared', 'metric_head': 'linear',
                    'backbone_stride': 32, 'resize_mode': 'stretch', 'crop_padding': 0,
                    'crop_padding_ratio': 0.0, 'dino_part_stripes': 0}
        if any(c.get(k) != value for k, value in required.items()):
            raise ValueError(f'Неподдерживаемая конфигурация {name}.')
        if c.get('image_size') != [256, 128]:
            raise ValueError('E22 ожидает image_size=[256,128] в порядке высота, ширина.')
        if type(c.get('selected_epoch')) is not int or c['selected_epoch'] < 1:
            raise ValueError('Нет выбранной обученной эпохи E22.')
        if backbone == 'dinov2_vitb14' and (c.get('dino_variant') != 'b14' or c.get('dino_unfreeze_blocks') != 2):
            raise ValueError('Ожидается E20 DINOv2-B/14 с двумя обучаемыми блоками.')


def checkpoint_config(checkpoint):
    if (not isinstance(checkpoint, dict) or checkpoint.get('schema') != SCHEMA
            or checkpoint.get('experiment') != 'E-22' or checkpoint.get('debug_only')
            or checkpoint.get('member_weights') != [1.0, 1.0]):
        raise ValueError('Неизвестный формат объединённого checkpoint E22.')
    members = checkpoint.get('members')
    if not isinstance(members, list) or any(not isinstance(m, dict) for m in members):
        raise ValueError('В checkpoint отсутствуют участники E22.')
    config = {'schema': SCHEMA, 'embedding_dim': checkpoint.get('embedding_dim'),
              'aggregation': checkpoint.get('aggregation'), 'flip_tta': True,
              'members': [{k: deepcopy(m.get(k)) for k in ('name', 'weight', 'source_sha256', 'config')}
                          for m in members]}
    validate_config(config)
    return config


def validate_bundle_fields(manifest, config, calibration):
    validate_config(config)
    if manifest['model']['dimension'] != 512:
        raise ValueError('E22 возвращает 512 признаков.')
    provenance = manifest['provenance']
    if (provenance.get('source_commit') != SOURCE_COMMIT
            or provenance.get('source_role') != 'supplied_reproduction_source'
            or provenance.get('inference_code_sha256') != source_hash()):
        raise ValueError('Код извлечения E22 не совпадает с пакетом.')
    require_hash(provenance.get('handoff_metadata_sha256'), 'handoff_metadata_sha256')
    if (calibration.get('calibration_protocol') != PROTOCOL
            or calibration.get('model_sha256') != manifest['artifacts']['weights']['sha256']
            or calibration.get('inference_config_sha256') != manifest['artifacts']['config']['sha256']
            or calibration.get('code_hash') != provenance['inference_code_sha256']):
        raise ValueError('Калибровка E22 не соответствует весам, обработке или коду.')
    original = calibration.get('original')
    if (not isinstance(original, dict) or original.get('score_mode') != 'cosine'
            or original.get('query_expansion', {}).get('k') != 0
            or original.get('selected', {}).get('threshold') != calibration.get('threshold_cosine')):
        raise ValueError('Порог должен совпадать с исходной калибровкой E22.')
