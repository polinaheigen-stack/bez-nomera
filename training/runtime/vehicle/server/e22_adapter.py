"""Offline E15+E20 loader. No camera metadata, downloads or train-time fallback."""
import numpy as np
import torch
from PIL import Image

from .e22_contract import checkpoint_config
from .e22_source.convnext import ConvNeXtTiny
from .e22_source.dinov2 import DinoV2ReID
from .e22_source.image_utils import image_to_tensor, prepare_crop
from .e22_source.preprocess import BoundingBox


def load_encoder_state(model, state):
    if not isinstance(state, dict):
        raise ValueError('В checkpoint E22 отсутствует словарь весов.')
    expected = model.state_dict()
    missing = set(expected) - set(state)
    # The supplied linear classifier was used only during training.
    extra = set(state) - set(expected) - {'classifier.weight', 'classifier.bias'}
    bad = [name for name, value in state.items()
           if not torch.is_tensor(value) or (value.is_floating_point() and not torch.isfinite(value).all().item())]
    mismatch = [name for name in set(expected) & set(state)
                if not torch.is_tensor(state[name]) or expected[name].shape != state[name].shape
                or expected[name].dtype != state[name].dtype]
    if missing or extra or bad or mismatch:
        raise ValueError(f'Несовместимые веса E22: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}, '
                         f'invalid={bad[:3]}, mismatch={mismatch[:3]}')
    model.load_state_dict({name: state[name] for name in expected}, strict=True)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError('E22 вернул некорректный или нулевой вектор.')
    return values / norms


class E22Adapter:
    dimension = 512

    def __init__(self, bundle, device):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Запрошена CUDA, но видеокарта недоступна.')
        # Tensor-only loading: the checkpoint never supplies executable Python.
        checkpoint = torch.load(bundle.weights_path, map_location='cpu', weights_only=True)
        if checkpoint_config(checkpoint) != bundle.config:
            raise ValueError('Конфигурация пакета не совпадает с checkpoint E22.')
        if checkpoint.get('calibration') != bundle.calibration['original']:
            raise ValueError('Калибровка пакета не совпадает с checkpoint E22.')
        models = []
        for index, member in enumerate(checkpoint['members']):
            c = member['config']
            kwargs = dict(embedding_dim=512, num_classes=0, bnneck_mode=c['bnneck_mode'],
                          metric_head=c['metric_head'], metric_scale=c['metric_scale'], metric_margin=c['metric_margin'])
            if index == 0:
                model = ConvNeXtTiny(**kwargs)
            else:
                model = DinoV2ReID(**kwargs, variant=c['dino_variant'],
                                  unfreeze_blocks=c['dino_unfreeze_blocks'], part_stripes=c['dino_part_stripes'])
            load_encoder_state(model, member['model'])
            model.requires_grad_(False).eval().to(self.device)
            models.append(model)
        # Publish only the fully loaded pair. The checkpoint tensors can be freed.
        self.models = tuple(models)

    def embed(self, image, bbox):
        return self.embed_batch([image], [bbox])[0]

    def embed_batch(self, images, bboxes):
        if len(images) != len(bboxes):
            raise ValueError('Число фотографий и рамок E22 должно совпадать.')
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        crops = []
        for image, bbox in zip(images, bboxes):
            if len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bbox):
                raise ValueError('Рамка E22 должна содержать четыре целых числа.')
            x, y, w, h = bbox
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.width or y+h > image.height:
                raise ValueError('Рамка E22 выходит за границы фотографии.')
            crops.append(prepare_crop(image, BoundingBox(x, y, w, h),
                                      image_size=(256, 128), resize_mode='stretch'))
        tensor = torch.stack([image_to_tensor(crop) for crop in crops]).to(self.device)
        flipped = torch.stack([image_to_tensor(crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
                               for crop in crops]).to(self.device)
        members = []
        with torch.inference_mode():
            for model in self.models:
                # Same order as upstream encode_batch: normalize each view inside
                # the model, average views, then normalize on the CPU in float32.
                vectors = ((model(tensor) + model(flipped)) / 2).float().cpu().numpy()
                members.append(normalize_rows(vectors))
        # Upstream ensemble_embeddings.py normalizes each member again before mean.
        return normalize_rows(np.mean(np.stack([normalize_rows(v) for v in members]), axis=0)).astype(np.float32)
