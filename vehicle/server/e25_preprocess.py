"""Native PIL geometry + equivalent contiguous float32 CPU normalization.

Each call owns its output. No image, crop, tensor or embedding cache. The only
reused tensors are immutable ImageNet mean/std constants. Numerical equivalence
must still pass the complete GPU/calibration gates before any production use.
"""
import numpy as np
from PIL import Image
import torch

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device='cpu')[:, None, None]
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device='cpu')[:, None, None]


def normalized_contiguous(canvas):
    """Same three float32 operations, operating on a private contiguous CHW."""
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('The frozen native preprocessing requires default float32.')
    # uint8 -> float32 is exact for all 256 channel values. Moving the layout
    # conversion ahead of normalization avoids strided broadcast arithmetic.
    tensor = torch.from_numpy(np.array(canvas, dtype=np.uint8)).permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    return tensor.div_(255.0).sub_(_MEAN).div_(_STD)


def image_tensor(image, bbox, image_size):
    # Deliberately preserve the native validation, geometry and PIL operations.
    if len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bbox):
        raise ValueError('Bounding box requires four integers')
    x, y, w, h = map(int, bbox)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.width or y+h > image.height:
        raise ValueError('Bounding box is outside original image')
    rgb = image if image.mode == 'RGB' else image.convert('RGB')
    crop = rgb.crop((x, y, x+w, y+h))
    height, width = image_size
    scale = min(width / crop.width, height / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (width, height), (124, 116, 104))
    canvas.paste(resized, ((width-resized.width)//2, (height-resized.height)//2))
    return normalized_contiguous(canvas)


def batch_one(image, bbox, image_size):
    # Tensor is already contiguous CHW. The view has the same native contiguous
    # BCHW strides as torch.stack([native_image_tensor(...)]) without a copy.
    return image_tensor(image, bbox, image_size).unsqueeze(0)
