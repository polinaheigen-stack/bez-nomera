from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .preprocess import BoundingBox, crop_vehicle


IMAGE_SIZE = (256, 128)  # height, width; standard vehicle Re-ID aspect ratio
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
LETTERBOX_FILL = tuple(int(round(value * 255)) for value in MEAN)


def resize_crop(
    image: Image.Image,
    image_size: tuple[int, int],
    resize_mode: str = "stretch",
) -> Image.Image:
    """Resize a crop, optionally preserving its aspect ratio with mean padding."""
    if resize_mode == "stretch":
        return image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    if resize_mode != "letterbox":
        raise ValueError("resize_mode must be 'stretch' or 'letterbox'")

    target_height, target_width = image_size
    scale = min(target_width / image.width, target_height / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), LETTERBOX_FILL)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def prepare_crop(
    image: Image.Image,
    bbox: BoundingBox | None = None,
    training: bool = False,
    padding: int = 0,
    padding_ratio: float = 0.0,
    augmentation: str = "baseline",
    image_size: tuple[int, int] = IMAGE_SIZE,
    resize_mode: str = "stretch",
) -> Image.Image:
    if bbox is not None:
        if padding_ratio < 0:
            raise ValueError("padding_ratio must be non-negative")
        ratio_padding = round(max(bbox.w, bbox.h) * padding_ratio)
        image = crop_vehicle(image, bbox, padding=padding + ratio_padding)
    image = image.convert("RGB")
    if training:
        if augmentation not in {"baseline", "strong"}:
            raise ValueError(f"unsupported augmentation profile: {augmentation}")
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if augmentation == "baseline":
            if random.random() < 0.25:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        else:
            if random.random() < 0.7:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.6:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.6:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
    return resize_crop(image, image_size, resize_mode=resize_mode)


def image_to_tensor(image: Image.Image):
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def apply_random_erasing(tensor, probability: float = 0.25):
    """Apply lightweight random erasing after normalization to a training tensor."""
    import torch

    if random.random() >= probability:
        return tensor
    _, height, width = tensor.shape
    area = height * width
    target_area = random.uniform(0.02, 0.15) * area
    aspect = random.uniform(0.3, 3.3)
    erase_h = max(1, min(height, int(round((target_area * aspect) ** 0.5))))
    erase_w = max(1, min(width, int(round((target_area / aspect) ** 0.5))))
    if erase_h >= height or erase_w >= width:
        return tensor
    top = random.randint(0, height - erase_h)
    left = random.randint(0, width - erase_w)
    fill = torch.empty((tensor.shape[0], erase_h, erase_w), dtype=tensor.dtype)
    fill.normal_(0.0, 0.5)
    tensor[:, top : top + erase_h, left : left + erase_w] = fill
    return tensor
