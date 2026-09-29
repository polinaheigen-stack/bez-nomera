from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageFilter


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "BoundingBox":
        return cls(*(int(row[key]) for key in ("x", "y", "w", "h")))


def crop_vehicle(image: Image.Image, bbox: BoundingBox, padding: int = 0) -> Image.Image:
    """Crop the supplied vehicle box. Detection is intentionally out of scope."""
    if bbox.w <= 0 or bbox.h <= 0:
        raise ValueError("Bounding box dimensions must be positive")

    left = max(0, bbox.x - padding)
    top = max(0, bbox.y - padding)
    right = min(image.width, bbox.x + bbox.w + padding)
    bottom = min(image.height, bbox.y + bbox.h + padding)
    if left >= right or top >= bottom:
        raise ValueError("Bounding box is outside image bounds")
    return image.convert("RGB").crop((left, top, right, bottom))


def mask_regions(image: Image.Image, regions: Iterable[BoundingBox]) -> Image.Image:
    """Blur explicitly supplied forbidden regions; never runs OCR."""
    result = image.convert("RGB").copy()
    blurred = result.filter(ImageFilter.GaussianBlur(radius=12))
    for region in regions:
        left = max(0, region.x)
        top = max(0, region.y)
        right = min(result.width, region.x + region.w)
        bottom = min(result.height, region.y + region.h)
        if left < right and top < bottom:
            result.paste(blurred.crop((left, top, right, bottom)), (left, top))
    return result
