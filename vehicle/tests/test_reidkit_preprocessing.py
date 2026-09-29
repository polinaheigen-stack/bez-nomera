"""Ensure the optimized crop path preserves the trained preprocessing exactly."""
import io
import unittest

import numpy as np
from PIL import Image
import torch

from vehicle.server.reidkit_adapter import image_tensor


def previous_image_tensor(image, bbox, image_size):
    """Frozen pre-optimization transform, including RGB copy of the full frame."""
    x, y, w, h = map(int, bbox)
    crop = image.convert('RGB').crop((x, y, x+w, y+h))
    height, width = image_size
    scale = min(width / crop.width, height / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (width, height), (124, 116, 104))
    canvas.paste(resized, ((width-resized.width)//2, (height-resized.height)//2))
    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


class ReIDKitPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_exact_transform(self, image, bbox, image_size):
        before = image.tobytes()
        original_size, original_mode = image.size, image.mode
        expected = previous_image_tensor(image, bbox, image_size)
        actual = image_tensor(image, bbox, image_size)
        self.assertTrue(torch.equal(expected, actual), (original_mode, original_size, bbox, image_size))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (3, *image_size))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(image.tobytes(), before)
        self.assertEqual((image.size, image.mode), (original_size, original_mode))

    def test_exact_pixels_for_modes_boundaries_and_model_resolutions(self):
        rng = np.random.default_rng(20260928)
        # Each source varies spatially and by channel, exposing crop, colour,
        # padding and rounding mistakes rather than checking constant images.
        shapes = {'RGB': (19, 31, 3), 'RGBA': (19, 31, 4), 'L': (19, 31)}
        boxes = [(0, 0, 31, 19), (3, 2, 21, 13), (25, 14, 6, 5), (30, 18, 1, 1), (0, 0, 1, 19), (0, 18, 31, 1)]
        for mode, shape in shapes.items():
            image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
            self.assertEqual(image.mode, mode)
            for bbox in boxes:
                for image_size in ((336, 336), (384, 384)):
                    with self.subTest(mode=mode, bbox=bbox, size=image_size):
                        self.assert_exact_transform(image, bbox, image_size)

    def test_one_pixel_source(self):
        for mode, colour in (('RGB', (7, 211, 63)), ('RGBA', (127, 8, 199, 12)), ('L', 73)):
            for image_size in ((336, 336), (384, 384)):
                with self.subTest(mode=mode, size=image_size):
                    self.assert_exact_transform(Image.new(mode, (1, 1), colour), (0, 0, 1, 1), image_size)

    def test_unloaded_jpeg_and_png_inputs_match_loaded_reference(self):
        pixels = np.random.default_rng(17).integers(0, 256, (51, 83, 3), dtype=np.uint8)
        for format_name in ('JPEG', 'PNG'):
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format=format_name)
            for image_size in ((336, 336), (384, 384)):
                with self.subTest(format=format_name, size=image_size):
                    with Image.open(io.BytesIO(buffer.getvalue())) as original:
                        original.load()
                        expected = previous_image_tensor(original, (67, 42, 16, 9), image_size)
                    with Image.open(io.BytesIO(buffer.getvalue())) as lazy_image:
                        actual = image_tensor(lazy_image, (67, 42, 16, 9), image_size)
                    self.assertTrue(torch.equal(expected, actual))

    def test_invalid_bbox_remains_rejected(self):
        image = Image.new('RGB', (31, 19))
        for bbox in ((-1, 0, 1, 1), (0, 0, 0, 1), (30, 18, 2, 1), (0, 0, 1, 20), (True, 0, 1, 1), (0, 0, 1.0, 1)):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                image_tensor(image, bbox, (336, 336))


if __name__ == '__main__':
    unittest.main()
