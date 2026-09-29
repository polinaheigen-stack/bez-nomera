"""Offline inference bridge for a trained single reidkit model, never E22 weights."""
from contextlib import nullcontext
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .reidkit_source.model import ReIDModel


def image_tensor(image, bbox, image_size):
    # Exact VehicleDataset(training=False) recipe for valid web input bboxes.
    if len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bbox):
        raise ValueError('Bounding box requires four integers')
    x, y, w, h = map(int, bbox)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.width or y+h > image.height:
        raise ValueError('Bounding box is outside original image')
    # RGB inputs already have the required pixels. crop() creates an independent
    # image, so avoid copying the whole decoded frame before cropping it.
    rgb = image if image.mode == 'RGB' else image.convert('RGB')
    crop = rgb.crop((x, y, x+w, y+h))
    height, width = image_size
    scale = min(width / crop.width, height / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (width, height), (124, 116, 104))
    canvas.paste(resized, ((width-resized.width)//2, (height-resized.height)//2))
    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


class ReIDKitAdapter:
    dimension = 512

    def __init__(self, bundle, device):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; fallback is forbidden')
        checkpoint = torch.load(bundle.weights_path, map_location='cpu', weights_only=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get('kind') != 'trained_vehicle_reid'
                or checkpoint.get('format_version') != 1 or checkpoint.get('config') != bundle.config
                or checkpoint.get('provenance') != bundle.manifest['provenance']['training']
                or type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] < 0
                or type(checkpoint.get('num_classes')) is not int or checkpoint['num_classes'] < 2):
            raise ValueError('Checkpoint/config/provenance mismatch or DEBUG checkpoint')
        self.config = bundle.config
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = ReIDModel(self.config, checkpoint['num_classes'], initial_weights=None)
        state = checkpoint.get('model_state')
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError('Checkpoint has missing/extra state keys')
        for key, value in state.items():
            if (not torch.is_tensor(value) or value.shape != expected[key].shape or value.dtype != expected[key].dtype
                    or (value.is_floating_point() and not torch.isfinite(value).all().item())):
                raise ValueError(f'Invalid checkpoint tensor: {key}')
        model.load_state_dict(state, strict=True)
        self.model = model.requires_grad_(False).eval().to(self.device)

    def embed(self, image, bbox):
        return self.embed_batch([image], [bbox])[0]

    @torch.inference_mode()
    def embed_batch(self, images, bboxes):
        if len(images) != len(bboxes):
            raise ValueError('Images and bounding boxes differ in length')
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        tensor = torch.stack([image_tensor(image, box, self.config['image_size']) for image, box in zip(images, bboxes)]).to(self.device)
        # Same AMP policy as reidkit.engine.extract, including optional flip TTA.
        amp = torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device.type == 'cuda' else nullcontext()
        with amp:
            vectors = self.model(tensor)['embedding']
            if self.config.get('flip_tta', False):
                vectors = F.normalize(vectors.float() + self.model(torch.flip(tensor, dims=[3]))['embedding'].float(), dim=1)
        return F.normalize(vectors.float(), dim=1).cpu().numpy().astype(np.float32)
