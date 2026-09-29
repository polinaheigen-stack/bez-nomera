"""Original-image crops and camera-diverse identity batches; metadata is not model input."""
from __future__ import annotations

import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset, Sampler


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    required = {"image_id", "x", "y", "w", "h"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing bbox columns in {path}")
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id) or image_id in seen:
            raise ValueError(f"Unsafe or repeated image_id: {image_id!r}")
        seen.add(image_id)
        x, y, w, h = (int(row[key]) for key in ("x", "y", "w", "h"))
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError(f"Invalid bbox for {image_id}")
    return rows


class VehicleDataset(Dataset):
    def __init__(self, csv_path, data_root, image_size, training=False, classes=None,
                 erasing_probability=0.0):
        self.rows = read_csv(csv_path)
        self.root = Path(data_root).resolve()
        self.height, self.width = map(int, image_size)
        if min(self.height, self.width) <= 0:
            raise ValueError("image_size must be [height, width] with positive values")
        self.training = bool(training)
        self.classes = {str(k): int(v) for k, v in (classes or {}).items()}
        self.labels = [self.classes.get(str(r.get("vehicle_id", "")), -1) for r in self.rows]
        self.cameras = [str(r.get("camera_id", "unknown")) for r in self.rows]
        self.erasing_probability = float(erasing_probability)
        if not 0 <= self.erasing_probability <= 1:
            raise ValueError("erasing_probability must be in [0, 1]")
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / "images" / f"{row['image_id']}.jpg") as original:
            original = original.convert("RGB")
            x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(original.width, x + w), min(original.height, y + h)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"BBox does not intersect image {row['image_id']}")
            crop = original.crop((x0, y0, x1, y1))
        if self.training:
            if random.random() < 0.5:
                crop = ImageOps.mirror(crop)
            crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.85, 1.15))
        scale = min(self.width / crop.width, self.height / crop.height)
        resized = crop.resize((max(1, round(crop.width * scale)),
                               max(1, round(crop.height * scale))), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.width, self.height), (124, 116, 104))
        canvas.paste(resized, ((self.width - resized.width) // 2,
                              (self.height - resized.height) // 2))
        tensor = torch.from_numpy(np.array(canvas, dtype=np.float32)).permute(2, 0, 1) / 255.0
        tensor = (tensor - self.mean) / self.std
        # Optional, disabled by default: at most a small part of the canvas is hidden.
        if self.training and random.random() < self.erasing_probability:
            eh = max(1, round(self.height * random.uniform(0.08, 0.20)))
            ew = max(1, round(self.width * random.uniform(0.08, 0.20)))
            y0 = random.randint(0, self.height - eh)
            x0 = random.randint(0, self.width - ew)
            tensor[:, y0:y0 + eh, x0:x0 + ew] = 0
        return {"image": tensor, "label": self.labels[index], "image_id": row["image_id"]}


class IdentityBatchSampler(Sampler):
    """P distinct vehicle identities, K images each, preferring two or more cameras."""
    def __init__(self, labels, cameras, identities_per_batch, images_per_identity, seed):
        if len(labels) != len(cameras):
            raise ValueError("labels and cameras have different lengths")
        self.p, self.k = int(identities_per_batch), int(images_per_identity)
        if self.p < 2 or self.k < 2:
            raise ValueError("Triplet batches require P >= 2 identities and K >= 2 images")
        self.groups = defaultdict(lambda: defaultdict(list))
        for index, (label, camera) in enumerate(zip(labels, cameras)):
            if int(label) < 0:
                raise ValueError("Training sampler requires explicit class labels")
            self.groups[int(label)][str(camera)].append(index)
        if len(self.groups) < self.p:
            raise ValueError("Not enough distinct identities for requested batch")
        self.seed, self.epoch = int(seed), 0
        self.num_batches = max(1, math.ceil(len(labels) / (self.p * self.k)))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        identity_pool = []
        for _ in range(self.num_batches):
            chosen = []
            while len(chosen) < self.p:
                if not identity_pool:
                    identity_pool = sorted(self.groups)
                    rng.shuffle(identity_pool)
                label = identity_pool.pop()
                if label not in chosen:
                    chosen.append(label)
            batch = []
            for label in chosen:
                cameras = list(self.groups[label])
                rng.shuffle(cameras)
                selected = [rng.choice(self.groups[label][camera]) for camera in cameras[:self.k]]
                all_indices = [i for entries in self.groups[label].values() for i in entries]
                remaining = [i for i in all_indices if i not in selected]
                rng.shuffle(remaining)
                selected.extend(remaining[:max(0, self.k - len(selected))])
                while len(selected) < self.k:
                    selected.append(rng.choice(all_indices))
                rng.shuffle(selected)
                batch.extend(selected)
            yield batch
