"""Controlled stand4/5 training: shared supervised objective, optional E25 relations."""
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import ImageEnhance, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from .compact_model import (CONFIG, CompactModel, PRETRAINED_SHA256, amp_context,
                            dependencies, source_identity, state_digest, verify_pretrained)
from .frame_buffer import FrameBuffer
from .model_bundle import digest_json, sha256
from .reidkit_adapter import image_tensor
from .service import atomic_json

TRAIN_CONFIG = {'epochs': 20, 'P': 8, 'K': 4, 'seed': 20260929,
                'lr_backbone': 2e-5, 'lr_head': 1e-3, 'weight_decay': .01,
                'warmup_epochs': 2, 'label_smoothing': .1, 'kd_temperature': .1}
AUGMENTATION = {'horizontal_flip_probability': .5, 'brightness': [.9, 1.1],
                'contrast': [.9, 1.1], 'saturation': [.9, 1.1], 'erasing': False,
                'bbox_jitter': False, 'supervised_view': 'weak_augmented_original_bbox',
                'kd_view': 'clean_original_bbox_matching_cached_teacher'}


def resolve_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError('Profile must be an explicit JSON object')
    stand = profile.get('stand_id', profile.get('stand'))
    if type(stand) is not int or stand not in (4, 5) or ('stand' in profile and profile['stand'] != stand):
        raise ValueError('Expected stand_id 4 or 5')
    expected = 0. if stand == 4 else 1.
    if type(profile.get('kd_weight')) not in (int, float) or profile['kd_weight'] != expected:
        raise ValueError('Only stand5 enables KL distillation with weight1')
    for key, value in TRAIN_CONFIG.items():
        if key in profile and profile[key] != value:
            raise ValueError('Controlled experiment requires fixed training setting: ' + key)
    return stand, expected


def pk_batches(labels, *, P=8, K=4, seed=20260929, epoch=1):
    """Uniform identity sampling. A PK epoch is sampled, not one visit per image."""
    if P < 2 or K < 2:
        raise ValueError('PK sampling requires at least two identities and two images per identity')
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    if len(groups) < P:
        raise ValueError(f'PK{P}x{K} requires at least {P} distinct training identities')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    identities = np.asarray(sorted(groups))
    for _ in range(max(1, math.ceil(len(labels) / (P * K)))):
        selected = rng.choice(identities, P, replace=False)
        yield [int(index) for label in selected
               for index in rng.choice(groups[int(label)], K, replace=len(groups[int(label)]) < K)]


def softmargin_batch_hard(embeddings, labels):
    """Softplus(hardest positive distance - nearest negative distance), no self pairs."""
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError('Triplet expects [N,D] embeddings and [N] labels')
    values = F.normalize(embeddings.float(), dim=1)
    distance = torch.cdist(values, values, p=2)
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives, negatives = same & ~diagonal, ~same
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not bool(valid.any()):
        raise ValueError('Triplet batch has no anchor with both a positive and negative')
    hard_positive = distance.masked_fill(~positives, -torch.inf).max(dim=1).values[valid]
    hard_negative = distance.masked_fill(~negatives, torch.inf).min(dim=1).values[valid]
    return F.softplus(hard_positive - hard_negative).mean()


def relational_kl(student, teacher, *, temperature=.1):
    """Row KL(teacher neighbours || student neighbours), self excluded; teacher detached."""
    if student.ndim != 2 or teacher.ndim != 2 or len(student) != len(teacher) or len(student) < 2:
        raise ValueError('Relational KL requires >=2 paired examples; feature dimensions may differ')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite KL temperature required')
    s, t = F.normalize(student.float(), dim=1), F.normalize(teacher.detach().float(), dim=1)
    count = len(s)
    mask = ~torch.eye(count, dtype=torch.bool, device=s.device)
    # Select off-diagonal elements instead of multiplying zero probabilities by -inf.
    sl = (s @ s.T)[mask].reshape(count, count - 1) / temperature
    tl = (t @ t.T)[mask].reshape(count, count - 1) / temperature
    return F.kl_div(F.log_softmax(sl, dim=1), F.softmax(tl, dim=1), reduction='batchmean')


def augmented_tensor(image, bbox, rng):
    x, y, w, h = bbox
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.width or y + h > image.height:
        raise ValueError('Training bbox outside original image')
    crop = image.crop((x, y, x + w, y + h)).convert('RGB')
    if rng.random() < .5:
        crop = ImageOps.mirror(crop)
    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(.9, 1.1))
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(.9, 1.1))
    crop = ImageEnhance.Color(crop).enhance(rng.uniform(.9, 1.1))
    return image_tensor(crop, (0, 0, crop.width, crop.height), CONFIG['image_size'])


def learning_rate_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    remaining = max(1, total_steps - warmup_steps - 1)
    progress = min(1., max(0., (step - warmup_steps) / remaining))
    return .5 * (1 + math.cos(math.pi * progress))


def train_model(profile, train_rows, image_paths, labels, pretrained_path, pretrained_sha,
                output, dev_evaluate, teacher_vectors=None, *, device='cuda',
                allow_cpu_test=False, model_factory=None, _test_config=None):
    """Return a completed self-describing checkpoint; no resume or partial success.

    dev_evaluate(model.eval(), epoch) must use only the separate DEV split and
    return a finite fraction under key 'mAP@10'. Input arrays are ordered by rows.
    Test-only injection is rejected unless BOTH an explicit flag and tiny factory
    are supplied; its checkpoint is permanently rejected by CompactProvider.
    """
    stand, kd_weight = resolve_profile(profile)
    synthetic = allow_cpu_test and model_factory is not None and device == 'cpu'
    if (allow_cpu_test or model_factory is not None or _test_config is not None) and not synthetic:
        raise ValueError('Test overrides require explicit CPU mode plus a tiny model factory')
    if not synthetic and device != 'cuda':
        raise ValueError('Production training requires CUDA; no CPU fallback')
    settings = {**TRAIN_CONFIG, **(_test_config or {})} if synthetic else deepcopy(TRAIN_CONFIG)
    if not train_rows or len({row['image_id'] for row in train_rows}) != len(train_rows):
        raise ValueError('Training image IDs must be nonempty and unique')
    if not callable(dev_evaluate):
        raise ValueError('A separate DEV retrieval evaluation callback is required')
    if (stand == 4 and teacher_vectors is not None) or (stand == 5 and teacher_vectors is None):
        raise ValueError('Teacher targets belong only to stand5 and are mandatory there')
    target = None
    if teacher_vectors is not None:
        target = np.asarray(teacher_vectors)
        if (target.dtype != np.float32 or target.shape != (len(train_rows), 1024) or not np.isfinite(target).all()
                or not np.allclose(np.linalg.norm(target, axis=1), 1, atol=1e-4)):
            raise ValueError('Ordered teacher targets must be normalized float32 [N,1024]')
        target = target.copy()  # Cache cannot change under a training step.
    paths = ({str(row['image_id']): Path(path) for row, path in zip(train_rows, image_paths)}
             if not isinstance(image_paths, dict) else {str(k): Path(v) for k, v in image_paths.items()})
    if not isinstance(image_paths, dict) and len(image_paths) != len(train_rows):
        raise ValueError('Image paths and rows differ in length')
    try:
        identity_values = [str(labels[row['image_id']]) for row in train_rows]
    except KeyError as error:
        raise ValueError('Missing training identity') from error
    if any(not identity for identity in identity_values):
        raise ValueError('Empty training identity')
    classes = {identity: index for index, identity in enumerate(sorted(set(identity_values)))}
    encoded = np.asarray([classes[value] for value in identity_values], dtype=np.int64)
    list(pk_batches(encoded, P=settings['P'], K=settings['K'], seed=settings['seed'], epoch=1))
    provenance_input = deepcopy(profile.get('provenance', {}))
    expected = provenance_input.get('data_contract', {}).get('images', {})
    if synthetic and not expected:
        expected = {row['image_id']: {'sha256': sha256(paths[row['image_id']])} for row in train_rows}
    if any(row['image_id'] not in expected or 'sha256' not in expected[row['image_id']] for row in train_rows):
        raise ValueError('Frozen input-image hashes are required in provenance.data_contract.images')
    def verify_inputs():
        for row in train_rows:
            if sha256(paths[row['image_id']]) != expected[row['image_id']]['sha256']:
                raise ValueError('Training image differs from frozen data contract: ' + row['image_id'])
    verify_inputs()
    if not synthetic:
        verify_pretrained(pretrained_path, pretrained_sha)
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; training did not start')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = {'status': 'running', 'training_finished': False, 'stand_id': stand,
               'settings': settings, 'kd_weight': kd_weight, 'augmentation': AUGMENTATION,
               'test_only': synthetic, 'epochs': [], 'selected_epoch': None,
               'selection_basis': 'dev_raw_retrieval_mAP@10', 'error': None}
    atomic_json(output / 'training-history.json', history)
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(settings['seed'])
        np.random.seed(settings['seed'])
        torch.manual_seed(settings['seed'])
        if device == 'cuda':
            torch.cuda.manual_seed_all(settings['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model = model_factory() if synthetic else CompactModel(pretrained_path, pretrained_sha)
        if synthetic and not getattr(model, 'test_only', False):
            raise ValueError('CPU tests must use a model explicitly marked test_only')
        model = model.to(device)
        initial_sha = state_digest(model.state_dict())
        classifier = nn.Linear(384, len(classes)).to(device)
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': settings['lr_backbone']},
            {'params': [*model.projection.parameters(), *classifier.parameters()], 'lr': settings['lr_head']}],
            weight_decay=settings['weight_decay'])
        steps_per_epoch = max(1, math.ceil(len(train_rows) / (settings['P'] * settings['K'])))
        total_steps, warm_steps = settings['epochs'] * steps_per_epoch, settings['warmup_epochs'] * steps_per_epoch
        sources = source_identity()
        versions = dependencies() if not synthetic else {'synthetic_test': str(torch.__version__)}
        provenance = {**provenance_input, 'source_sha256': sources, 'dependencies': versions,
                      'pretrained_sha256': pretrained_sha, 'initial_model_state_sha256': initial_sha,
                      'training': settings, 'augmentation': deepcopy(AUGMENTATION), 'kd_weight': kd_weight,
                      'teacher_targets_sha256': hashlib_array(target) if target is not None else None,
                      'train_ids_sha256': digest_json([row['image_id'] for row in train_rows]),
                      'train_identity_mapping_sha256': digest_json(dict(zip([row['image_id'] for row in train_rows], identity_values))),
                      'selection_basis': 'dev_raw_retrieval_mAP@10', 'control_used_for_selection': False,
                      'independent_quality_claim_allowed': False,
                      'loss': 'CE(label_smoothing=.1)+softmargin_batchhard+kd_weight*KL(teacher||student), T=.1, no T^2 multiplier',
                      'epoch_definition': 'ceil(N/(P*K)) uniformly sampled identity batches; repetitions permitted; not a complete unique-image pass'}
        best, best_epoch, global_step = -float('inf'), None, 0
        best_path = output / 'selected-state.pt'
        for epoch in range(1, settings['epochs'] + 1):
            model.train()
            classifier.train()
            started, totals, drawn = time.perf_counter(), dict(ce=0., triplet=0., kd=0., total=0.), 0
            batches = list(pk_batches(encoded, P=settings['P'], K=settings['K'], seed=settings['seed'], epoch=epoch))
            sequence = [{**train_rows[index], 'path': paths[train_rows[index]['image_id']], 'target_index': index}
                        for batch in batches for index in batch]
            rng = random.Random(settings['seed'] + epoch * 1_000_003)
            pending = []
            def step(batch):
                nonlocal global_step, drawn
                indices, augmented, clean = zip(*batch)
                augmented = torch.stack(augmented).to(device)
                y = torch.from_numpy(encoded[list(indices)]).to(device)
                factor = learning_rate_factor(global_step, total_steps, warm_steps)
                for group, base_lr in zip(optimizer.param_groups, (settings['lr_backbone'], settings['lr_head'])):
                    group['lr'] = base_lr * factor
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device):
                    features = model(augmented)
                    logits = classifier(features)
                ce = F.cross_entropy(logits.float(), y, label_smoothing=settings['label_smoothing'])
                triplet = softmargin_batch_hard(features, y)
                kd = features.sum() * 0.
                if kd_weight:
                    # Keep supervised stochastic state identical across profiles. KD
                    # has its own clean view; targets were computed on that same bbox.
                    devices = [torch.device(device).index or torch.cuda.current_device()] if device == 'cuda' else []
                    with torch.random.fork_rng(devices=devices), amp_context(device):
                        clean_features = model(torch.stack(clean).to(device))
                    kd = relational_kl(clean_features, torch.from_numpy(target[list(indices)]).to(device),
                                       temperature=settings['kd_temperature'])
                loss = ce + triplet + kd_weight * kd
                if not all(torch.isfinite(value).item() for value in (ce, triplet, kd, loss)):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_([*model.parameters(), *classifier.parameters()], 1., error_if_nonfinite=True)
                optimizer.step()
                for name, value in (('ce', ce), ('triplet', triplet), ('kd', kd), ('total', loss)):
                    totals[name] += float(value.detach()) * len(indices)
                drawn += len(indices)
                global_step += 1
            with FrameBuffer(sequence, ahead=1, verify_hash=True) as frames:
                for frame in frames:
                    image_id = frame.row['image_id']
                    if frame.sha256 != expected[image_id]['sha256']:
                        raise ValueError('Training frame changed before decode: ' + image_id)
                    image = frame.require_image()
                    weak = augmented_tensor(image, frame.row['bbox'], rng)
                    clean = image_tensor(image, frame.row['bbox'], CONFIG['image_size']) if kd_weight else None
                    frame.verify_unchanged()
                    pending.append((frame.row['target_index'], weak, clean))
                    if len(pending) == settings['P'] * settings['K']:
                        step(pending)
                        pending = []
            if pending or drawn != len(sequence):
                raise ValueError('Incomplete PK training epoch')
            if device == 'cuda':
                torch.cuda.synchronize()
            training_seconds = time.perf_counter() - started
            model.eval()
            dev = dev_evaluate(model, epoch)
            metric = dev.get('mAP@10') if isinstance(dev, dict) else None
            if type(metric) not in (float, int) or not math.isfinite(metric) or not 0 <= metric <= 1:
                raise ValueError('DEV callback must return finite mAP@10 as fraction [0,1]')
            verify_inputs()
            if not synthetic:
                verify_pretrained(pretrained_path, pretrained_sha)
            if metric > best:
                best, best_epoch = float(metric), epoch
                torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, best_path)
            history['epochs'].append({'epoch': epoch, 'drawn_samples': drawn, 'pk_batches': len(batches),
                'train_seconds': training_seconds, 'loss': {key: value / drawn for key, value in totals.items()},
                'dev': dev, 'last_lr': [group['lr'] for group in optimizer.param_groups]})
            history['selected_epoch'] = best_epoch
            history['initial_model_state_sha256'] = initial_sha
            atomic_json(output / 'training-history.json', history)
        if source_identity() != sources:
            raise ValueError('Training/runtime source changed')
        state = torch.load(best_path, map_location='cpu', weights_only=True)
        from .compact_model import validate_state
        validate_state(model, state)
        checkpoint = output / 'compact.pt'
        torch.save({'kind': 'owned_compact_reid', 'format_version': 1, 'training_finished': True,
                    'test_only': synthetic, 'stand_id': stand, 'config': CONFIG, 'selected_epoch': best_epoch,
                    'selected_dev_mAP@10': best, 'provenance': provenance, 'model_state': state}, checkpoint)
        best_path.unlink()
        history.update(status='completed', training_finished=True, checkpoint_sha256=sha256(checkpoint))
        return checkpoint
    except BaseException as error:
        history.update(status='failed', training_finished=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(output / 'training-history.json', history)


def hashlib_array(values):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
