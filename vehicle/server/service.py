"""Bounded, serial inference jobs. Each query uses only itself and its frozen gallery."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
import zipfile

import numpy as np
from PIL import Image, UnidentifiedImageError

from .provider import sha256
from .runtime_settings import UnavailableProvider, device_options, read_device, release_cuda_cache
from .schemas import EvaluationReport, Limits, ModelInfo, RunEvaluation
from .validation import rank_vectors, write_outputs
from .retrieval import score_definition, validate_retrieval

LOG = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 40_000_000
ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
SCORE_DEFINITION = '(cosine + 1) / 2; similarity, not probability'


class ServiceError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def parse_csv(data, max_images=2000):
    try:
        text = data.decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != ['image_id', 'x', 'y', 'w', 'h']:
            raise ValueError('Ожидаются ровно колонки image_id,x,y,w,h в указанном порядке.')
        rows = []
        seen = set()
        for raw in reader:
            image_id = raw['image_id']
            if not image_id or not ID_PATTERN.fullmatch(image_id) or '..' in image_id or image_id in seen:
                raise ValueError('ID изображения некорректен или повторяется.')
            if None in raw or any(raw[k] is None for k in ('x', 'y', 'w', 'h')):
                raise ValueError('Число полей CSV не совпадает с заголовком.')
            bbox = []
            for key in ('x', 'y', 'w', 'h'):
                value = float(raw[key])
                if not math.isfinite(value) or value != int(value):
                    raise ValueError('Координаты рамки должны быть целыми пикселями.')
                bbox.append(int(value))
            if min(bbox[:2]) < 0 or min(bbox[2:]) <= 0:
                raise ValueError('Рамка требует x,y >= 0 и w,h > 0.')
            rows.append({'image_id': image_id, 'bbox': tuple(bbox)})
            seen.add(image_id)
            if len(rows) > max_images:
                raise ServiceError(413, 'IMAGE_LIMIT', 'Превышено допустимое число изображений.')
        if not rows:
            raise ValueError('CSV не содержит изображений.')
        return rows
    except (UnicodeError, csv.Error, ValueError, TypeError, OverflowError) as error:
        raise ServiceError(422, 'INVALID_CSV', str(error) or 'CSV должен быть в UTF-8.') from error


def normalize_rows(rows, maximum):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
    for row in rows:
        writer.writerow([row['image_id'], *(row.get('bbox') or [row[k] for k in ('x', 'y', 'w', 'h')])])
    return parse_csv(buffer.getvalue().encode(), maximum)


def checked_image(path, bbox):
    try:
        with Image.open(path) as image:
            if image.format not in ('JPEG', 'PNG'):
                raise ServiceError(422, 'INVALID_IMAGE', 'Разрешены только JPEG и PNG.')
            width, height = image.size
            if width * height > Image.MAX_IMAGE_PIXELS:
                raise ServiceError(422, 'INVALID_IMAGE', 'Снимок превышает лимит 40 миллионов пикселей.')
            if getattr(image, 'n_frames', 1) != 1:
                raise ServiceError(422, 'INVALID_IMAGE', 'Многостраничные и анимированные изображения не поддерживаются.')
            x, y, w, h = bbox
            if x + w > width or y + h > height:
                raise ServiceError(422, 'INVALID_IMAGE', 'Рамка выходит за границы исходного снимка.')
            image.verify()
        # verify() checks the container; load() also checks compressed pixels.
        with Image.open(path) as image:
            image.load()
        return width, height
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        LOG.warning('Uploaded image could not be decoded or verified', exc_info=True)
        raise ServiceError(422, 'INVALID_IMAGE', 'Не удалось прочитать изображение. Загрузите неповреждённый JPEG или PNG.') from error


def normalized(vector, dimension=None):
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not 1 <= values.size <= 16384 or not np.isfinite(values).all():
        raise ValueError('Model returned invalid embedding')
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 1e-12 or (dimension is not None and values.size != dimension):
        raise ValueError('Model returned invalid embedding dimension or norm')
    return values / norm


class Service:
    def __init__(self, mode, provider, data_root, limits=None, *, provider_factory=None, device_probe=None, reset_on_start=False):
        if mode != 'prod':
            raise ValueError('Only production mode is supported.')
        self.mode, self.provider = mode, provider
        self.prefix = '/api/v1'
        self.root = Path(data_root).resolve() / mode
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits or Limits()
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'vehicle-{mode}')
        self.evaluation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'vehicle-evaluation-{mode}')
        self.galleries, self.runs, self.assets = {}, {}, {}
        self.pending = 0
        self.storage_reserved = 0
        self.storage_limit = int(os.getenv('VEHICLE_MAX_STORAGE_MB', '8192')) * 1024**2
        self.closed = False
        self.uploads = 0
        self.provider_factory = provider_factory
        self.device_probe = device_probe or device_options
        self.settings_path = self.root / 'runtime-settings.json'
        self.selected_device = read_device(self.settings_path) if provider_factory else self._active_device() or 'cpu'
        self.switching = False
        self.settings_error = None
        self.devices = [
            {'id': 'cuda', 'name': 'GPU · NVIDIA CUDA', 'available': False, 'reason': 'Проверка доступности CUDA…'},
            {'id': 'cpu', 'name': 'CPU', 'available': True, 'reason': None},
        ]
        self._reset_pending = bool(reset_on_start)
        if not self._reset_pending:
            self._restore()

    def _active_device(self):
        if not self.provider.available:
            return None
        value = str((self.provider.model or {}).get('device', '')).split(':')[0]
        return value if value in ('cpu', 'cuda') else None

    def settings(self):
        with self.lock:
            return {'selected_device': self.selected_device, 'active_device': self._active_device(),
                    'switching': self.switching,
                    'can_switch': bool(self.provider_factory and not (self.closed or self.switching or self.pending or self.uploads)),
                    'devices': deepcopy(self.devices), 'error': self.settings_error}

    def start_runtime(self):
        """Start once from ASGI lifespan so importing the app never loads weights."""
        with self.lock:
            if self._reset_pending:
                self._reset_owned_state()
                self._reset_pending = False
            if self.provider_factory and not self.switching and not self.closed:
                self.switching = True
                self.executor.submit(self._switch_device, self.selected_device, True)

    def _reset_owned_state(self):
        """Clear only this web session's uploads/results, at process startup."""
        if self.closed or self.pending or self.uploads or self.switching or self.galleries or self.runs or self.assets:
            raise RuntimeError('Fresh-start cleanup is only allowed before the web session starts.')
        root = self.root
        if root.is_symlink() or getattr(root, 'is_junction', lambda: False)() or root.resolve() != root:
            raise ValueError('Refusing to clear linked application storage.')
        targets = []
        for path in root.iterdir():
            if not re.fullmatch(r'(?:[a-f0-9]{32}|upload-[a-z0-9_]{8})', path.name):
                continue
            if path.is_symlink() or getattr(path, 'is_junction', lambda: False)() or path.resolve().parent != root or not path.is_dir():
                raise ValueError('Refusing cleanup outside an owned upload/result directory.')
            targets.append(path)
        # Validate every target before removing any. runtime-settings.json and
        # external models, evaluation reports and source datasets are excluded.
        for path in targets:
            shutil.rmtree(path)
        LOG.info('Fresh web session: removed %s previous upload/result directories', len(targets))

    def select_device(self, device):
        with self.lock:
            if device not in ('cpu', 'cuda'):
                raise ServiceError(422, 'INVALID_DEVICE', 'Устройство должно быть cpu или cuda.')
            if not self.provider_factory:
                raise ServiceError(409, 'DEVICE_FIXED', 'Устройство задано оператором этого сервиса.')
            if self.closed or self.switching or self.pending or self.uploads:
                raise ServiceError(409, 'DEVICE_BUSY', 'Дождитесь завершения загрузки, индексирования и всех поисков.')
            option = next(value for value in self.devices if value['id'] == device)
            if not option['available']:
                raise ServiceError(409, 'DEVICE_UNAVAILABLE', option['reason'] or 'Устройство недоступно.')
            stale = any(item['public']['status'] != 'ready' or item.get('model_sha256') != self.model_identity()
                        for item in self.galleries.values())
            if device == self._active_device() and not stale:
                self.settings_error = None
                return self.settings()
            previous_device = self.selected_device
            self.selected_device = device
            self.switching = True
            self.settings_error = None
            try:
                self.executor.submit(self._switch_device, device, False, previous_device)
            except Exception:
                self.switching = False
                self.selected_device = self._active_device() or read_device(self.settings_path)
                raise
            return self.settings()

    def _switch_device(self, device, startup, previous_device=None):
        old_device = self._active_device()
        old_provider, candidate = self.provider, None
        published = False
        try:
            if startup:
                options = self.device_probe()
                with self.lock:
                    self.devices = options
                # The initial choice is durable even when no GPU is present.
                atomic_json(self.settings_path, {'device': device})
            option = next(value for value in self.devices if value['id'] == device)
            if not option['available']:
                raise ValueError(option['reason'] or 'Устройство недоступно.')
            candidate = self.provider if old_device == device else self.provider_factory(device=device)
            if not candidate.available:
                raise ValueError(candidate.reason or 'Не удалось загрузить модель на выбранном устройстве.')
            actual = str((candidate.model or {}).get('device', '')).split(':')[0]
            if actual != device:
                raise ValueError('Модель загрузилась на другом устройстве; автоматическая замена запрещена.')
            # Persist before publication; a storage error preserves the old model.
            atomic_json(self.settings_path, {'device': device})
            with self.lock:
                if self.closed:
                    return
                self.provider = candidate
                self.selected_device = device
                published = True
                galleries = []
                for item in self.galleries.values():
                    if item.get('model_sha256') == self.model_identity() and item['public']['status'] == 'ready':
                        continue
                    mmap = getattr(item.get('vectors'), '_mmap', None)
                    if mmap is not None:
                        mmap.close()
                    item['vectors'] = None
                    item['model_sha256'] = self.model_identity()
                    item['public'].update(status='indexing', processed=0, error=None)
                    self._event(item, 'gallery_reindex_queued', device=device)
                    galleries.append(item)
            if old_provider is not candidate:
                self._close_provider(old_provider)
                old_provider = None
            if old_device == 'cuda' and device == 'cpu':
                release_cuda_cache()
            # One worker, same immutable assets/order. Historical runs and their
            # exports/provenance are never rewritten by a runtime transition.
            for item in galleries:
                try:
                    with self.lock:
                        self._save(item, 'gallery')
                    self._index(item, release=False)
                except Exception:
                    LOG.exception('Could not persist reindexed gallery')
                    with self.lock:
                        item['public'].update(status='failed', error='Не удалось сохранить новый индекс галереи.')
                        try:
                            self._save(item, 'gallery')
                        except Exception:
                            LOG.exception('Could not persist gallery failure')
            failed = sum(item['public']['status'] == 'failed' for item in galleries)
            if failed:
                with self.lock:
                    self.settings_error = f'Устройство переключено. Не удалось переиндексировать галереи: {failed}.'
        except Exception as error:
            LOG.warning('Web device transition failed: %s', error, exc_info=True)
            with self.lock:
                self.settings_error = str(error) if isinstance(error, ValueError) else 'Не удалось переключить устройство. Подробности в журнале сервера.'
                if not published:
                    self.selected_device = old_device or previous_device or device
                if not self.provider.available:
                    unavailable = UnavailableProvider()
                    unavailable.reason = self.settings_error
                    self.provider = unavailable
        finally:
            # A shutdown racing the background loader can reject a fully built
            # candidate. Release its graphs; never close the still-active model.
            if not published and candidate is not None and candidate is not old_provider:
                self._close_provider(candidate)
            elif published and old_provider is not None and old_provider is not candidate:
                self._close_provider(old_provider)
            with self.lock:
                self.switching = False

    @staticmethod
    def _close_provider(provider):
        close = getattr(provider, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                LOG.exception('Could not fully release model runtime')

    def begin_upload(self):
        with self.lock:
            self.require_available()
            self.uploads += 1

    def end_upload(self):
        with self.lock:
            self.uploads -= 1

    def _restore(self):
        # A restarted process never claims that interrupted computation completed.
        for path in self.root.glob('*/state.json'):
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                if saved.get('mode') != self.mode:
                    continue
                for asset in saved['assets']:
                    asset_path = (path.parent / asset['filename']).resolve()
                    if asset_path.parent != path.parent.resolve() or not asset_path.is_file():
                        raise ValueError('Missing asset')
                    asset['path'] = asset_path
                    self.assets[asset['id']] = asset
                item = saved['item']
                internal = {'public': item, 'asset_ids': saved['asset_ids'], 'directory': path.parent, 'events': saved.get('events', []), 'vectors': None}
                if saved['kind'] == 'gallery':
                    internal['model_sha256'] = saved.get('model_sha256')
                    if item['status'] == 'ready' and (path.parent / 'vectors.npy').is_file():
                        internal['vectors'] = np.load(path.parent / 'vectors.npy', allow_pickle=False, mmap_mode='r')
                        if not self.provider_factory and internal['model_sha256'] != self.model_identity():
                            item.update(status='failed', error='Модель изменилась. Загрузите галерею заново.')
                    else:
                        item.update(status='failed', error='Индексирование было прервано перезапуском сервера.')
                    self.galleries[item['id']] = internal
                else:
                    internal['cancel'] = threading.Event()
                    internal['query_vectors'] = []
                    internal['calibration_sha256'] = saved.get('calibration_sha256')
                    internal['retrieval'] = saved.get('retrieval')
                    internal['retrieval_fingerprint'] = saved.get('retrieval_fingerprint')
                    internal['evaluation_id'] = saved.get('evaluation_id')
                    item.setdefault('evaluation', RunEvaluation().model_dump())
                    if item['evaluation']['status'] in ('waiting', 'running'):
                        item['evaluation'].update(status='failed', report=None, error='Оценка прервана перезапуском сервера. Загрузите разметку заново.')
                    if item['status'] in ('queued', 'running'):
                        item.update(status='failed', export_ready=False, timing=None, error='Вычисление прервано перезапуском сервера.')
                    if item['export_ready'] and not (path.parent / 'export.zip').is_file():
                        item.update(export_ready=False, artifacts_url=None, status='failed', error='Архив результата отсутствует.')
                    self.runs[item['id']] = internal
            except Exception:
                LOG.exception('Could not restore local state %s', path.name)

    def model_identity(self):
        model = self.provider.model or {}
        # Includes preprocessing, TTA, architecture and runtime weights. A changed
        # threshold alone does not invalidate gallery vectors.
        return getattr(self.provider, 'inference_fingerprint', None) or model.get('inference_fingerprint') or model.get('sha256')

    def _save(self, item, kind):
        assets = []
        for aid in item['asset_ids']:
            asset = {k: v for k, v in self.assets[aid].items() if k != 'path'}
            assets.append(asset)
        atomic_json(item['directory'] / 'state.json', {'mode': self.mode, 'kind': kind, 'item': item['public'],
                    'assets': assets, 'asset_ids': item['asset_ids'], 'events': item['events'],
                    'model_sha256': item.get('model_sha256'), 'calibration_sha256': item.get('calibration_sha256'),
                    'retrieval': item.get('retrieval'), 'retrieval_fingerprint': item.get('retrieval_fingerprint'),
                    'evaluation_id': item.get('evaluation_id')})

    def _event(self, item, event, **details):
        item['events'].append({'time': now(), 'event': event, **details})

    def require_available(self):
        if self.switching:
            raise ServiceError(409, 'DEVICE_BUSY', 'Дождитесь загрузки модели и переиндексирования галерей.')
        if not self.provider.available:
            raise ServiceError(503, 'MODEL_UNAVAILABLE', self.provider.reason or 'Модель недоступна.')

    def status(self):
        with self.lock:
            return {'contract_version': '1', 'mode': self.mode, 'available': self.provider.available and not self.switching,
                    'reason': 'Загрузка модели и переиндексирование галерей…' if self.switching else self.provider.reason,
                    'model': self.provider.model, 'threshold': self.provider.threshold, 'limits': self.limits.model_dump()}

    def report(self):
        with self.lock:
            model = deepcopy(self.provider.model)
            available = self.provider.available and not self.switching
            fingerprint = getattr(self.provider, 'inference_fingerprint', None) or (model or {}).get('inference_fingerprint')
            calibration = getattr(self.provider, 'calibration_sha256', None)
        unavailable = {'mode': self.mode, 'measured': False, 'model': model, 'dataset': None, 'hardware': None,
                'run_id': None, 'metrics': None, 'notes': [
                    'Качество рассчитывается отдельно на размеченной контрольной выборке официальным оценщиком.',
                    'Время и скорость пользовательского пакета доступны в разделе «Поиск»; они не заменяют оценку качества или конкурсный замер GPU.']}
        path = os.getenv('VEHICLE_EVALUATION_REPORT')
        directory = os.getenv('VEHICLE_EVALUATION_REPORTS_DIR')
        if not available:
            unavailable['notes'].append('Дождитесь готовности модели и завершения смены устройства.')
            return unavailable
        if not path and directory:
            if all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in (fingerprint, calibration)):
                path = Path(directory) / f'{fingerprint}-{calibration}.json'
            else:
                unavailable['notes'].append('Текущая модель не содержит полной подписи вычислений и порога для выбора отчёта.')
        if path:
            try:
                if Path(path).stat().st_size > 1024 * 1024:
                    raise ValueError('Report too large')
                report = EvaluationReport.model_validate_json(Path(path).read_text(encoding='utf-8-sig'))
                if report.mode != 'prod' or not report.measured or not report.model or report.model.sha256 != (model or {}).get('sha256'):
                    raise ValueError('Report belongs to another model')
                if fingerprint and (report.model.inference_fingerprint != fingerprint or report.calibration_sha256 != calibration):
                    raise ValueError('Report belongs to another inference configuration or calibration')
                if not report.model.sha256 or not all((report.dataset, report.hardware, report.run_id)):
                    raise ValueError('Report lacks provenance')
                metrics = report.metrics.model_dump()
                if not any(value is not None for value in metrics.values()):
                    raise ValueError('Report has no measurements')
                for key, value in metrics.items():
                    if value is not None and (not math.isfinite(value) or value < 0 or (key in ('map_at_10', 'rank_1', 'rank_5', 'micro_f1', 'tnr') and value > 1)):
                        raise ValueError('Metric out of range')
                return report.model_dump()
            except FileNotFoundError:
                unavailable['notes'].append('Для текущей модели, устройства и порога отчёт ещё не подключён. Выполните контрольную проверку и опубликуйте её отчёт.')
            except Exception:
                unavailable['notes'].append('Файл отчёта отклонён: проверьте происхождение, модель и диапазоны метрик.')
        elif not directory:
            unavailable['notes'].append('Каталог проверенных отчётов не подключён к серверу.')
        return unavailable

    def ensure_storage(self, additional_bytes=0):
        with self.lock:
            used = sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file())
            if used + self.storage_reserved + additional_bytes > self.storage_limit:
                raise ServiceError(413, 'STORAGE_FULL', 'Локальное хранилище заполнено. Оператору необходимо архивировать результаты.')
            if shutil.disk_usage(self.root).free < additional_bytes + 256 * 1024**2:
                raise ServiceError(413, 'DISK_FULL', 'Недостаточно свободного места для безопасной загрузки.')

    def _reserve(self):
        with self.lock:
            self.require_available()
            if self.closed or self.pending >= self.limits.max_pending_runs:
                raise ServiceError(429, 'QUEUE_FULL', 'Очередь заполнена. Дождитесь завершения текущих задач.')
            # Explicit local retention cap avoids unbounded memory/disk growth.
            if len(self.galleries) + len(self.runs) >= 100:
                raise ServiceError(429, 'STORAGE_LIMIT', 'Достигнут лимит 100 объектов. Оператору необходимо архивировать хранилище и перезапустить стенд.')
            self.pending += 1

    def _release(self):
        with self.lock:
            self.pending -= 1

    def _ingest(self, rows, image_paths, directory):
        if directory.resolve().parent != self.root:
            raise ValueError('Storage target is outside the mode root')
        rows = normalize_rows(rows, self.limits.max_images)
        if set(image_paths) != {r['image_id'] for r in rows}:
            raise ServiceError(422, 'IMAGE_CSV_MISMATCH', 'Файлы и image_id в CSV должны совпадать в точности.')
        reservation = sum(Path(path).stat().st_size for path in image_paths.values())
        with self.lock:
            self.ensure_storage(reservation)
            self.storage_reserved += reservation
        total = 0
        assets = []
        try:
            directory.mkdir()
            for row in rows:
                path = Path(image_paths[row['image_id']])
                size = path.stat().st_size
                total += size
                if size > self.limits.max_file_mb * 1024**2 or total > self.limits.max_upload_mb * 1024**2:
                    raise ServiceError(413, 'UPLOAD_LIMIT', 'Превышен лимит размера загрузки.')
                width, height = checked_image(path, row['bbox'])
                aid = uuid.uuid4().hex
                # Content was decoded before copying; a filename cannot become a server path.
                with Image.open(path) as image:
                    extension = '.jpg' if image.format == 'JPEG' else '.png'
                destination = directory / (aid + extension)
                shutil.copyfile(path, destination)
                assets.append({'id': aid, 'filename': destination.name, 'path': destination, 'image_id': row['image_id'],
                               'bbox': row['bbox'], 'width': width, 'height': height, 'sha256': sha256(destination)})
            with self.lock:
                self.assets.update({a['id']: a for a in assets})
            return [a['id'] for a in assets]
        except Exception:
            if directory.is_dir():
                shutil.rmtree(directory)
            raise
        finally:
            with self.lock:
                self.storage_reserved -= reservation

    def ref(self, asset_id):
        asset = self.assets[asset_id]
        return {'image_id': asset['image_id'], 'image_url': f'{self.prefix}/assets/{asset_id}',
                'crop_url': f'{self.prefix}/assets/{asset_id}?crop=true', 'bbox': asset['bbox'],
                'width': asset['width'], 'height': asset['height']}

    def import_gallery(self, name, rows, image_paths):
        self.require_available()
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ServiceError(422, 'INVALID_NAME', 'Название галереи должно содержать от 1 до 120 символов.')
        if len(rows) < 10:
            raise ServiceError(422, 'GALLERY_TOO_SMALL', 'Для топ-10 нужны минимум 10 изображений галереи.')
        self._reserve()
        gid = uuid.uuid4().hex
        directory = self.root / gid
        asset_ids = []
        try:
            asset_ids = self._ingest(rows, image_paths, directory)
            public = {'id': gid, 'name': name.strip(), 'count': len(rows), 'processed': 0, 'status': 'indexing', 'created_at': now(), 'error': None}
            item = {'public': public, 'asset_ids': asset_ids, 'directory': directory, 'vectors': None, 'events': [], 'model_sha256': self.model_identity()}
            with self.lock:
                self.galleries[gid] = item
                self._event(item, 'gallery_queued')
                self._save(item, 'gallery')
                self.executor.submit(self._index, item)
                return deepcopy(public)
        except Exception:
            with self.lock:
                self.galleries.pop(gid, None)
                self._discard_ingest(directory, asset_ids)
            self._release()
            raise

    def _discard_ingest(self, directory, asset_ids):
        if directory.resolve().parent != self.root or not re.fullmatch(r'[a-f0-9]{32}', directory.name):
            raise ValueError('Refusing cleanup outside owned object directory')
        for aid in asset_ids:
            self.assets.pop(aid, None)
        if directory.is_dir():
            shutil.rmtree(directory)

    def _embed_many(self, asset_ids):
        if getattr(self.provider, 'supports_path_extraction', False):
            assets = [self.assets[aid] for aid in asset_ids]
            values = self.provider.embed_paths([a['path'] for a in assets],
                [a['bbox'] for a in assets], image_ids=[a['image_id'] for a in assets])
            dimension = (self.provider.model or {}).get('dimension')
            return [normalized(value, dimension) for value in values]
        values = []
        for aid in asset_ids:
            asset = self.assets[aid]
            with Image.open(asset['path']) as image:
                image.load()
                dimension = (self.provider.model or {}).get('dimension')
                values.append(normalized(self.provider.embed(image, asset['bbox'], image_id=asset['image_id']), dimension))
        return values

    def _embed(self, asset_id):
        return self._embed_many([asset_id])[0]

    def _extraction_batches(self, item):
        size = int(getattr(self.provider, 'preferred_batch_size', 1))
        for begin in range(0, len(item['asset_ids']), size):
            if item['cancel'].is_set() or self.closed:
                return
            selected = item['asset_ids'][begin:begin + size]
            started = time.perf_counter()
            values, error = None, None
            try:
                if item['model_sha256'] != self.model_identity():
                    raise RuntimeError('Model changed during extraction')
                values = self._embed_many(selected)
            except Exception as caught:
                error = caught
            # Per-row duration includes its share of batch extraction plus its
            # own retrieval. Overall wall-clock timing remains authoritative.
            share_ms = (time.perf_counter() - started) * 1000 / len(selected)
            for index, aid in enumerate(selected):
                yield aid, None if error else values[index], error, share_ms

    def _index(self, item, *, release=True):
        try:
            vectors = []
            size = int(getattr(self.provider, 'preferred_batch_size', 1))
            for begin in range(0, len(item['asset_ids']), size):
                selected = item['asset_ids'][begin:begin + size]
                if self.closed:
                    raise RuntimeError('Shutdown')
                if item['model_sha256'] != self.model_identity():
                    raise RuntimeError('Model changed during gallery indexing')
                vectors.extend(self._embed_many(selected))
                with self.lock:
                    item['public']['processed'] += len(selected)
            matrix = np.stack(vectors).astype(np.float32)
            temporary = item['directory'] / 'vectors.tmp.npy'
            np.save(temporary, matrix, allow_pickle=False)
            temporary.replace(item['directory'] / 'vectors.npy')
            with self.lock:
                item['vectors'] = np.load(item['directory'] / 'vectors.npy', allow_pickle=False, mmap_mode='r')
                item['public']['status'] = 'ready'
                self._event(item, 'gallery_ready', count=len(vectors))
        except Exception:
            LOG.exception('Gallery indexing failed')
            with self.lock:
                item['public'].update(status='failed', error='Не удалось извлечь признаки галереи. Подробности в журнале сервера.')
                self._event(item, 'gallery_failed')
        finally:
            try:
                with self.lock:
                    self._save(item, 'gallery')
            finally:
                if release:
                    self._release()

    def list_galleries(self):
        with self.lock:
            items = sorted(self.galleries.values(), key=lambda item: (item['public']['created_at'], item['public']['id']), reverse=True)
            return deepcopy([item['public'] for item in items])

    def get_gallery(self, gid):
        with self.lock:
            if gid not in self.galleries:
                raise ServiceError(404, 'GALLERY_NOT_FOUND', 'Галерея не найдена в текущем режиме.')
            return deepcopy(self.galleries[gid]['public'])

    def create_run(self, gallery_id, rows, image_paths):
        self.require_available()
        self.get_gallery(gallery_id)
        with self.lock:
            if self.galleries[gallery_id]['public']['status'] != 'ready':
                raise ServiceError(409, 'GALLERY_NOT_READY', 'Дождитесь окончания индексирования галереи.')
            if self.galleries[gallery_id].get('model_sha256') != self.model_identity():
                raise ServiceError(409, 'GALLERY_MODEL_CHANGED', 'Модель или обработка изображения изменились. Пересчитайте галерею.')
            self._reserve()
        rid = uuid.uuid4().hex
        directory = self.root / rid
        asset_ids = []
        try:
            asset_ids = self._ingest(rows, image_paths, directory)
            public = {'id': rid, 'mode': self.mode, 'status': 'queued', 'gallery_id': gallery_id, 'model': deepcopy(self.provider.model),
                      'evaluation': RunEvaluation().model_dump(),
                      'timing': None, 'threshold': self.provider.threshold, 'total': len(rows), 'processed': 0, 'failed': 0, 'created_at': now(),
                      'results': [], 'error': None, 'export_ready': False, 'artifacts_url': None, 'evidence_url': f'{self.prefix}/runs/{rid}/evidence'}
            item = {'public': public, 'asset_ids': asset_ids, 'directory': directory, 'events': [], 'cancel': threading.Event(),
                    'query_vectors': [], 'model_sha256': self.model_identity(), 'calibration_sha256': getattr(self.provider, 'calibration_sha256', None),
                    'retrieval': validate_retrieval(getattr(self.provider, 'retrieval', None)),
                    'retrieval_fingerprint': getattr(self.provider, 'retrieval_fingerprint', None)}
            with self.lock:
                self.runs[rid] = item
                item['_queued_monotonic'] = time.perf_counter()
                self._event(item, 'run_queued')
                self._save(item, 'run')
                self.executor.submit(self._execute, item)
                return deepcopy(public)
        except Exception:
            with self.lock:
                self.runs.pop(rid, None)
                self._discard_ingest(directory, asset_ids)
            self._release()
            raise

    def _execute(self, item):
        public = item['public']
        gallery = self.galleries[public['gallery_id']]
        try:
            with self.lock:
                if item['cancel'].is_set():
                    return
                item['_started_monotonic'] = time.perf_counter()
                public['status'] = 'running'
                public['timing'] = self._timing(item)
                self._event(item, 'run_started')
            for aid, extracted, extraction_error, extraction_ms in self._extraction_batches(item):
                if item['cancel'].is_set() or self.closed:
                    break
                if (item['model_sha256'] != self.model_identity() or item['calibration_sha256'] != getattr(self.provider, 'calibration_sha256', None)
                        or item.get('retrieval') != getattr(self.provider, 'retrieval', None)
                        or item.get('retrieval_fingerprint') != getattr(self.provider, 'retrieval_fingerprint', None)
                        or public['threshold'] != self.provider.threshold):
                    raise RuntimeError('Model or calibration changed during the run')
                started = time.perf_counter()
                result = {'query': self.ref(aid), 'status': 'error', 'candidates': [], 'duration_ms': 0, 'error': None}
                try:
                    if extraction_error is not None:
                        raise extraction_error
                    vector = extracted
                    scores, order, accepted = rank_vectors(vector, gallery['vectors'], public['threshold'], item.get('retrieval'))
                    item['query_vectors'].append(vector)
                    result['status'] = 'matched' if accepted else 'rejected'
                    result['candidates'] = [{'rank': rank + 1, 'image': self.ref(gallery['asset_ids'][int(gi)]),
                                             'score': float(scores[gi]), 'accepted': int(gi) in accepted}
                                            for rank, gi in enumerate(order[:10])]
                except Exception:
                    LOG.exception('Query failed in run %s', public['id'])
                    result['error'] = 'Не удалось обработать снимок. Это ошибка обработки, а не отказ от совпадения.'
                result['duration_ms'] = round(extraction_ms + (time.perf_counter() - started) * 1000, 3)
                with self.lock:
                    if item['cancel'].is_set():
                        break
                    public['results'].append(result)
                    public['processed'] += 1
                    public['failed'] += int(result['status'] == 'error')
                    public['timing'] = self._timing(item)
                    self._event(item, 'query_finished', image_id=result['query']['image_id'], status=result['status'], duration_ms=result['duration_ms'])
            with self.lock:
                item.setdefault('_processing_finished', time.perf_counter())
                public['timing'] = self._timing(item)
                if item['cancel'].is_set() or self.closed:
                    public.update(status='cancelled', error='Запуск отменён. Полный конкурсный экспорт недоступен.')
                elif public['failed']:
                    public.update(status='failed', error='В пакете есть ошибки. Устраните их и повторите запуск для полного экспорта.')
                else:
                    self._event(item, 'inference_completed')
                    self._export(item, gallery)
                    public.update(status='completed', export_ready=True, artifacts_url=f"{self.prefix}/runs/{public['id']}/export")
                self._event(item, 'run_finished', status=public['status'])
                self._save(item, 'run')
                self._settle_evaluation(item)
        except Exception:
            LOG.exception('Run failed %s', public['id'])
            with self.lock:
                item.setdefault('_processing_finished', time.perf_counter())
                public['timing'] = self._timing(item)
                public.update(status='failed', export_ready=False, artifacts_url=None, error='Не удалось завершить запуск. Подробности в журнале сервера.')
                self._event(item, 'run_failed')
                self._settle_evaluation(item)
                self._save(item, 'run')
        finally:
            # Query vectors have already been exported; retaining them would grow RAM with history.
            item['query_vectors'] = []
            self._release()

    def _timing(self, item):
        """Real monotonic search time; excludes upload, queue and export packaging."""
        start = item.get('_started_monotonic')
        if start is None:
            return item['public'].get('timing')
        end = item.get('_processing_finished', time.perf_counter())
        elapsed = max(0., end - start)
        successful = [r for r in item['public']['results'] if r['status'] != 'error']
        return {'queue_ms': round(max(0., start - item['_queued_monotonic']) * 1000, 3),
                'processing_ms': round(elapsed * 1000, 3), 'successful_images': len(successful),
                'images_per_second': round(len(successful) / elapsed, 3) if successful and elapsed else None,
                'mean_image_ms': round(sum(r['duration_ms'] for r in successful) / len(successful), 3) if successful else None}

    def _run_snapshot(self, item):
        return deepcopy({**item['public'], 'timing': self._timing(item)})

    def list_runs(self):
        with self.lock:
            items = sorted(self.runs.values(), key=lambda item: (item['public']['created_at'], item['public']['id']), reverse=True)
            return [self._run_snapshot(item) for item in items[:30]]

    def _run(self, rid):
        if rid not in self.runs:
            raise ServiceError(404, 'RUN_NOT_FOUND', 'Запуск не найден в текущем режиме.')
        return self.runs[rid]

    def get_run(self, rid):
        with self.lock:
            return self._run_snapshot(self._run(rid))

    def cancel(self, rid):
        with self.lock:
            item = self._run(rid)
            if item['public']['status'] in ('queued', 'running'):
                item['cancel'].set()
                item.setdefault('_processing_finished', time.perf_counter())
                item['public']['timing'] = self._timing(item)
                item['public'].update(status='cancelled', error='Запуск отменён. Полный конкурсный экспорт недоступен.')
                self._event(item, 'cancel_requested')
                self._settle_evaluation(item)
                self._save(item, 'run')
            return deepcopy(item['public'])

    def get_evaluation(self, rid):
        with self.lock:
            return deepcopy(self._run(rid)['public']['evaluation'])

    def submit_evaluation(self, rid, contents):
        """Store labels for scoring only. No identity/camera data reaches the provider."""
        from .evaluation import validate_ground_truth
        if not contents or len(contents) > 1024 * 1024:
            raise ServiceError(413, 'GROUND_TRUTH_LIMIT', 'Разметка должна занимать от 1 байта до 1 МиБ.')
        with self.lock:
            item = self._run(rid)
            if self.closed:
                raise ServiceError(503, 'SERVICE_CLOSED', 'Сервер завершает работу.')
            if item['public']['evaluation']['status'] in ('waiting', 'running'):
                raise ServiceError(409, 'EVALUATION_BUSY', 'Оценка уже ожидает завершения поиска или выполняется.')
            if item['public']['status'] not in ('queued', 'running', 'completed'):
                raise ServiceError(409, 'RUN_NOT_COMPLETED', 'Оценить можно только полностью успешный поиск. Создайте новый запуск.')
            gallery = self.galleries[item['public']['gallery_id']]
            qids = [self.assets[aid]['image_id'] for aid in item['asset_ids']]
            gids = [self.assets[aid]['image_id'] for aid in gallery['asset_ids']]
            digest = hashlib.sha256(contents).hexdigest()
            state = RunEvaluation(status='waiting', ground_truth_sha256=digest).model_dump()
            # An attempted replacement cannot continue showing measurements from old labels.
            item['public']['evaluation'] = state
            item['evaluation_id'] = None
            try:
                validate_ground_truth(contents, qids, gids)
            except (ValueError, UnicodeError, csv.Error, TypeError, AttributeError) as error:
                state.update(status='failed', report=None, error='Разметка должна содержать ровно все ID запросов и галереи этого запуска: image_id,vehicle_id,camera_id,split.')
                self._save(item, 'run')
                raise ServiceError(422, 'INVALID_GROUND_TRUTH', state['error']) from error
            try:
                self.ensure_storage(len(contents) + 1024 * 1024)
                item['evaluation_id'] = uuid.uuid4().hex
                directory = self._evaluation_directory(item)
                directory.mkdir(parents=True, exist_ok=False)
                (directory / 'ground_truth.csv').write_bytes(contents)
                for name, aids in (('query', item['asset_ids']), ('gallery', gallery['asset_ids'])):
                    with (directory / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
                        writer = csv.writer(stream)
                        writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
                        for aid in aids:
                            asset = self.assets[aid]
                            writer.writerow([asset['image_id'], *asset['bbox']])
                self._save(item, 'run')
                self._settle_evaluation(item)
            except (OSError, ServiceError, RuntimeError):
                state.update(status='failed', report=None, error='Не удалось сохранить разметку или поставить оценку в очередь. Повторите загрузку.')
                try:
                    self._save(item, 'run')
                except OSError:
                    LOG.exception('Could not save failed evaluation state')
                raise
            return deepcopy(state)

    def _evaluation_directory(self, item):
        identifier = item.get('evaluation_id')
        if not isinstance(identifier, str) or not re.fullmatch('[a-f0-9]{32}', identifier):
            raise ValueError('No valid evaluation revision')
        return item['directory'] / 'evaluation' / identifier

    def _settle_evaluation(self, item):
        """Called with self.lock held after an inference status change or label upload."""
        state = item['public']['evaluation']
        if state['status'] != 'waiting':
            return
        if item['public']['status'] in ('failed', 'cancelled'):
            state.update(status='failed', report=None, error='Поиск не завершён успешно. Качество не рассчитано.')
            self._save(item, 'run')
        elif item['public']['status'] == 'completed':
            state['status'] = 'running'
            self._save(item, 'run')
            self.evaluation_executor.submit(self._evaluate_run, item, item['evaluation_id'])

    def _evaluate_run(self, item, revision):
        from .evaluation import evaluate_export
        reserved = 0
        try:
            directory = self._evaluation_directory(item)
            # Preserve the immutable inference ZIP and its exact files. Evaluation outputs
            # live in their own revision directory; a retry cannot expose an old report.
            names = ('submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json', 'provenance.json')
            required = sum((item['directory'] / name).stat().st_size for name in names) + 1024 * 1024
            with self.lock:
                self.ensure_storage(required)
                self.storage_reserved += required
                reserved = required
            for name in names:
                shutil.copyfile(item['directory'] / name, directory / name)
            evaluate_export(directory, directory / 'query.csv', directory / 'gallery.csv', directory / 'ground_truth.csv', current_run=True)
            report = EvaluationReport.model_validate_json((directory / 'evaluation-report.json').read_text(encoding='utf-8'))
            if report.run_id != item['public']['id'] or not report.model or report.model != ModelInfo.model_validate(item['public']['model']) or report.ground_truth_sha256 != item['public']['evaluation']['ground_truth_sha256']:
                raise ValueError('Evaluation provenance does not match this run')
            with self.lock:
                if item['evaluation_id'] == revision:
                    item['public']['evaluation'].update(status='completed', report=report.model_dump(), error=None)
                    self._save(item, 'run')
        except Exception:
            LOG.exception('Evaluation failed for run %s', item['public']['id'])
            with self.lock:
                if item['evaluation_id'] == revision:
                    item['public']['evaluation'].update(status='failed', report=None, error='Не удалось рассчитать качество официальным оценщиком. Проверьте разметку и журнал сервера.')
                    self._save(item, 'run')
        finally:
            with self.lock:
                self.storage_reserved -= reserved

    def evaluation_report_path(self, rid):
        with self.lock:
            item = self._run(rid)
            if item['public']['evaluation']['status'] != 'completed':
                raise ServiceError(409, 'EVALUATION_NOT_READY', 'Отчёт доступен после успешного расчёта качества этого запуска.')
            path = self._evaluation_directory(item) / 'metrics.json'
            if not path.is_file():
                raise ServiceError(409, 'EVALUATION_NOT_READY', 'Файл отчёта отсутствует. Загрузите разметку заново.')
            return path

    def evidence(self, rid):
        with self.lock:
            item = self._run(rid)
            public = item['public']
            gallery = self.galleries[public['gallery_id']]
            def inputs(aids):
                return [{k: self.assets[a][k] for k in ('image_id', 'sha256', 'bbox', 'width', 'height')} for a in aids]
            gallery_inputs = inputs(gallery['asset_ids'])
            gallery_hash = hashlib.sha256(json.dumps(gallery_inputs, sort_keys=True).encode()).hexdigest()
            files = {p.name: sha256(p) for p in item['directory'].glob('*') if p.name in ('submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json')}
            return {'contract_version': '1', 'mode': self.mode, 'run_id': rid, 'timing': self._timing(item), 'status': public['status'], 'model': public['model'],
                    'calibration_sha256': item.get('calibration_sha256'), 'threshold': public['threshold'], 'score_definition': score_definition(item.get('retrieval')),
                    'retrieval': item.get('retrieval'), 'retrieval_fingerprint': item.get('retrieval_fingerprint'),
                    'gallery_id': public['gallery_id'], 'gallery_sha256': gallery_hash,
                    'input_order': {'query': [self.assets[a]['image_id'] for a in item['asset_ids']], 'gallery': [self.assets[a]['image_id'] for a in gallery['asset_ids']]},
                    'inputs': {'query': inputs(item['asset_ids']), 'gallery': gallery_inputs}, 'files_sha256': files, 'events': deepcopy(item['events']),
                    'notes': ['Этот файл фиксирует поиск. Качество при наличии разметки сохраняется отдельно в отчёте оценки этого запуска.',
                              'Запросы независимы. Cosine-равенства: порядок gallery CSV; равенства реранкера: исходный cosine-порядок.']}

    def _export(self, item, gallery):
        directory = item['directory']
        query_ids = [self.assets[a]['image_id'] for a in item['asset_ids']]
        gallery_ids = [self.assets[a]['image_id'] for a in gallery['asset_ids']]
        write_outputs(directory, query_ids, gallery_ids, np.stack(item['query_vectors']), gallery['vectors'], item['public']['threshold'], retrieval=item.get('retrieval'))
        # Metadata describes the completed package, while the public run becomes completed only after ZIP succeeds.
        proof = self.evidence(item['public']['id'])
        proof['status'] = 'completed'
        proof['events'].append({'time': now(), 'event': 'export_created'})
        atomic_json(directory / 'provenance.json', proof)
        (directory / 'events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in proof['events']), encoding='utf-8')
        temp = directory / 'export.tmp'
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ('submission.csv', 'candidates.csv', 'embeddings.npy', 'embedding_order.json', 'retrieval.json', 'provenance.json', 'events.jsonl'):
                archive.write(directory / name, name)
        temp.replace(directory / 'export.zip')

    def export_path(self, rid):
        with self.lock:
            item = self._run(rid)
            if item['public']['status'] != 'completed' or not item['public']['export_ready']:
                raise ServiceError(409, 'EXPORT_NOT_READY', 'Экспорт доступен только для полностью успешного запуска.')
            return item['directory'] / 'export.zip'

    def asset_path(self, asset_id, crop=False):
        with self.lock:
            if asset_id not in self.assets:
                raise ServiceError(404, 'ASSET_NOT_FOUND', 'Изображение не найдено в текущем режиме.')
            asset = self.assets[asset_id]
        if not crop:
            return asset['path']
        destination = asset['path'].with_suffix('.crop.jpg')
        # Concurrent requests may share a crop; serialize atomic cache publication.
        with self.lock:
            if not destination.is_file():
                with Image.open(asset['path']) as image:
                    x, y, w, h = asset['bbox']
                    result = image.crop((x, y, x + w, y + h)).convert('RGB')
                    temp = destination.with_suffix('.tmp')
                    result.save(temp, format='JPEG', quality=94)
                    temp.replace(destination)
        return destination

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for rid in self.runs:
                self.cancel(rid)
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.evaluation_executor.shutdown(wait=True, cancel_futures=False)
        self._close_provider(self.provider)
        for item in self.galleries.values():
            mmap = getattr(item.get('vectors'), '_mmap', None)
            if mmap is not None:
                mmap.close()
