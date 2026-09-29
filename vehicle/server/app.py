"""Run from repository root: python -m uvicorn vehicle.server.app:app --host 127.0.0.1 --port 8017.

VEHICLE_WEB_ROOT selects a built static client; VEHICLE_DATA_ROOT selects local storage.
Only the production API is available; missing model artifacts disable inference.
"""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
import uuid

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from .provider import ProductionProvider
from .runtime_settings import UnavailableProvider
from .schemas import ApiError, DeviceChoice, Evidence, EvaluationReport, Gallery, Run, RunEvaluation, RuntimeSettings, ServiceStatus
from .service import ID_PATTERN, Service, ServiceError, parse_csv

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


class RequestLimitsMiddleware:
    def __init__(self, app, max_bytes=2049 * 1024**2):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault('state', {})['request_id'] = request_id
        headers = dict(scope.get('headers', []))
        max_bytes = self.max_bytes
        if scope.get('method') == 'POST' and scope.get('path', '').startswith('/api/v1/runs/') and scope.get('path', '').endswith('/evaluation'):
            max_bytes = 1024 * 1024 + 65536  # One 1 MiB label file plus bounded multipart overhead.
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            length = max_bytes + 1
        if length < 0 or length > max_bytes:
            response = JSONResponse({'error': {'code': 'UPLOAD_LIMIT', 'message': 'Превышен лимит загрузки.', 'request_id': request_id}}, status_code=413)
            return await response(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > max_bytes:
                    raise ServiceError(413, 'UPLOAD_LIMIT', 'Превышен лимит загрузки.')
            return message

        async def secure_send(message):
            if message['type'] == 'http.response.start':
                additions = [(b'x-request-id', request_id.encode()), (b'x-content-type-options', b'nosniff'),
                             (b'referrer-policy', b'no-referrer'), (b'cache-control', b'no-store')]
                if scope.get('path', '').startswith('/api/'):
                    additions.append((b'content-security-policy', b"default-src 'none'; frame-ancestors 'none'"))
                message['headers'] = [*message.get('headers', []), *additions]
            await send(message)
        await self.app(scope, bounded_receive, secure_send)


ERROR_RESPONSES = {code: {'model': ApiError} for code in (404, 409, 413, 422, 429, 503)}


def upload_schema(field):
    return {
    'requestBody': {
        'required': True,
        'content': {
            'multipart/form-data': {
                'schema': {
                    'type': 'object', 'required': [field, 'csv', 'images'], 'additionalProperties': False,
                    'properties': {
                        field: {'type': 'string'},
                        'csv': {'type': 'string', 'format': 'binary'},
                        'images': {'type': 'array', 'items': {'type': 'string', 'format': 'binary'}},
                    },
                },
            },
        },
    },
}


async def ingest_request(request, service, gallery):
    service.require_available()
    gate = request.app.state.upload_gate
    try:
        await asyncio.wait_for(gate.acquire(), timeout=0.1)
    except TimeoutError as error:
        raise ServiceError(429, 'UPLOAD_BUSY', 'Другая загрузка проверяется. Повторите немного позже.') from error
    upload_started = False
    try:
        # Reserve runtime ownership before multipart parsing. A later settings
        # request cannot swap the provider while this upload is being published.
        service.begin_upload()
        upload_started = True
        # Multipart parser spools files before the endpoint can inspect each one.
        # Reserve enough free space before parsing; one upload parses at a time.
        declared = int(request.headers.get('content-length', '0'))
        service.ensure_storage(declared or service.limits.max_upload_mb * 1024**2)
        async with request.form(max_files=service.limits.max_images + 1, max_fields=4, max_part_size=1024 * 1024) as form:
            expected_field = 'name' if gallery else 'gallery_id'
            if set(form.keys()) != {expected_field, 'csv', 'images'} or len(form.getlist(expected_field)) != 1 or len(form.getlist('csv')) != 1:
                raise ServiceError(422, 'INVALID_FORM', f'Ожидаются {expected_field}, один CSV и изображения.')
            field = form.get(expected_field)
            csv_file = form.get('csv')
            files = form.getlist('images')
            if not isinstance(field, str) or not isinstance(csv_file, UploadFile) or not files or not all(isinstance(f, UploadFile) for f in files):
                raise ServiceError(422, 'INVALID_FORM', 'CSV и изображения нужно передать как файлы.')
            if len(files) > service.limits.max_images:
                raise ServiceError(413, 'IMAGE_LIMIT', 'Превышен лимит числа изображений.')
            data = await csv_file.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ServiceError(413, 'CSV_LIMIT', 'CSV превышает 1 МиБ.')
            rows = parse_csv(data, service.limits.max_images)
            if not gallery:
                status = service.get_gallery(field)
                if status['status'] != 'ready':
                    raise ServiceError(409, 'GALLERY_NOT_READY', 'Галерея ещё не готова.')
            with tempfile.TemporaryDirectory(prefix='upload-', dir=service.root) as temporary:
                paths = {}
                total = len(data)
                quota_checkpoint = total
                for upload in files:
                    filename = upload.filename or ''
                    # Reject both Windows and POSIX path syntax, never use uploaded names for disk paths.
                    if '/' in filename or '\\' in filename or ':' in filename:
                        raise ServiceError(422, 'INVALID_FILENAME', 'Имя файла не должно содержать путь.')
                    image_id = Path(filename).stem
                    if not ID_PATTERN.fullmatch(image_id) or '..' in image_id or image_id in paths or Path(filename).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                        raise ServiceError(422, 'INVALID_FILENAME', 'Допустимы уникальные JPEG/PNG с именем image_id.')
                    path = Path(temporary) / uuid.uuid4().hex
                    size = 0
                    with path.open('wb') as target:
                        while chunk := await upload.read(1024 * 1024):
                            size += len(chunk)
                            total += len(chunk)
                            if size > service.limits.max_file_mb * 1024**2 or total > service.limits.max_upload_mb * 1024**2:
                                raise ServiceError(413, 'UPLOAD_LIMIT', 'Превышен лимит размера загрузки.')
                            if total - quota_checkpoint >= 16 * 1024**2:
                                service.ensure_storage(16 * 1024**2)
                                quota_checkpoint = total
                            target.write(chunk)
                    paths[image_id] = path
                operation = service.import_gallery if gallery else service.create_run
                return await asyncio.to_thread(operation, field, rows, paths)
    finally:
        if upload_started:
            service.end_upload()
        gate.release()


def register_routes(app, service):
    router = APIRouter(prefix=service.prefix, tags=[service.mode], responses=ERROR_RESPONSES)

    @router.get('/status', response_model=ServiceStatus)
    def status():
        return service.status()

    @router.get('/settings', response_model=RuntimeSettings)
    def settings():
        return service.settings()

    @router.put('/settings/device', response_model=RuntimeSettings, status_code=202)
    def select_device(choice: DeviceChoice):
        return service.select_device(choice.device)

    @router.get('/galleries', response_model=list[Gallery])
    def galleries():
        return service.list_galleries()

    @router.post('/galleries', response_model=Gallery, status_code=202, openapi_extra=upload_schema('name'))
    async def create_gallery(request: Request):
        return await ingest_request(request, service, gallery=True)

    @router.get('/galleries/{gallery_id}', response_model=Gallery)
    def gallery(gallery_id: str):
        return service.get_gallery(gallery_id)

    @router.get('/runs', response_model=list[Run])
    def runs():
        return service.list_runs()

    @router.post('/runs', response_model=Run, status_code=202, openapi_extra=upload_schema('gallery_id'))
    async def create_run(request: Request):
        return await ingest_request(request, service, gallery=False)

    @router.get('/runs/{run_id}', response_model=Run)
    def run(run_id: str):
        return service.get_run(run_id)

    @router.post('/runs/{run_id}/cancel', response_model=Run)
    def cancel(run_id: str):
        return service.cancel(run_id)

    @router.get('/runs/{run_id}/evidence', response_model=Evidence)
    def evidence(run_id: str):
        return service.evidence(run_id)

    @router.get('/runs/{run_id}/evaluation', response_model=RunEvaluation)
    def evaluation(run_id: str):
        return service.get_evaluation(run_id)

    @router.post('/runs/{run_id}/evaluation', response_model=RunEvaluation, status_code=202,
                 openapi_extra={'requestBody': {'required': True, 'content': {'multipart/form-data': {'schema': {
                     'type': 'object', 'required': ['ground_truth'], 'additionalProperties': False,
                     'properties': {'ground_truth': {'type': 'string', 'format': 'binary'}}}}}}})
    async def evaluate(run_id: str, request: Request):
        service.get_run(run_id)
        async with request.form(max_files=1, max_fields=0, max_part_size=1024 * 1024) as form:
            if set(form.keys()) != {'ground_truth'} or len(form.getlist('ground_truth')) != 1 or not isinstance(form.get('ground_truth'), UploadFile):
                raise ServiceError(422, 'INVALID_FORM', 'Передайте один файл ground_truth.csv в поле ground_truth.')
            contents = await form['ground_truth'].read(1024 * 1024 + 1)
            if len(contents) > 1024 * 1024:
                raise ServiceError(413, 'GROUND_TRUTH_LIMIT', 'Разметка превышает 1 МиБ.')
            return await asyncio.to_thread(service.submit_evaluation, run_id, contents)

    @router.get('/runs/{run_id}/evaluation/report', response_class=FileResponse,
                responses={200: {'content': {'application/json': {'schema': {'type': 'string', 'format': 'binary'}}}}})
    def evaluation_report(run_id: str):
        return FileResponse(service.evaluation_report_path(run_id), media_type='application/json', filename=f'evaluation-{run_id}.json')

    @router.get('/runs/{run_id}/export', response_class=FileResponse, responses={200: {'content': {'application/zip': {'schema': {'type': 'string', 'format': 'binary'}}}}})
    def export(run_id: str):
        return FileResponse(service.export_path(run_id), media_type='application/zip', filename=f'{service.mode.upper()}-{run_id}.zip')

    @router.get('/assets/{asset_id}', response_class=FileResponse, responses={200: {'content': {'image/jpeg': {'schema': {'type': 'string', 'format': 'binary'}}, 'image/png': {'schema': {'type': 'string', 'format': 'binary'}}}}})
    def asset(asset_id: str, crop: bool = False):
        return FileResponse(service.asset_path(asset_id, crop))

    @router.get('/report', response_model=EvaluationReport)
    def report():
        return service.report()

    app.include_router(router)


def create_app(provider=None, data_root=None, web_root=None, *, provider_factory=None, device_probe=None):
    services = []

    @asynccontextmanager
    async def lifespan(app):
        for service in services:
            service.start_runtime()
        yield
        for service in services:
            await asyncio.to_thread(service.close)

    app = FastAPI(title='Без номера · локальный стенд', version='1.0.0', docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(RequestLimitsMiddleware)
    app.state.upload_gate = asyncio.Semaphore(1)
    app.state.services = services
    root = Path(data_root or os.getenv('VEHICLE_DATA_ROOT') or REPO_ROOT / 'vehicle' / 'var')
    reset_setting = os.getenv('VEHICLE_RESET_ON_START', '0')
    if reset_setting not in ('0', '1'):
        raise ValueError('VEHICLE_RESET_ON_START must be 0 or 1.')

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse({'error': {'code': error.code, 'message': error.message, 'request_id': request.state.request_id}}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({'error': {'code': 'INVALID_REQUEST', 'message': 'Проверьте типы и обязательные поля запроса.', 'request_id': request.state.request_id}}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({'error': {'code': f'HTTP_{error.status_code}', 'message': str(error.detail), 'request_id': request.state.request_id}}, status_code=error.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        LOG.exception('Unhandled API error request_id=%s', request.state.request_id, exc_info=error)
        return JSONResponse({'error': {'code': 'INTERNAL_ERROR', 'message': 'Ошибка сервера. Сохраните идентификатор запроса для диагностики.', 'request_id': request.state.request_id}}, status_code=500)

    def create_service(mode, provider, data_root=None):
        if any(s.mode == mode for s in services):
            raise ValueError('Service already registered')
        service = Service(mode, provider, data_root or root,
                          provider_factory=runtime_factory, device_probe=device_probe,
                          reset_on_start=reset_setting == '1')
        services.append(service)
        register_routes(app, service)
        return service

    # Explicit provider injection retains the fixed-provider Service contract.
    runtime_factory = (provider_factory or ProductionProvider) if provider is None else provider_factory
    create_service('prod', provider if provider is not None else UnavailableProvider())

    static_root = Path(web_root or os.getenv('VEHICLE_WEB_ROOT') or REPO_ROOT / 'vehicle' / 'web' / 'dist').resolve()

    @app.get('/{path:path}', include_in_schema=False)
    async def frontend(path: str):
        if path in ('api', 'docs', 'redoc') or path.startswith(('api/', 'openapi')):
            raise ServiceError(404, 'NOT_FOUND', 'Метод API не найден.')
        candidate = (static_root / path).resolve()
        if candidate != static_root and static_root not in candidate.parents:
            raise ServiceError(404, 'NOT_FOUND', 'Файл не найден.')
        if candidate.is_file():
            return FileResponse(candidate)
        index = static_root / 'index.html'
        if index.is_file() and not Path(path).suffix:
            return FileResponse(index)
        raise ServiceError(404, 'NOT_FOUND', 'Клиент ещё не собран или файл отсутствует.')

    return app


app = create_app()
