# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69
FROM ${BASE_IMAGE} AS source-check
WORKDIR /checked
COPY . ./
# The manifest does not hash itself. Weights are imported before this build.
ARG SOURCE_MANIFEST_SHA256=computed-in-image
RUN python verify_context.py > /source-verification.json && python -c "import json,os; actual=json.load(open('/source-verification.json'))['source_manifest_sha256']; expected=os.getenv('SOURCE_MANIFEST_SHA256','computed-in-image'); assert expected in ('computed-in-image',actual), 'Build argument differs from the verified source SHA256'"

FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS web-build
WORKDIR /build/vehicle/web
COPY --from=source-check /checked/vehicle/web/package.json /checked/vehicle/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY --from=source-check /checked/vehicle/web/ ./
RUN npm run typecheck && npm run build

FROM ${BASE_IMAGE} AS runtime
USER 0:0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VEHICLE_DATA_ROOT=/app/var VEHICLE_WEB_ROOT=/app/web VEHICLE_MODEL_BUNDLE=/app/models/e27 VEHICLE_DEVICE=cuda VEHICLE_CPU_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/tmp/huggingface TORCH_HOME=/tmp/torch USER=vehicle LOGNAME=vehicle HOME=/tmp CUBLAS_WORKSPACE_CONFIG=:4096:8
COPY --from=source-check /checked/requirements-model.lock /checked/requirements-web.lock /app/
RUN python -m pip install --no-cache-dir -r /app/requirements-model.lock -r /app/requirements-web.lock && python -m pip check && python -c "import torch,torchvision,timm,numpy,PIL; assert torch.__version__.split('+')[0]=='2.8.0'; assert torchvision.__version__.split('+')[0]=='0.23.0'; assert timm.__version__=='1.0.20'; assert numpy.__version__=='2.2.6'; assert PIL.__version__=='11.3.0'"
RUN groupadd --gid 10001 vehicle && useradd --uid 10001 --gid 10001 --no-create-home vehicle && mkdir -p /app/var /results && chown 10001:10001 /app/var /results
COPY --from=source-check /checked/vehicle/__init__.py /app/vehicle/__init__.py
COPY --from=source-check /checked/vehicle/server/ /app/vehicle/server/
COPY --from=source-check /checked/docs/sources/falcon-evaluation-2026-09-21/evaluate.py /app/docs/sources/falcon-evaluation-2026-09-21/evaluate.py
COPY --from=source-check /checked/scripts/ /app/scripts/
COPY --from=source-check /checked/control/ /app/control/
COPY --from=source-check /checked/models/e27/ /app/models/e27/
COPY --from=source-check /checked/SOURCE_SHA256.json /app/web-source-manifest.json
COPY --from=source-check /source-verification.json /app/source-verification.json
COPY --from=web-build /build/vehicle/web/dist/ /app/web/
RUN python -m compileall -q /app/vehicle/server /app/scripts && python -m vehicle.server.model_bundle check --bundle /app/models/e27 && python -m pip freeze > /app/build-runtime.txt
ARG SOURCE_MANIFEST_SHA256=computed-in-image
# Direct Compose builds retain the actual SHA in source-verification.json.
LABEL org.opencontainers.image.version=E27 org.bez-nomera.runtime-target=runtime org.bez-nomera.source-manifest-sha256=${SOURCE_MANIFEST_SHA256}
ENV VEHICLE_SOURCE_REVISION=E27 VEHICLE_SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256}
USER 10001:10001
EXPOSE 8017
ENTRYPOINT ["python", "-m", "vehicle.server.runtime_entrypoint"]
CMD ["python", "-m", "uvicorn", "vehicle.server.app:app", "--host", "0.0.0.0", "--port", "8017", "--workers", "1"]
# Control API health is separate from model readiness at /api/v1/status.
HEALTHCHECK --interval=15s --timeout=10s --start-period=180s --retries=3 CMD python -c "import json,urllib.request; s=json.load(urllib.request.urlopen('http://127.0.0.1:8017/api/v1/settings',timeout=5)); assert s.get('selected_device') in ('cpu','cuda')" || exit 1
