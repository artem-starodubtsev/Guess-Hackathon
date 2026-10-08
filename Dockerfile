FROM python:3.12-slim AS assets
WORKDIR /bundle
COPY scripts/download_assets.py scripts/download_assets.py
COPY runtime-assets.json ./
RUN python scripts/download_assets.py --destination /assets

FROM python:3.12-slim

ARG TORCH_FLAVOR=cu128
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FASHION_DEVICE=auto \
    HF_HOME=/tmp/huggingface TORCH_HOME=/tmp/torch
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/${TORCH_FLAVOR} \
    && sed '/^--/d; /^torch==/d' /tmp/requirements.txt > /tmp/runtime.txt \
    && python -m pip install --no-cache-dir -r /tmp/runtime.txt

COPY fashion_atlas/ fashion_atlas/
COPY web/ web/
COPY --from=assets /assets/ ./

RUN useradd --uid 10001 --create-home appuser
USER appuser
EXPOSE 8767
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8767/api/health', timeout=4)"
CMD ["python", "-m", "fashion_atlas", "--host", "0.0.0.0", "--port", "8767"]
