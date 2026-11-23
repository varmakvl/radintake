# radintake service image: CPU by default; OCR via system Tesseract.
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RADINTAKE_OFFLINE=1 RADINTAKE_DATA_DIR=/data RADINTAKE_TRACE_DIR=/data/traces
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 fonts-dejavu-core curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY schemas ./schemas
COPY prompts ./prompts
RUN pip install --no-deps -e . \
 && useradd --create-home --uid 10001 radintake && mkdir -p /data/traces && chown -R radintake /data
# Editable install keeps REPO_ROOT=/app, where configs/ and schemas/ live.
USER radintake
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "--factory", "radintake.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
