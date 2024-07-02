FROM python:3.11-buster as builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app/
ADD redbox-core/ /app/redbox-core
ADD README.md /app
ADD pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --no-ansi --with worker,api --without ai,dev

FROM python:3.11-slim-buster as runtime
RUN apt update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Downlaod Sentence Tokenizer
RUN python -m nltk.downloader punkt

# Download POS tagger
RUN python -m nltk.downloader averaged_perceptron_tagger

# Download the OCR models
ADD worker/download_ocr_models.py /app/
RUN python /app/download_ocr_models.py


WORKDIR /app/
ADD redbox-core/ /app/redbox-core
ADD README.md /app/README.md
ADD worker/ /app/worker

ADD worker/health.sh /app/
RUN chmod +x /app/health.sh

CMD ["faststream", "run", "worker.src.app:app", "--workers", "1"]
