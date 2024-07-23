FROM python:3.12-bookworm as builder

RUN apt update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app/worker

COPY worker/pyproject.toml worker/poetry.lock .
COPY redbox-core/ /app/redbox-core
COPY README.md /app/README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

# Downlaod Sentence Tokenizer
RUN poetry run python -m nltk.downloader punkt

# Download POS tagger
RUN poetry run python -m nltk.downloader averaged_perceptron_tagger

# # Download the OCR models
COPY worker/download_ocr_models.py .
RUN poetry run python -m download_ocr_models

FROM python:3.12-slim-bookworm as runtime

ENV VIRTUAL_ENV=/app/worker/.venv \
    PATH="/app/worker/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app/worker

COPY redbox-core/ /app/redbox-core
COPY worker/ /app/worker

RUN chmod +x health.sh

CMD ["poetry", "run", "faststream", "run", "worker.app:app"]
