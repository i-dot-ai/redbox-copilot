FROM python:3.11-slim-bookworm as builder

WORKDIR /app/

RUN pip install poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Add redbox python package and install it with poetry
ADD redbox/ /app/redbox
ADD pyproject.toml poetry.lock /app/
ADD download_embedder.py /app/
ADD redbox/ /app/redbox

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --no-ansi --with api,ai --without worker,dev

FROM python:3.11-slim-buster as runtime
ARG EMBEDDING_MODEL

WORKDIR /app/

RUN apt-get update && apt-get install --yes build-essential curl > /dev/null

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY core_api/ /app/core_api/
ADD redbox/ /app/redbox
ADD download_embedder.py /app/

# Download the model
RUN type=cache python download_embedder.py --embedding_model ${EMBEDDING_MODEL}

# Run FastAPI
EXPOSE 5002
CMD ["uvicorn", "core_api.src.app:app", "--host", "0.0.0.0", "--port", "5002"]
