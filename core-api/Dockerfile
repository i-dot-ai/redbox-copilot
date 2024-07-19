FROM python:3.11-bookworm as builder

RUN apt-get update && apt-get install --yes build-essential curl > /dev/null

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app/core-api

COPY core-api/pyproject.toml core-api/poetry.lock .
COPY redbox-core/ /app/redbox-core
COPY README.md /app/README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.11-slim-bookworm as runtime

ENV VIRTUAL_ENV=/app/core-api/.venv \
    PATH="/app/core-api/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app/core-api

# Add redbox python package and install it with poetry
ADD core-api/ /app/core-api

# Run FastAPI
EXPOSE 5002
CMD ["poetry", "run", "uvicorn", "core_api.app:app", "--host", "0.0.0.0", "--port", "5002"]
