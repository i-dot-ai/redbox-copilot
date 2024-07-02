FROM python:3.11-buster as builder

WORKDIR /app/

RUN pip install poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Add redbox python package and install it with poetry
ADD redbox-core/ /app/redbox-core
ADD pyproject.toml poetry.lock README.md /app/

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --no-ansi --with api,ai --without worker,dev

FROM python:3.11-slim-bookworm as runtime

WORKDIR /app/

RUN apt-get update && apt-get install --yes build-essential curl > /dev/null

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY core_api/ /app/core_api/
ADD redbox-core/ /app/redbox-core
ADD README.md /app/README.md

# Run FastAPI
EXPOSE 5002
CMD ["uvicorn", "core_api.src.app:app", "--host", "0.0.0.0", "--port", "5002"]
