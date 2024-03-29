FROM node:18 AS node_modules
WORKDIR /app
COPY django_app/frontend/package*.json ./
RUN npm install

FROM python:3.11-buster as builder

WORKDIR /app/
# Copy files generated by node_modules build step

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install poetry

ADD django_app/poetry.lock /app/poetry.lock
ADD django_app/pyproject.toml /app/pyproject.toml

RUN --mount=type=cache,target=$POETRY_CACHE_DIR  poetry install --no-ansi --with django-app

FROM python:3.11-slim-buster as runtime

ADD django_app/poetry.lock /app/poetry.lock
ADD django_app/pyproject.toml /app/pyproject.toml

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app/

RUN pip install poetry

RUN poetry install --no-ansi --with django-app

COPY --from=node_modules /app/node_modules /app/django_app/frontend/node_modules

ADD django_app/ /app/django_app/

COPY django_app/docker/web/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8090
CMD ["sh", "start.sh"]