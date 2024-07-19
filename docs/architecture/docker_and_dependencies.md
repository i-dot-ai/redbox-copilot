# Docker and Dependencies

Each of the Microservices in the Redbox system is built into a Docker container. This allows us to run the Microservices in a consistent environment across different platforms. It also allows us to easily scale the number of instances of a Microservice up or down depending on the load.


## Poetry Groups

We use Poetry to manage the dependencies for each of the Microservices. Poetry is a Python dependency management tool that allows us to specify the dependencies for a Microservice in a `pyproject.toml` file. 

For example, the `core-api` Microservice has the following groups:

```bash
cd core-api
poetry install --without dev
```

!!! warning "Be explicit with poetry groups"
    Note that you have to explicitly state which groups you want to install and which to exclude. This is because Poetry will install all groups by default.


## Buildkit

We are using Buildkit to build our Docker containers. Buildkit is a new build system that is part of Docker. It is faster and more efficient than the old build system. It also has some new features that make it easier to build Docker containers. 

We use multi-stage builds to keep the size of our Docker containers small. This means that we have a separate `builder` stage and a separate `runner` stage. The `builder` stage is used to install depenencies to a virtualenv using poetry and then we copy that virtualenv to the `runner` stage. This keeps the size of the `runner` stage small.

### Example Dockerfile

```Dockerfile
# BUILDER
FROM python:3.11-bookworm as builder

WORKDIR /app/

RUN pip install poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock /app/
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install

# RUNNER

FROM python:3.11-slim-buster as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

CMD ["python", "-V"]
```
