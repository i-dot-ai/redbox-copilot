# npm

FROM node:18 AS npm-packages

WORKDIR /src

COPY django_app/frontend/package.json .
RUN npm install

COPY django_app/frontend/staticfiles.json django_app/frontend/.parcelrc django_app/frontend/*.js ./
COPY django_app/frontend/src/ ./src/
COPY django_app/frontend/images/ ./images/
RUN npx parcel build


# poetry
FROM python:3.11-slim AS poetry-packages

RUN apt-get update && apt-get install --yes build-essential > /dev/null

RUN pip install poetry poetry-plugin-bundle

WORKDIR /src

COPY django_app/pyproject.toml .
COPY django_app/poetry.lock .

# do this so that poetry bundle can run without the project - can't pass --no-root to bundle
#RUN touch README.md
RUN poetry install
RUN poetry bundle venv ./venv


# app
FROM python:3.11-slim

RUN apt-get update && apt-get install --yes libpq-dev curl > /dev/null

WORKDIR /usr/src/app

COPY django_app/ .

COPY --from=npm-packages /src/node_modules/ frontend/node_modules/
COPY --from=npm-packages /src/dist/ frontend/dist/
COPY --from=poetry-packages /src/venv ./venv

ENV DJANGO_SETTINGS_MODULE='redbox_app.settings'
ENV PYTHONPATH "${PYTHONPATH}:/."

EXPOSE 8090

RUN chmod +x start.sh
CMD ["./start.sh"]
