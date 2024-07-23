import logging
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated

from elasticsearch import Elasticsearch
from fastapi import Depends, FastAPI, Response
from fastapi.responses import RedirectResponse

from core_api.dependencies import get_elasticsearch_client
from core_api.routes.chat import chat_app
from core_api.routes.file import file_app
from redbox import __version__ as redbox_version
from redbox.models import Settings, StatusResponse

print('Lewis is testing')

# === Logging ===

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

env = Settings()

# === API Setup ===

start_time = datetime.now(tz=UTC)


# Create API

app = FastAPI(
    title="Core API",
    description="Redbox Core API",
    version=redbox_version,
    openapi_tags=[
        {"name": "health", "description": "Health check"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# === API Routes ===

# Basic Setup


@app.get("/", include_in_schema=False, response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", status_code=HTTPStatus.OK, tags=["health"])
def health(response: Response, es: Annotated[Elasticsearch, Depends(get_elasticsearch_client)]) -> StatusResponse:
    """Returns the health of the API

    Returns:
        StatusResponse: The health of the API
    """

    uptime = datetime.now(UTC) - start_time
    uptime_seconds = uptime.total_seconds()

    logger.info("es: %s", es)
    ping = es.ping()
    if ping:
        status = "ready"
    else:
        status = "unavailable"
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
    logger.info("status: %s", status)
    return StatusResponse(
        status=status,
        uptime_seconds=uptime_seconds,
        version=app.version,
    )


app.mount("/chat", chat_app)
app.mount("/file", file_app)
