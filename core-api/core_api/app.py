import logging
import os
from datetime import UTC, datetime
from http import HTTPStatus

from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse

from core_api.routes.chat import chat_app
from core_api.routes.file import file_app
from redbox import __version__ as redbox_version
from redbox.models import Settings, StatusResponse

# === Logging ===

filehandler = logging.FileHandler("coreapi.log")
filehandler.setLevel(logging.INFO)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger()
logger.addHandler(filehandler)

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
def health(response: Response) -> StatusResponse:
    """Returns the health of the API

    Returns:
        StatusResponse: The health of the API
    """

    uptime = datetime.now(UTC) - start_time
    uptime_seconds = uptime.total_seconds()

    logger.info("es: %s", env.elasticsearch_client())
    ping = env.elasticsearch_client().ping()
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
