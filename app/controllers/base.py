import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request

from app.config import config
from app.models.exception import HttpException

MAX_TASK_ID_LENGTH = 128


def normalize_task_id(value: object) -> str:
    """Return a log-safe request ID, replacing invalid client input with a UUID."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TASK_ID_LENGTH
        or not value.isprintable()
    ):
        return str(uuid4())
    return value


def get_task_id(request: Request) -> str:
    return normalize_task_id(request.headers.get("x-task-id"))


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def get_api_key_values(request: Request) -> list[str]:
    """Return every API key header in the request, keeping duplicates for security validation."""

    # Starlette Headers provide getlist(), which can distinguish duplicate headers sent by a
    # proxy or client. The lightweight Request stand-ins in unit tests use a plain dict, so keep
    # a compatibility fallback.
    get_list = getattr(request.headers, "getlist", None)
    if callable(get_list):
        return [value for value in get_list("x-api-key") if isinstance(value, str)]

    api_key = get_api_key(request)
    return [api_key] if isinstance(api_key, str) else []


def verify_token(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
):
    """Validate the API key depending on configuration.

    An empty key keeps the existing local no-auth mode; once an admin explicitly configures a
    non-empty key, API routes and task artifact downloads require the client to supply the same
    value via the ``x-api-key`` header. The parameter declaration also makes Swagger show the
    header, which helps debugging in protected environments.
    """

    configured_key = config.app.get("api_key", "")
    if configured_key in (None, ""):
        return None

    # The config value must be a string. Reject wrong types such as lists or numbers so implicit
    # string conversion cannot produce hard-to-spot auth behavior; the error message also never
    # includes the actual key.
    if not isinstance(configured_key, str):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=500,
            message="API authentication is misconfigured",
        )

    # The FastAPI parameter declares x-api-key in OpenAPI; the actual check always reads the
    # Request so it can detect the same header being sent more than once. Ordinary clients and
    # reverse proxies may pick duplicate headers in different orders, so they must be rejected
    # rather than silently using the first or last value.
    token_values = get_api_key_values(request)
    if not token_values and isinstance(x_api_key, str):
        token_values = [x_api_key]

    if len(token_values) != 1:
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    # compare_digest only supports ASCII for str. Request headers are untrusted input, and an
    # attacker could send Latin-1 characters to trigger a TypeError. Encoding both sides to
    # UTF-8 bytes keeps the constant-time comparison and supports Unicode keys that are valid
    # in TOML.
    token = token_values[0]
    if not secrets.compare_digest(
        token.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    return None
