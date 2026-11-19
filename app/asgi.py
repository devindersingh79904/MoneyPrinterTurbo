"""Application implementation - ASGI."""

import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Centralise API process startup recovery and shutdown logging."""
    logger.info("startup event")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "API key authentication is disabled; keep the API on a trusted network"
        )
    elif isinstance(configured_api_key, str):
        # Log only the protected scope; never output the key, its length or a digest, so
        # credentials stay out of the logging system.
        logger.info("API key authentication is enabled for /api/v1 and /tasks")
    else:
        logger.error(
            "API key authentication is misconfigured: app.api_key must be a string"
        )

    # Cross-platform publishing runs in this process's thread pool and does not resume
    # after a service restart. On startup, settle any active state in Redis whose worker
    # process is confirmed gone into failure so tasks never become undeletable.
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()

    # Redis queue entries persist across API restarts. No task_done callback
    # exists in the new process to dispatch them, so fill its worker slots now.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    redis_task_manager = None
    if isinstance(video_controller.task_manager, RedisTaskManager):
        redis_task_manager = video_controller.task_manager
        redis_task_manager.resume_queued_tasks()
        redis_task_manager.start_queue_dispatcher()
    try:
        yield
    finally:
        if redis_task_manager is not None:
            redis_task_manager.stop_queue_dispatcher()
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Rejected inputs can contain NaN/Infinity, and custom validators attach
    # exception objects to ctx. Neither can be emitted by JSONResponse directly.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    # Describe the first rejected field instead of always claiming a missing
    # field: out-of-range values and malformed JSON are not "field required".
    message = "invalid request parameters"
    if errors and isinstance(errors[0], dict):
        location = ".".join(str(part) for part in errors[0].get("loc") or ())
        detail = errors[0].get("msg") or message
        message = f"{location}: {detail}" if location else str(detail)
    return JSONResponse(
        status_code=400,
        content=utils.get_response(status=400, data=errors, message=message),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """Fold a configured entry into the Origin form a browser actually sends.

    RFC 6454 fixes the Origin request header as ``scheme://host[:port]``: no path, no
    trailing slash, scheme and host normalised to lower case and the protocol's default
    port omitted. The ``https://frontend.example/`` a user copies from the browser address
    bar, as well as ``https://frontend.example:443`` with the default port spelled out, are
    not character-for-character equal to the ``https://frontend.example`` the browser
    sends, so the whitelist silently fails: the frontend only sees a CORS error and the
    server only leaves one blocked log line, neither pointing at the configuration itself.

    Returning ``None`` means the entry can never match any Origin (missing scheme, scheme
    other than http/https, empty or whitespace-containing host name, malformed IPv6
    literal, invalid or out-of-range port); the caller should drop it and warn. Parse
    failures must collapse into ``None``: this function runs at module import time, so a
    ``ValueError`` escaping from ``urlsplit()`` would stop the whole API from starting.
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # netloc is only validated when reading hostname / port: a malformed IPv6 literal
        # is raised by urlsplit(), a non-numeric or out-of-0-65535 port when reading port.
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # Only http/https carry browser Origin semantics, and both have an omittable default port.
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # Lower-case explicitly rather than relying on urlsplit's case handling across Python versions.
    host = host.lower()
    # Browsers keep the square brackets around IPv6 literals when serialising an Origin;
    # default-port spellings such as ``:443``, ``:0443`` and ``:80`` must be folded into
    # the port-less form to match what a real frontend sends.
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """Parse the browser cross-origin whitelist.

    CORS only constrains cross-origin JavaScript running in a browser; it does not affect
    curl, Postman, n8n or server-side SDKs. When unset, return an empty list, meaning
    cross-origin access is closed by default; users who really deploy a separate web
    frontend enable it explicitly via ``CORS_ALLOWED_ORIGINS``.

    Each origin is first folded into the canonical Origin header form, so
    ``https://a.example/``, ``HTTPS://A.Example`` and ``https://a.example:443`` with the
    explicit default port all yield the same result, and spellings that can never match
    an Origin are dropped with a warning. A malformed entry only drops itself and never
    aborts parsing here: this function is called at module import time, and a parse
    exception would keep the API from starting.
    """

    if not raw_origins:
        return []

    # Strip whitespace around comma-separated items and ignore empty ones, so the common
    # environment variable form ``https://a.example, https://b.example,`` does not
    # produce origins that can never match.
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"ignoring configured CORS origin that cannot match a browser "
                f"Origin header: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure CORS from an explicit whitelist; an empty whitelist keeps the default same-origin
    policy.
    """

    if not allowed_origins:
        logger.info(
            "browser cross-origin API access is disabled; set "
            "CORS_ALLOWED_ORIGINS to enable trusted origins"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` is a compatibility mode the user chose explicitly, so startup is not refused;
        # but without authentication it lets any web page read and call the API, so a
        # traceable security warning must be left behind.
        logger.warning(
            "CORS allows every browser origin while API key authentication is "
            "disabled; configure app.api_key or restrict CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Starlette reflects any Origin when ``*`` and credentials are both enabled. The
        # wildcard mode needs no cookie authentication, so credentials are switched off
        # deliberately; explicit origins keep the old behaviour so the credentials request
        # mode of existing separate web frontends is unaffected.
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # When a remote HTTPS frontend reaches an API on localhost or the LAN, modern
        # browsers send an extra Private Network Access preflight. Only exact whitelisted
        # origins are granted; the wildcard mode keeps refusing so arbitrary sites cannot
        # probe the user's private network services.
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """Return whether the browser request origin is same-origin or an explicitly whitelisted origin.
    """

    origin = request.headers.get("origin")
    if not origin:
        # curl, Postman, n8n and server-side SDKs usually send no Origin. Keep such requests
        # so the security fix does not accidentally change the contract of existing API clients.
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # Browsers may send Origin on same-origin POSTs too. Compare only scheme + authority
    # and ignore path and query; a reverse-proxy deployment that does not forward the
    # public scheme/host correctly can declare the external origin via an explicit
    # CORS_ALLOWED_ORIGINS instead of relying on untrusted forwarded headers.
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure both the server-side Origin guard and the browser CORS response policy."""

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """Actively reject untrusted browser origins, covering simple requests that need no CORS
        preflight.
        """

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"blocked untrusted browser origin: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="cross-origin browser request is not allowed",
                ),
            )

        return await call_next(request)

    # Registered last, the CORS middleware sits outside the Origin guard: trusted
    # preflights succeed directly, untrusted ones are refused by CORS, and actual requests
    # that need no preflight still hit the 403 guard above.
    configure_cors(instance, allowed_origins)


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """Protect the static route for task artifacts so API auth cannot be bypassed by direct download.

    ``/tasks`` is mounted separately via StaticFiles and cannot reuse APIRouter
    dependencies, so the same verify_token is called from middleware. The auth function
    lets requests through when api_key is not configured; OPTIONS preflights are also
    left to the CORS middleware.
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    if is_task_file and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# Follow the browser same-origin policy by default; open cross-origin access only when the user
# explicitly configures trusted web origins.
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
