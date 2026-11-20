"""Google Veo text-to-video through the Gemini API (AI Studio key).

Each script keyword becomes one paid Veo clip. Submission uses
``models/{model}:predictLongRunning``; the returned operation is polled until
``done`` and the generated file is downloaded with the same API key. As with the
other paid sources, a submission whose outcome is unknown is never retried
automatically, so one keyword can never be billed twice.
"""

import hashlib
import math
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import generation_progress, google_accounts, material_upload
from app.utils.request_logging import sanitize_payload, trace_http_call

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "veo-3.1-fast-generate-preview"
MODELS = {
    "veo-3.1-fast-generate-preview": "Veo 3.1 Fast",
    "veo-3.1-generate-preview": "Veo 3.1",
    "veo-3.0-fast-generate-001": "Veo 3 Fast",
    "veo-3.0-generate-001": "Veo 3",
    "veo-2.0-generate-001": "Veo 2",
}
# Veo 3.x accepts 4, 6 or 8 seconds; Veo 2 accepts 5-8 seconds.
DEFAULT_DURATIONS = (4, 6, 8)
VEO2_DURATIONS = (5, 6, 7, 8)
SUPPORTED_RESOLUTIONS = ("720p", "1080p")
DEFAULT_PROMPT_TEMPLATE = (
    "Cinematic, realistic stock footage of {term}. Smooth camera motion, "
    "natural lighting, no on-screen text, no captions, no logos, no watermark."
)
DEFAULT_NEGATIVE_PROMPT = "text, captions, subtitles, watermark, logo, blurry, distorted"
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MISSING_KEY_MESSAGE = (
    "Google Veo needs a Gemini API key: pick a Google account or set "
    "google_veo_api_key / gemini_api_key"
)


class GoogleVeoError(RuntimeError):
    def __init__(self, message, task_id=""):
        super().__init__(message)
        self.task_id = task_id


class GoogleVeoUnconfirmedError(GoogleVeoError):
    """The paid operation may exist remotely; never resubmit it automatically."""


class GoogleVeoFilteredError(GoogleVeoError):
    """Google's safety filter produced no clip for this prompt; nothing was billed."""


def get_api_key(settings: Mapping | None = None) -> str:
    """Key of the selected Google account, else google_veo_api_key, gemini_api_key, env."""
    try:
        return google_accounts.resolve_api_key("veo", settings)
    except google_accounts.GoogleAccountError as exc:
        raise GoogleVeoError(f"Google Veo account: {exc}") from exc


def is_enabled(settings: Mapping | None = None) -> bool:
    return google_accounts.has_api_key("veo", settings)


def model_id(settings: Mapping | None = None) -> str:
    values = config.app if settings is None else settings
    # Any veo-* id is accepted so newer Google models work without a code change.
    value = str(values.get("google_veo_model", "") or DEFAULT_MODEL).strip()
    if not value.startswith("veo-"):
        raise GoogleVeoError(f"Unsupported Google Veo model: {value!r}")
    return value


def allowed_durations(settings: Mapping | None = None) -> tuple[int, ...]:
    """Clip lengths (seconds) the configured model accepts."""
    values = config.app if settings is None else settings
    model = str(values.get("google_veo_model", "") or DEFAULT_MODEL).strip()
    return VEO2_DURATIONS if model.startswith("veo-2") else DEFAULT_DURATIONS


def clip_seconds(requested, settings: Mapping | None = None) -> int:
    """Snap the requested clip length to the nearest length the model accepts (rounding up)."""
    allowed = allowed_durations(settings)
    try:
        value = int(requested)
    except (TypeError, ValueError, OverflowError):
        value = allowed[0]
    for seconds in allowed:
        if seconds >= value:
            return seconds
    return allowed[-1]


def _resolution() -> str:
    value = str(config.app.get("google_veo_resolution", "720p") or "720p").strip().lower()
    if value not in SUPPORTED_RESOLUTIONS:
        raise GoogleVeoError(
            f"Unsupported Google Veo resolution {value!r}; expected 720p or 1080p"
        )
    return value


def _base_url() -> str:
    value = str(config.app.get("google_veo_base_url", "") or DEFAULT_BASE_URL).strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.netloc or parts.query or parts.username:
        raise GoogleVeoError("Google Veo base URL must be an HTTPS URL without query or credentials")
    return value


def _setting_seconds(key, default, minimum, maximum):
    try:
        value = float(config.app.get(key, default))
    except (ValueError, TypeError):
        return default
    return min(max(value, minimum), maximum) if math.isfinite(value) else default


def _request(method, url, **kwargs):
    verify = config.app.get("tls_verify", True)
    if isinstance(verify, str):
        verify = verify.lower().strip() not in {"false", "0", "off", "no", ""}
    return trace_http_call(
        method,
        getattr(requests, method.lower()),
        url,
        proxies=config.proxy,
        verify=bool(verify),
        allow_redirects=False,
        **kwargs,
    )


def _clean(text, secret) -> str:
    text = str(text or "")
    if secret:
        text = text.replace(secret, "***")
    return str(sanitize_payload(text))[:500]


def _detail(response, secret) -> str:
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        text = error.get("message", "") if isinstance(error, dict) else str(error)
    except (ValueError, TypeError):
        text = ""
    return _clean(text, secret)


def _prompt(term: str) -> str:
    template = str(config.app.get("google_veo_prompt_template", "") or DEFAULT_PROMPT_TEMPLATE)
    return template.replace("{term}", term) if "{term}" in template else f"{template} {term}"


def _wait(operation, base, headers, secret) -> dict:
    deadline = time.monotonic() + _setting_seconds("google_veo_run_timeout", 1200, 60, 7200)
    interval = _setting_seconds("google_veo_poll_interval", 10, 2, 60)
    generation_progress.remote("google_veo", operation)
    failures = 0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        phase = max(min(remaining / 2, 30), 0.001)
        try:
            response = _request(
                "GET", f"{base}/{operation}", headers=headers, timeout=(phase, phase)
            )
            if response.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            if response.status_code != 200:
                raise GoogleVeoUnconfirmedError(
                    f"Google Veo operation status unavailable: HTTP {response.status_code}, "
                    f"{_detail(response, secret)}",
                    operation,
                )
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("operation response is not an object")
        except GoogleVeoError:
            raise
        except (requests.RequestException, OSError, ValueError) as exc:
            failures += 1
            generation_progress.remote("google_veo", operation, retrying=True)
            if failures > 5:
                raise GoogleVeoUnconfirmedError(
                    "Google Veo polling failed repeatedly; the paid clip may still finish remotely",
                    operation,
                ) from exc
            time.sleep(min(interval, max(deadline - time.monotonic(), 0)))
            continue
        failures = 0
        if body.get("error"):
            raise GoogleVeoError(
                f"Google Veo generation failed: {_clean(body['error'], secret)}", operation
            )
        if body.get("done"):
            return body
        generation_progress.remote("google_veo", operation, "running")
        time.sleep(min(interval, max(deadline - time.monotonic(), 0)))
    raise GoogleVeoUnconfirmedError(
        "Google Veo clip is still generating after the local wait timeout", operation
    )


def _video_uri(body, operation, secret) -> str:
    response = body.get("response") if isinstance(body.get("response"), dict) else {}
    result = response.get("generateVideoResponse") or response
    samples = result.get("generatedSamples") or result.get("generatedVideos") or []
    for sample in samples if isinstance(samples, list) else []:
        video = sample.get("video") if isinstance(sample, dict) else None
        uri = video.get("uri") if isinstance(video, dict) else None
        if isinstance(uri, str) and uri.startswith("https://"):
            return uri
    reasons = result.get("raiMediaFilteredReasons")
    if reasons:
        # A safety filter produced no clip; Google does not bill filtered output.
        raise GoogleVeoFilteredError(
            f"Google Veo blocked this prompt: {_clean(reasons, secret)}", operation
        )
    raise GoogleVeoError("Google Veo finished without a downloadable video", operation)


def _download(uri, headers, output_dir, operation):
    """Send the API key only to Google's API host; drop it once redirected to storage."""
    generation_progress.downloading()
    api_host = urlsplit(uri).hostname
    url = uri
    response = None
    staged = None
    try:
        for _hop in range(6):
            same_host = urlsplit(url).hostname == api_host
            response = _request(
                "GET", url, headers=headers if same_host else {},
                timeout=(30, 180), stream=True,
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            next_url = urljoin(url, response.headers.get("Location", ""))
            response.close()
            response = None
            if urlsplit(next_url).scheme != "https" or not urlsplit(next_url).hostname:
                raise ValueError("video download redirected to an invalid URL")
            url = next_url
        else:
            raise ValueError("too many video download redirects")
        if response.status_code != 200:
            raise ValueError(f"video download failed: HTTP {response.status_code}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(operation.encode()).hexdigest()[:24]
        destination = Path(output_dir) / f"google-veo-{name}.mp4"
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".part", delete=False) as stream:
            staged = Path(stream.name)
            count = 0
            for chunk in response.iter_content(1024 * 1024):
                count += len(chunk)
                if count > MAX_DOWNLOAD_BYTES:
                    raise ValueError("video download exceeds 512 MiB")
                stream.write(chunk)
                generation_progress.downloading(count)
            if count == 0:
                raise ValueError("video download is empty")
        generation_progress.verifying()
        material_upload._validate_video(str(staged))
        from moviepy import VideoFileClip

        with VideoFileClip(str(staged)) as clip:
            duration = float(clip.duration)
            width, height = clip.size
        if not math.isfinite(duration) or duration < 1:
            raise ValueError("generated video is shorter than one second")
        staged.replace(destination)
        return destination, duration, width, height
    except Exception as exc:
        raise GoogleVeoError(
            f"Google Veo generated a paid clip but download/validation failed "
            f"({type(exc).__name__}); operation={operation}",
            operation,
        ) from exc
    finally:
        if response is not None:
            response.close()
        if staged is not None:
            staged.unlink(missing_ok=True)


def generate_video(*, search_term, video_aspect, clip_duration, output_dir) -> MaterialInfo:
    secret = get_api_key()
    if not secret:
        raise GoogleVeoError(MISSING_KEY_MESSAGE)
    model = model_id()
    aspect = VideoAspect(video_aspect)
    if aspect not in {VideoAspect.landscape, VideoAspect.portrait}:
        raise GoogleVeoError("Google Veo supports only 16:9 and 9:16")
    term = str(search_term or "").strip()
    if not term:
        raise GoogleVeoError("Google Veo prompt must not be empty")
    seconds = clip_seconds(clip_duration)
    parameters = {
        "aspectRatio": aspect.value,
        "durationSeconds": seconds,
        "sampleCount": 1,
    }
    negative = str(config.app.get("google_veo_negative_prompt", DEFAULT_NEGATIVE_PROMPT) or "")
    if negative.strip():
        parameters["negativePrompt"] = negative.strip()
    if not model.startswith("veo-2"):
        resolution = _resolution()
        if resolution == "1080p" and seconds != 8:
            # Veo 3.x renders 1080p only as 8-second clips.
            seconds = parameters["durationSeconds"] = 8
        parameters["resolution"] = resolution
    payload = {"instances": [{"prompt": _prompt(term)}], "parameters": parameters}
    base = _base_url()
    headers = {"x-goog-api-key": secret, "Content-Type": "application/json"}
    logger.info(f"generating video with Google Veo: model={model}, term={term!r}, duration={seconds}s")

    generation_progress.submitting("google_veo")
    try:
        response = _request(
            "POST", f"{base}/models/{model}:predictLongRunning",
            json=payload, headers=headers, timeout=(30, 90),
        )
    except requests.RequestException as exc:
        raise GoogleVeoUnconfirmedError(
            "Google Veo submission returned no response; a paid clip may already exist"
        ) from exc
    if response.status_code >= 500 or 300 <= response.status_code < 400:
        raise GoogleVeoUnconfirmedError(
            f"Google Veo submission unconfirmed: HTTP {response.status_code}; not resubmitting"
        )
    if not 200 <= response.status_code < 300:
        raise GoogleVeoError(
            f"Google Veo request rejected: HTTP {response.status_code}, {_detail(response, secret)}"
        )
    try:
        operation = str(response.json().get("name") or "").strip()
    except (ValueError, TypeError, AttributeError):
        operation = ""
    if not operation:
        raise GoogleVeoUnconfirmedError("Google Veo accepted the request without an operation name")
    logger.info(f"Google Veo paid clip started: operation={operation}")

    body = _wait(operation, base, headers, secret)
    uri = _video_uri(body, operation, secret)
    path, duration, width, height = _download(uri, headers, output_dir, operation)
    return MaterialInfo(
        provider="google_veo",
        url=str(path),
        duration=int(duration),
        source_info={
            "provider": "google_veo",
            "search_term": term,
            "asset_id": operation,
            "rendition": {"id": model, "width": width, "height": height},
        },
    )
