import copy
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import asdict

from loguru import logger

from app.config import config
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm
from app.services import codex_image, jalapeno_video, volcengine_seedance
from app.services.loomloom import LoomLoomConfirmedVideoRequest
from app.utils.logging_utils import (
    format_log_record,
    log_scope_thread_id,
    runtime_log_filter,
)


# The WebUI config lives in a process-level global dict. The original synchronous implementation
# held runtime_config_lock for the whole generation, so different browser sessions effectively
# ran serially anyway. Concurrency is pinned to 1 here, which keeps the original config
# consistency and avoids multiple threads pointlessly waiting outside the config lock.
_task_manager = InMemoryTaskManager(
    max_concurrent_tasks=1,
    max_queued_tasks=max(1, int(config.app.get("max_queued_tasks", 100))),
)
_task_logs: dict[str, deque[str]] = {}
_task_logs_lock = threading.RLock()
_MAX_LOG_TASKS = 20
_MAX_LOG_RECORDS_PER_TASK = 1000
_submission_lock = threading.RLock()
_active_submissions: dict[str, str] = {}
# Streamlit cannot push widget updates from a background thread; Fragment polling is the only
# option. 0.5 seconds is enough to keep WebUI logs close to real-time terminal output without
# the constant browser load of high-frequency refreshes.
TASK_LOG_REFRESH_INTERVAL_SECONDS = 0.5


def is_generation_active(task_id: str) -> bool:
    """Use the background reservation as the source of truth, covering the window between enqueue and state publication."""
    with _submission_lock:
        return task_id in _active_submissions.values()


def _submission_signature(
    params, references, request, endpoint, preview, codex_image_context=None
):
    # Keep only a digest; never store credentials or uploaded content. Different key/model/input
    # means a different request, and the same content may be generated again after completion;
    # only duplicate submissions still queued or running are merged.
    request_inputs = asdict(request) if request is not None else None
    if request_inputs is not None:
        # Two tabs each generate their own idempotency ID for the same quote input; that is not
        # a content difference. Keep account, model, input rows and quote version, and continue
        # with the first confirmed request.
        request_inputs.pop("client_request_id", None)
    payload = {
        "params": params.model_dump(mode="json", warnings=False),
        "settings": {
            name: config.snapshot_config_with_pending(getattr(config, name))
            for name in ("app", "azure", "elevenlabs", "siliconflow", "chatterbox")
        },
        "references": [hashlib.sha256(value or b"").hexdigest() for value in references],
        "request": request_inputs,
        "endpoint": asdict(endpoint) if endpoint is not None else None,
        # The voiceover source changes the actual output; the temp path varies with the local
        # task ID and is not an input difference.
        "voice_preview": {
            key: preview[key]
            for key in ("script", "voice_name", "voice_rate", "voice_volume", "duration", "params", "audio_digest")
            if key in preview
        } if preview else None,
    }
    if codex_image_context is not None:
        payload["codex_image"] = asdict(codex_image_context)
    if preview and not preview.get("audio_digest"):
        # When an old caller provides no content digest, the same duration alone does not prove
        # two files are identical. Never read arbitrary paths before validation; rather keep
        # separate requests for different files.
        payload["voice_preview"]["audio_file"] = preview.get("audio_file")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _append_task_log(task_id: str, message: str) -> None:
    """Keep a bounded number of log lines per task for safe polling by a Streamlit Fragment."""
    with _task_logs_lock:
        records = _task_logs.get(task_id)
        if records is None:
            # Keep logs only for recent tasks so a long-running WebUI service does not keep
            # growing in memory. dict preserves insertion order; task logs are only for UI
            # diagnostics, so evicting the oldest records does not affect tasks.
            if len(_task_logs) >= _MAX_LOG_TASKS:
                oldest_task_id = next(iter(_task_logs))
                _task_logs.pop(oldest_task_id, None)
            records = deque(maxlen=_MAX_LOG_RECORDS_PER_TASK)
            _task_logs[task_id] = records
        records.append(message.rstrip())


def get_task_logs(task_id: str) -> list[str]:
    """Return a log snapshot so page rendering never holds the lock used by background threads."""
    with _task_logs_lock:
        return list(_task_logs.get(task_id, ()))


def _run_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context: codex_image.CodexImageContext | None = None,
) -> dict:
    """
    Run the existing video pipeline in a background thread.

    Loguru sinks are a process-level resource, so they must be filtered by the current worker
    thread; otherwise logs from concurrently running API tasks or other pages leak into this
    task. Helper threads the task starts for downloads, clip encoding and heartbeats are
    attributed to the worker thread via bind_log_scope, and their logs must be collected too,
    otherwise those long phases show nothing in the WebUI. The page only reads plain list
    snapshots and never touches Streamlit session_state from a background thread, which rules
    out delta path corruption on refresh at the source.
    """
    log_handler_id = None
    worker_thread_id = threading.get_ident()
    try:
        if capture_logs:
            log_handler_id = logger.add(
                lambda message: _append_task_log(task_id, str(message)),
                level=0,
                format=format_log_record,
                colorize=False,
                filter=lambda record: (
                    log_scope_thread_id(record["thread"].id) == worker_thread_id
                    and runtime_log_filter(record)
                ),
            )

        # The whole task still uses the original config lock, preventing another WebUI session
        # from changing process-level config such as provider or keys mid-generation, which
        # would make one video use different settings at different points.
        with config.runtime_config_lock():
            return tm.start(
                task_id=task_id,
                params=params,
                voice_preview=voice_preview,
                loomloom_video_request=loomloom_video_request,
                voxcpm_reference_audio=voxcpm_reference_audio,
                voxcpm_prompt_audio=voxcpm_prompt_audio,
                voxcpm_prompt_text=voxcpm_prompt_text,
                **({"codex_image_context": codex_image_context}
                   if codex_image_context is not None else {}),
                **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
                   if jalapeno_video_endpoint is not None else {}),
            )
    except Exception as exc:
        # tm.start already turns pipeline exceptions into a failed state; this additionally
        # guards the WebUI wrapper layer (log sink, config lock, etc.). Any background thread
        # exception must leave a terminal state, so the task manager never shows "generating"
        # forever after the worker thread exits.
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "task_id": task_id,
            "state": const.TASK_STATE_FAILED,
            "progress": 0,
            "failed_stage": "webui_worker",
            "error": error,
        }
        sm.state.update_task(
            task_id,
            state=failure["state"],
            progress=failure["progress"],
            failed_stage=failure["failed_stage"],
            error=failure["error"],
        )
        logger.exception(
            f"unexpected WebUI generation worker failure, "
            f"task_id={task_id}, error={exc}"
        )
        return failure
    finally:
        with _submission_lock:
            for signature, active_id in list(_active_submissions.items()):
                if active_id == task_id:
                    del _active_submissions[signature]
        if log_handler_id is not None:
            try:
                logger.remove(log_handler_id)
            except ValueError:
                logger.debug(
                    f"WebUI task log handler already removed: task_id={task_id}"
                )


def submit_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool = True,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
) -> str:
    """
    Register and submit a WebUI video generation task, returning immediately.

    The task state must be written before the thread starts. That way the task is queryable as
    soon as this page script run ends, and a browser refresh or WebSocket reconnect does not
    depend on placeholders in the old page's memory.
    """
    task_params = params.model_copy(deep=True)
    # The subtitle timeline and nested preview parameters remain mutable in
    # Streamlit session state. Snapshot them before this job enters the queue.
    voice_preview_snapshot = copy.deepcopy(voice_preview) if voice_preview else None
    # Reference audio belongs only to this queued request. It is deliberately
    # separate from VideoParams so it cannot reach task history, presets, state
    # persistence, or logs. ``bytes`` is immutable; make an explicit snapshot
    # before the background worker starts so later WebUI reruns cannot share
    # mutable upload state with this task.
    voxcpm_reference_audio_snapshot = (
        bytes(voxcpm_reference_audio) if voxcpm_reference_audio else None
    )
    voxcpm_prompt_audio_snapshot = (
        bytes(voxcpm_prompt_audio) if voxcpm_prompt_audio else None
    )
    voxcpm_prompt_text_snapshot = str(voxcpm_prompt_text or "")
    # The confirmed request is a frozen data object passed only within this process. The API
    # key never enters VideoParams, task state, logs or on-disk history, and is not affected by
    # later page reruns.
    loomloom_request_snapshot = loomloom_video_request
    # Preparing the next piece of content is allowed while queued, but a confirmed billed config
    # must not follow the global form. The snapshot's key is excluded from repr, is only passed
    # with in-process task params, and is never written into VideoParams.
    if task_params.video_source == "jalapeno" and jalapeno_video_endpoint is None:
        jalapeno_video_endpoint = jalapeno_video.endpoint(
            config.snapshot_config_with_pending(config.app)
        )
    codex_image_context = (
        codex_image.capture_context(config.snapshot_config_with_pending(config.app))
        if task_params.video_source == "codex_image"
        else None
    )
    signature = _submission_signature(
        task_params,
        (
            voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text_snapshot.encode(),
        ),
        loomloom_request_snapshot,
        jalapeno_video_endpoint,
        voice_preview_snapshot,
        codex_image_context,
    )
    # Deduplicate and reserve under the same lock, so two tabs clicking at once still enqueue
    # only once.
    with _submission_lock:
        existing = _active_submissions.get(signature)
        if existing:
            return existing
        if task_id in _active_submissions.values():
            return task_id
        _active_submissions[signature] = task_id
    try:
        _enqueue_generation(
            task_id,
            task_params,
            capture_logs,
            voice_preview_snapshot,
            loomloom_request_snapshot,
            voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text_snapshot,
            jalapeno_video_endpoint,
            codex_image_context,
        )
    except Exception:
        with _submission_lock:
            if _active_submissions.get(signature) == task_id:
                del _active_submissions[signature]
        raise
    return task_id


def _enqueue_generation(
    task_id,
    task_params,
    capture_logs,
    voice_preview_snapshot,
    loomloom_request_snapshot,
    voxcpm_reference_audio_snapshot,
    voxcpm_prompt_audio_snapshot,
    voxcpm_prompt_text_snapshot,
    jalapeno_video_endpoint,
    codex_image_context=None,
):
    submitted_at = time.time()
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        video_subject=task_params.video_subject or task_params.video_script or task_id,
        generation_submitted_at=submitted_at,
        generation_activity={"phase": "queued", "started_at": submitted_at,
                             "updated_at": submitted_at},
    )
    try:
        _task_manager.add_task(
            _run_generation,
            task_id=task_id,
            params=task_params,
            capture_logs=capture_logs,
            voice_preview=voice_preview_snapshot,
            loomloom_video_request=loomloom_request_snapshot,
            voxcpm_reference_audio=voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio=voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text=voxcpm_prompt_text_snapshot,
            **({"codex_image_context": codex_image_context}
               if codex_image_context is not None else {}),
            **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
               if jalapeno_video_endpoint is not None else {}),
        )
    except Exception as exc:
        # A scheduling failure, like a pipeline failure, must become a queryable state so the
        # task manager never shows "generating" forever. Keep the exception type to quickly
        # locate queue problems from Docker or local logs.
        error = f"{type(exc).__name__}: {exc}"
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=0,
            failed_stage="scheduling",
            error=error,
        )
        logger.exception(
            f"failed to submit WebUI generation task, task_id={task_id}, error={exc}"
        )
        raise
