"""Safe reading and writing of persisted files in the task directory."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from app.utils import utils


_SCRIPT_LOCKS = tuple(threading.RLock() for _ in range(256))


def _script_lock(task_id: str):
    return _SCRIPT_LOCKS[hash(task_id) % len(_SCRIPT_LOCKS)]


def _script_file(task_id: str) -> Path:
    """Return the task script manifest path, reusing the shared task directory creation logic."""
    return Path(utils.task_dir(task_id)) / "script.json"


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    """
    Atomically write JSON inside the target directory so a process interruption cannot
    leave a half-written file.

    The temp file and target must be in the same directory for ``os.replace`` to keep its
    atomic replace semantics on common local file systems and Docker-mounted directories.
    The existing file is untouched until the write succeeds; on error only the temp file
    created here is removed, and the error is left to the caller to decide whether it
    affects the main flow.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=4,
                default=lambda value: value.__dict__,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_script_data(task_id: str, payload: Mapping[str, Any]) -> None:
    """Create or fully replace the task's ``script.json`` manifest."""
    with _script_lock(task_id):
        _write_json_atomic(_script_file(task_id), payload)


def patch_script_data(task_id: str, **updates: Any) -> bool:
    """
    Add to the task manifest while keeping existing fields; return ``False`` on failure.

    Material sources are auxiliary diagnostic data and must not block video generation
    because of file permissions, a transient disk error or a corrupted old file. This
    entry point therefore logs the full exception and degrades; the initial manifest is
    still created with ``write_script_data``, leaving the main flow to decide how to
    handle a failed write of the base task data.
    """
    try:
        with _script_lock(task_id):
            target = _script_file(task_id)
            with target.open("r", encoding="utf-8") as script_file:
                payload = json.load(script_file)
            if not isinstance(payload, dict):
                raise ValueError("task script data must be a JSON object")

            payload.update(updates)
            _write_json_atomic(target, payload)
            return True
    except FileNotFoundError:
        # ``download_videos`` may also be called on its own by tests, scripts or third-party
        # code; having no task manifest is then normal and should neither warn nor create a
        # partial file just for auxiliary records.
        logger.debug(
            f"skip task script update because script.json does not exist: "
            f"task_id={task_id}"
        )
        return False
    except Exception as exc:
        logger.warning(
            "failed to update task script data: "
            f"task_id={task_id}, fields={sorted(updates)}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False
