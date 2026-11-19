import functools
import os
import re
import sys
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
_PROJECT_ROOT_IN_MESSAGE_RE = re.compile(re.escape(PROJECT_ROOT) + r"(?![\w.\-])")
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# Loguru's default terminal handler has ID 0 at startup. When the WebUI reloads, only this base terminal
# output may be replaced; logger.remove() must not clear every handler, or the temporary sinks that
# running tasks use to collect WebUI logs would be deleted too.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
# Only swap the immutable snapshot; the logging thread never takes the config lock held by generation tasks.
_runtime_logging = ("DEBUG", False)


def get_logging_options() -> tuple[str, bool]:
    return _runtime_logging


def set_logging_options(level: str, log_requests: bool = False) -> None:
    """Adjust this process's logging immediately, without rebuilding handlers or interrupting running tasks."""
    global _runtime_logging
    level = str(level).strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"unsupported log level: {level}")
    if not isinstance(log_requests, bool):
        raise ValueError("log_requests must be a boolean")
    _runtime_logging = (level, log_requests)


def runtime_log_filter(record) -> bool:
    level, log_requests = _runtime_logging
    # Request tracing is a separate switch; users may explicitly enable diagnostics even at ERROR level.
    if record["extra"].get("request_trace"):
        return log_requests
    return record["level"].no >= logger.level(level).no


# The WebUI filters logs by the task worker thread's ID. Helper threads a task starts for parallel downloads,
# clip encoding, and heartbeats have their own thread IDs, so their log records are dropped entirely and
# the longest-running stages show nothing in the WebUI. Record a "helper thread -> thread that started it"
# mapping here so the log filter can attribute helper-thread records back to their task.
_log_scope_roots: dict[int, int] = {}
_log_scope_lock = threading.Lock()


def log_scope_thread_id(thread_id: int | None = None) -> int:
    """Return the root thread ID of the thread's logging scope; an unbound thread is its own root."""
    if thread_id is None:
        thread_id = threading.get_ident()
    with _log_scope_lock:
        return _log_scope_roots.get(thread_id, thread_id)


def bind_log_scope(func):
    """
    Wrap a function that will run in another thread so its logs are attributed to the thread calling this.

    Must be called from the thread submitting the task: the scope is fixed at wrap time, and threads started
    by helper threads are also attributed to the original task thread. The binding is removed after execution
    because the OS reuses thread IDs, and a stale mapping would count later unrelated threads' logs toward a
    task that has already finished.
    """
    root_thread_id = log_scope_thread_id()

    @functools.wraps(func)
    def run_in_log_scope(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id == root_thread_id:
            return func(*args, **kwargs)

        with _log_scope_lock:
            previous_root = _log_scope_roots.get(thread_id)
            _log_scope_roots[thread_id] = root_thread_id
        try:
            return func(*args, **kwargs)
        finally:
            with _log_scope_lock:
                # Thread pools reuse threads: restore the previous attribution instead of always deleting, so
                # nested wrapping does not clear a binding the outer level still uses.
                if previous_root is None:
                    _log_scope_roots.pop(thread_id, None)
                else:
                    _log_scope_roots[thread_id] = previous_root

    return run_in_log_scope


def _project_relative_path(file_path):
    """
    Shorten an absolute path to a project-relative path starting with ``./`` and always using forward slashes.

    On Windows the project may be launched through a mapped network drive or a ``subst`` drive. The paths in
    the call stack are then still ``X:\\MoneyPrinterTurbo\\...``, while ``PROJECT_ROOT`` resolves through
    ``realpath`` to ``C:\\...``, and ``os.path.relpath`` raises ``ValueError`` outright.
    An exception in the format function is caught by loguru and the whole record is dropped, emptying both
    the terminal and the WebUI log panel, so this must fall back to the original path. The same goes for
    files outside the project directory: prefixing ``./`` to a ``..`` path only makes it harder to read.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # On Windows relpath returns a backslash-separated path; joining directly would yield mixed-separator
    # output like ``./app\\utils``, inconsistent with logs on other platforms.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Format terminal and WebUI logs uniformly.

    Loguru hands the same record to multiple sinks. The first sink may already have converted the absolute
    path to a project-relative one, so both absolute paths and already-formatted ``./`` paths are handled here.
    The WebUI sink disables colors, but time, level, call site, and message stay identical to the terminal.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # Log messages sometimes contain absolute paths to task files. Shortening them to project-relative paths
    # keeps the WebUI and terminal from showing two different versions due to different init entry points.
    # Only replace the complete project root path; sibling directories like `<root>-backup` and `<root>2` merely
    # share the prefix, and a substring replacement would turn them into wrong paths like `.-backup` and `.2`.
    record["message"] = _PROJECT_ROOT_IN_MESSAGE_RE.sub(".", record["message"])
    return LOG_RECORD_FORMAT


def force_utf8_console() -> None:
    """
    Switch stdout and stderr to UTF-8 before anything is logged or printed.

    When they go to a file or a pipe on Windows, as with
    ``webui.bat > webui.log 2>&1``, Python encodes them in the ANSI code page,
    such as cp1252 on English and cp936 on Chinese Windows. A log record with
    a character outside it (the circled digits of generate_video(), Chinese
    on cp1252, emoji on cp936) raises UnicodeEncodeError in the terminal sink,
    and Loguru drops the record and prints "Logging error in Loguru Handler"
    in its place. The CLI's final print() failed the same way after the video
    was written. Streams that cannot be switched, such as None under pythonw,
    are left as they are.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Safely replace the process-level terminal log handler while keeping task-specific handlers.

    Streamlit may re-run logging initialization on hot reload or cache invalidation. Only the old terminal
    output is removed, precisely by its recorded handler ID, so WebUI logs that background tasks are writing
    are not interrupted. The lock protects the ID update when several browser sessions initialize at once.
    """
    global _terminal_handler_id

    set_logging_options(level, get_logging_options()[1])

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # A test or external entry point may already have removed this handler. Keep creating the new
                # terminal output without affecting other still-valid log sinks.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=0,
            filter=runtime_log_filter,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
