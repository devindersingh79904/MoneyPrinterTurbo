import copy
import errno
import json
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__, paths

root_dir = str(paths.runtime_root())
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_diagnostics_update_lock = threading.Lock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Keep the dict usage unchanged while making runtime config writes obey the same lock."""

    def __setitem__(self, key, value):
        # Every full Streamlit rerun writes the current widget values back to config. While a video task holds
        # runtime_config_lock, a write whose value is unchanged has no side effect and must not leave the
        # refreshed page stuck halfway through the form. Writes that really change config still go through the
        # lock below, so the provider, keys, or other global settings cannot change mid-generation.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` likewise leaves config unchanged when the key is absent. The WebUI uses this
        # idiom to mean "use the default policy", and a refresh must let it complete directly.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # As with __setitem__, setdefault on an existing key is read-only. Returning early keeps page refreshes
        # that only read default config unaffected by a long task's config lock.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Produce the pending-update key for a fixed in-process config section."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Update the WebUI runtime config without blocking.

    Video generation holds ``runtime_config_lock`` so a task cannot switch provider, keys, or voice config
    mid-run. A Streamlit widget change cannot wait on this long-task lock, or the browser appears frozen.
    Update immediately when the lock is free; when it is busy, keep only the latest value per config item
    and apply them all when the current task releases the lock.

    Returns True when the value took effect, False when it was queued as pending.
    """
    return update_config_values_nonblocking(config_section, {key: value})


def update_config_values_nonblocking(config_section, changes):
    """Queue the related fields of one operation together, so concurrent application cannot split the config."""
    # Every update enters the same queue before trying the config lock. When several pages modify the same
    # config item at once, queue order is the final order, and an earlier thread that acquires the lock cannot
    # accidentally drop a value a newer thread has already queued.
    with _pending_config_lock:
        for key, value in changes.items():
            _pending_config_updates[_pending_update_key(config_section, key)] = (
                config_section,
                key,
                copy.deepcopy(value),
            )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # The caller usually requests a save at the end of this Streamlit rerun, but that step cannot be relied
        # on. For example, if the page errors midway or the update lands exactly during the task's exit-save
        # phase, a background flush thread is still needed to ensure queued values eventually take effect.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return all(
            config_section.get(key, _MISSING) == value
            for key, value in changes.items()
        )
    finally:
        _config_save_lock.release()


def update_logging_options(level, log_requests):
    """Serially publish the runtime snapshot and the persist intent, without waiting on the generation task's long config lock."""
    from app.utils.logging_utils import get_logging_options, set_logging_options

    with _diagnostics_update_lock:
        set_logging_options(level, log_requests)
        level, log_requests = get_logging_options()
        return update_config_values_nonblocking(
            diagnostics, {"log_level": level, "log_requests": log_requests}
        )


def delete_config_nonblocking(config_section, key):
    """
    Delete a WebUI config item without blocking.

    "Use default" needs the config item really removed, not set to an empty string. While a video task holds
    the config lock, the delete intent overrides earlier queued updates for the same item and runs after the
    task finishes.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Apply the latest config values staged by the WebUI while holding the config write lock."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Keep holding the pending lock while applying. Threads reading the "current + pending" snapshot can
        # then only see the complete state before or after application, never a half-updated config set.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Return an effective snapshot of a config section, merging WebUI updates not yet applied.

    While a video task holds the lock the global config cannot be rewritten, but the user may still prepare
    the next piece of content. With this snapshot, LLM requests pick up the provider, model, and key just
    chosen in the UI without changing the video task in progress.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Apply and save all pending config while holding the config write lock."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # The in-memory config was applied successfully; on save failure keep only the pending-save flag. A video
        # task must not be marked failed because the config file is temporarily unwritable; the next page
        # interaction triggers the save again.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Wait for the long task to release the config lock and reliably flush the config updates accumulated meanwhile."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Ensure at most one background thread waits to flush config at a time."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    try:
        threading.Thread(
            target=_run_deferred_config_flush,
            name="mpt-config-flush",
            daemon=True,
        ).start()
    except Exception:
        # No worker owns this reservation when construction/start fails.
        # Leave queued updates and the save request intact so a later call
        # can schedule another flush rather than silently waiting forever.
        with _pending_config_lock:
            _pending_config_flush_scheduled = False
        raise


def try_save_config():
    """
    Save the WebUI config without blocking; when the lock is busy, save after the current long task ends.

    Plain API, CLI, and maintenance scripts can still call ``save_config`` for the original blocking write
    semantics; only the Streamlit rerun uses this function, so the page does not hang waiting for a video task.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Block other WebUI sessions from rewriting config during one complete operation that depends on global config.

    The project binds to the loopback address by default and config is still single-user global config. This
    lightweight lock mainly protects long operations such as generation and preview, so another tab cannot
    switch the provider or keys midway.
    """
    with _config_save_lock:
        # If the background flush thread was not yet scheduled when the previous short operation released the
        # lock, the new task must apply the queue before reading provider, keys, and other global config, rather
        # than running the whole pipeline on stale config.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Try to acquire the runtime config lock and return immediately whether it succeeded.

    A WebUI preview is a short user-triggered operation and should not wait minutes while a background video
    task holds the lock. The caller can prompt the user to retry later when the lock is unavailable; once
    acquired, the provider, keys, and model config still cannot be changed by other sessions during the preview.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Determine whether the current process runs inside a container.

    This is mainly used to choose the default Ollama address:
    - On a plain local run, `localhost` is the user's own machine;
    - Inside a Docker container, `localhost` is the container itself, and reaching the host's Ollama
      usually requires `host.docker.internal`.

    Checking only whether `/proc/1/cgroup` exists is not enough, since plain Linux has that file too.
    Return True only when an explicit container marker is detected, to avoid misclassifying non-Docker Linux users.
    The parameters stay injectable paths so unit tests can cover different environments.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # The Gateway in /proc/net/route is little-endian hex, e.g. 010011AC means 172.17.0.1. It is parsed
    # separately so that, when native Linux Docker has no host.docker.internal DNS record, the host can
    # still be reached via the container's default gateway.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Read the default gateway IP inside a Linux container.

    Docker Desktop usually provides `host.docker.internal`, but native Linux Docker does not necessarily
    provide that DNS name by default. The default gateway usually serves as a fallback address for host
    services; if the user's Ollama listens only on 127.0.0.1, the user still has to make Ollama listen on
    the host interface or configure `ollama_base_url` manually.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 8:
            continue

        destination = fields[1]
        gateway = fields[2]
        if (
            destination != "00000000"
            or fields[7] != "00000000"
            or gateway == "00000000"
        ):
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Return Ollama's default OpenAI-compatible base_url.

    Not used when the user explicitly configures `ollama_base_url`; this only handles the "best default when
    unset". Inside a container the default points at the host; a plain local run defaults to localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Load TOML, tolerating duplicate UTF-8 BOMs that Windows editors may write.

    ``utf-8-sig`` removes only one BOM at the start of the file. Some Windows editors or unzip/save
    workflows may write another BOM, pushing a second invisible character into the TOML parser and
    failing on the first line. A read-only normalization is done once, only after standard parsing fails,
    without writing the original file back, so a user's filled-in API key is never overwritten by accident.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise



def _initialize_config_from_example(example_file):
    """Publish a complete first-boot config without replacing another writer."""
    fd, temp_path = tempfile.mkstemp(
        prefix=".config-init-", suffix=".toml.tmp",
        dir=os.path.dirname(os.path.abspath(config_file)),
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            with open(example_file, "rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination. POSIX rename
                # overwrites it, so use an exclusive hard-link publication there.
                os.rename(temp_path, config_file)
            else:
                os.link(temp_path, config_file)
        except FileExistsError:
            return False
        except OSError as exc:
            raise OSError(
                exc.errno,
                "cannot safely initialize config.toml; copy config.example.toml "
                "to config.toml before starting the application",
            ) from exc
        return True
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # A successful Windows rename consumed the temporary path.
            pass


def load_config():
    os.makedirs(root_dir, mode=0o700, exist_ok=True)
    # Docker may create an empty directory when a missing config.toml is bind-mounted.
    # Only remove that empty placeholder; a nonempty directory may contain user data.
    if os.path.isdir(config_file):
        try:
            os.rmdir(config_file)
        except FileNotFoundError:
            # Another initializer already removed this empty bind-mount stub.
            pass
        except OSError as exc:
            if exc.errno == errno.EBUSY:
                # The directory is the bind mount itself, so it cannot be
                # removed from inside the container; it must be fixed on the host.
                raise IsADirectoryError(
                    f"{config_file} is a directory because config.toml did not exist "
                    "on the host when Docker mounted it. Stop the containers, delete "
                    "the empty config.toml folder next to the docker-compose file, "
                    "copy config.example.toml to config.toml there, then start again"
                ) from exc
            # Another initializer may already have replaced the empty Docker
            # stub with its complete regular file. Accept only that race, not
            # permission failures or a still-present/nonempty directory.
            if not (
                isinstance(exc, NotADirectoryError) and os.path.isfile(config_file)
            ):
                raise IsADirectoryError(
                    f"{config_file} is a directory and cannot be used as the config file; "
                    "move or rename it, then restart"
                ) from exc

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if not os.path.isfile(example_file):
            example_file = str(paths.config_template())
        if os.path.isfile(example_file):
            if _initialize_config_from_example(example_file):
                logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


class _ConfigTomlEncoder(toml.TomlEncoder):
    """
    Write strings as JSON strings, which are also valid TOML basic strings.

    toml 0.10.2's own string encoder (uiri/toml#404) drops the escaping of
    every backslash up to the last one followed by "x", so a Windows path
    with a folder such as "xiaoming" or "x64" is written as invalid TOML.
    It also writes a non-breaking space as the text "xa0".
    """

    def __init__(self):
        super().__init__()
        self.dump_funcs[str] = self._dump_str

    @staticmethod
    def _dump_str(value):
        # TOML, unlike JSON, also requires DEL to be escaped.
        return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")


def save_config():
    """
    Save the runtime config atomically.

    Different Streamlit sessions may trigger a config save at nearly the same time. Overwriting config.toml
    directly lets another thread read partially written TOML. An in-process reentrant lock serializes saves,
    and the content is first written to a temp file in the same directory, then swapped in atomically via
    os.replace.

    A Docker Desktop single-file bind mount makes config.toml itself the mount point, and the Linux kernel
    refuses to replace a mount point via rename/replace, returning EBUSY.
    In that case the only option is to overwrite the file in place under the lock; other exceptions still
    propagate, so permission, disk, or path errors are not masked.

    This keeps the project's existing single-user global config semantics without adding a complex multi-user
    config system; it mainly prevents config file corruption from multiple tabs or rapid reruns.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["kokoro"] = dict(kokoro)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["voxcpm"] = dict(voxcpm)
        config_to_save["ui"] = dict(ui)
        config_to_save["diagnostics"] = dict(diagnostics)
        serialized_config = toml.dumps(config_to_save, encoder=_ConfigTomlEncoder())

        # The WebUI calls save at the end of a full rerun. Return directly when nothing changed, so every click
        # on an ordinary widget does not cost a disk write and fsync.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
diagnostics = _SynchronizedConfig(_cfg.get("diagnostics", {}))
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
voxcpm = _SynchronizedConfig(_cfg.get("voxcpm", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path:
    # config.toml also allows "~" / "${VAR}" notation: expand first, then check existence,
    # otherwise isfile() never succeeds and the setting is silently ignored.
    ffmpeg_path = os.path.expanduser(os.path.expandvars(ffmpeg_path))
    if os.path.isfile(ffmpeg_path):
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

# A user-set IMAGEIO_FFMPEG_EXE may also use "~" / "${VAR}" notation (issue #1662).
# MoviePy 2.x reads that variable through imageio at import time and freezes it as
# moviepy.config.FFMPEG_BINARY, never re-parsing it afterwards; so the expansion must
# happen in this module (the earliest shared dependency of every entry point, imported
# before moviepy) for the encoding stage to get the real path.
_env_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
if _env_ffmpeg:
    os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.expanduser(
        os.path.expandvars(_env_ffmpeg)
    )

logger.info(f"{project_name} v{project_version}")
