import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Streamlit allows fairly large uploads by default, but background music is usually only a
# few MB. Set an explicit server-side limit so neither the API nor the WebUI writes a huge
# file fully to disk and disturbs video tasks in the same process.
MAX_BGM_UPLOAD_BYTES = 30 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)}
    # Win32 also treats the Latin-1 superscript digits ¹, ² and ³ as device numbers, so
    # they are handled like the plain-digit reserved names to stop COM¹.mp3 from slipping
    # through. The list matches the download file-name rules in webui/Main.py.
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in ("¹", "²", "³")}
)
# File names go verbatim into logs, API responses and the WebUI, so besides path separators
# and Win32 reserved names, every control character and every character that alters the
# displayed form must be rejected. Previously only ord < 32 (C0 controls) was blocked,
# missing the other half of the same problem:
# * C1 controls U+007F-U+009F: some log viewers render U+0085 as a line break, so a single
#   file name can forge extra, seemingly independent log lines.
# * Bidi controls U+200E/U+200F/U+202A-U+202E/U+2066-U+2069: characters after U+202E are
#   rendered reversed, so "photo\u202egnp.mp3" looks like another extension in Explorer
#   and in logs.
# * U+2028/U+2029 (Unicode line/paragraph separators) likewise create visual line breaks
#   inside one log line.
# None of these can be typed into a file-name box, so normal uploads are unaffected; the
# check matches normalize_task_id in app/controllers/base.py (which uses isprintable
# directly).
_UNSAFE_FILENAME_CHARACTERS = frozenset(
    chr(code)
    for code in (
        *range(0x00, 0x20),
        *range(0x7F, 0xA0),
        0x200E,
        0x200F,
        0x2028,
        0x2029,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    )
)
# MoviePy ultimately decodes background music through FFmpeg, so there is no need to limit
# it to MP3. Only mainstream, unambiguous audio extensions are allowed so video containers
# such as MP4 are not uploaded as background music by mistake. The tuple is also the single
# source of truth for the WebUI upload widget, so adding or removing formats later cannot
# leave the frontend and backend out of sync.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """The uploaded file does not meet the safety or format requirements for background music."""


class BgmServiceError(RuntimeError):
    """A server-side execution failure such as FFmpeg or the file system being unavailable."""


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Decide uniformly whether the current task needs any background music handling at all.

    The rule is independent of the source: when no source is selected, the volume is
    invalid or the volume is not above 0, random, custom, Sonilo and any future provider
    must all skip file resolution, external generation and the final mix. Keeping it in
    the generic BGM service avoids copying a zero-volume check for every new provider.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Return the persistent directory for user background music.

    Built-in songs are code resources and stay in resource/songs; user uploads are runtime
    data and must live under the Docker-mounted storage directory so they survive a
    container rebuild and do not pollute the Git working tree.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """Best-effort cleanup of the upload temp file without masking the caller's in-flight exception.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # Temp files use a reserved prefix and never appear in the BGM list; a cleanup
        # failure must not mask the more accurate original exception such as "invalid
        # audio", but it must leave the path and OS error behind for operators.
        logger.warning(
            f"failed to remove staged background music: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Extract a cross-platform display file name, rejecting illegal names and unsupported
    extensions.
    """
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(character in _UNSAFE_FILENAME_CHARACTERS for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("invalid background music filename")

    # Windows treats the first segment before the extension as a device name, so
    # CON.mp3 and LPT1.wav cannot be created as ordinary files. Even though the server
    # ends up using a UUID, rejecting such names early keeps the API's input behaviour
    # consistent across platforms.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("invalid background music filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"unsupported background music format; supported formats: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: int = 30) -> None:
    """
    Verify, using only the project's configured FFmpeg, that the file holds a fully
    decodable audio stream.

    The project allows imageio-ffmpeg to provide a portable FFmpeg, and that install does
    not guarantee FFprobe exists, so no separate binary dependency may be added.
    `-map 0:a:0` fails when there is no audio stream and `-xerror` turns decode errors
    into failures; a full decode also catches encrypted files or random data that
    happen to hit an audio frame header. The file may carry extra streams such as album
    art, but only the first audio stream is checked.
    """
    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                # Uploaded files must be self-contained media, not playlists
                # that cause validation/rendering to read adjacent files or URLs.
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mp3,mov,aac,wav,flac,ogg,asf",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                # Normalize timestamps from decoded samples. A header-only
                # WAV can exit successfully without producing any audio.
                "-af",
                "asetpts=N/SR/TB",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("FFmpeg background music validation timed out") from exc
    except OSError as exc:
        raise BgmServiceError("failed to run FFmpeg for background music validation") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("uploaded file must contain a decodable audio stream")
    decoded_audio = False
    for line in decoded.stdout.splitlines():
        key, separator, value = line.partition(b"=")
        if separator and key == b"out_time_us":
            try:
                decoded_audio = decoded_audio or int(value) > 0
            except ValueError:
                continue
    if not decoded_audio:
        raise BgmUploadError("uploaded file must contain nonempty decodable audio")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Verify that an audio file on disk can be fully decoded by the project's FFmpeg.

    An upload precheck usually needs only 30 seconds; a Sonilo-generated score can last
    up to 6 minutes, hence this reusable entry point with an adjustable timeout. The
    service depends only on FFmpeg and does not require FFprobe to be installed.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("background music file is empty or missing")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Write the upload stream to a temp file in the target directory and return the safe
    file name, temp path and byte count.

    The WebUI upload precheck and the final persist step must use exactly the same
    chunked reads, size limit and file-name rules; otherwise the UI could show the file as
    usable while the server rejects it once generation starts. The caller deletes or
    atomically replaces the temp file after probing the audio.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("failed to prepare background music storage") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("background music upload is not seekable") from exc

        # Keep the original extension so FFmpeg picks the right demuxer for header-less
        # formats such as raw AAC; the temp file still lives in the target directory so the
        # final os.replace is atomic.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("background music upload must be binary")
                total_bytes += len(chunk)
                if total_bytes > MAX_BGM_UPLOAD_BYTES:
                    raise BgmUploadError("background music file exceeds the 30 MB limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("background music file is empty")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("failed to stage background music upload") from exc
        raise
    finally:
        # Streamlit also needs the same UploadedFile for in-browser playback; restoring the
        # file pointer keeps the player and the final save from reading empty content
        # after validation.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Fully validate an uploaded audio file without persisting it, for the WebUI precheck before
    showing "ready".
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"background music upload validated: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Save user background music with chunked reads, a size limit and an atomic replace.

    Callers include FastAPI UploadFile and Streamlit UploadedFile, both of which expose a
    binary file interface. Writing and validating a temp file in the same directory and
    then committing it with os.replace prevents concurrent uploads or process
    interruptions from leaving half an audio file, and gives same-named uploads distinct
    UUID storage keys, so queued or running tasks always reference the original immutable
    file.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("failed to persist background music upload") from exc
        temp_path = ""
        logger.info(
            f"background music uploaded: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def _list_bgm_files(directories: tuple[str, ...]) -> list[str]:
    """Enumerate safe, supported background music files in directory priority order."""
    files_by_name: dict[str, str] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # Both the upload precheck and the final save briefly create files in the same
            # directory. Those temp files carry a valid audio extension but are not yet
            # validated, so the random BGM list must not pick them up early.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # Enumeration results need the real-path check too. Otherwise an attacker
                # could place an audio symlink to an external file in an allowed directory
                # and hand it to MoviePy via the random BGM path.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"skip unsafe background music file: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def list_builtin_bgm_files() -> list[str]:
    """
    List the built-in background music shipped with the project.

    The WebUI "preset songs" and the settings preset import/export use only this list, so
    a saved file name can be restored on another device running the same version; user
    uploads remain managed by the custom music feature.
    """
    return _list_bgm_files((utils.song_dir(),))


def list_bgm_files() -> list[str]:
    """List available uploaded and built-in background music; uploads win on duplicate names."""
    return _list_bgm_files((utils.song_dir(), uploaded_bgm_dir(create=True)))


def resolve_builtin_bgm_file(unsafe_path: str) -> str:
    """Resolve built-in background music by file name, rejecting paths, unknown files and user
    uploads.
    """
    if not unsafe_path:
        raise ValueError("background music filename is required")

    filename = str(unsafe_path)
    if filename != os.path.basename(filename):
        raise ValueError("preset background music must use a filename")

    files_by_name = {
        os.path.basename(file_path): file_path for file_path in list_builtin_bgm_files()
    }
    if filename not in files_by_name:
        raise ValueError("preset background music is not available")
    return files_by_name[filename]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Resolve a BGM file in the user upload directory and the built-in song directory,
    rejecting paths outside the two whitelists and staging files left by interrupted
    uploads.

    A file name hits the user directory first, while legacy forms such as `output000.mp3`,
    absolute whitelisted paths and `./resource/songs/output000.mp3` keep working. New
    uploads use UUIDs and normally never collide with built-in songs or earlier uploads.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("unsupported background music path")
    if os.path.basename(str(unsafe_path)).lower().startswith(_INTERNAL_UPLOAD_PREFIX):
        # Both the upload precheck and the final save briefly create intermediate files
        # with the ``.bgm-upload-`` prefix in the same directory. They carry a valid audio
        # extension but are not validated yet; ``_list_bgm_files`` already excludes them
        # from random BGM, and the resolver must reach the same conclusion, or a name
        # passed in via the API/CLI could hit an incomplete file from an interrupted write.
        # User-supplied names are compared case-insensitively, matching the Windows/macOS
        # file systems.
        raise ValueError("background music upload staging files are not selectable")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("background music file does not exist")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
