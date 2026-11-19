import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from typing import Callable, List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    ImageMotion,
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import subtitle_shaping, video_effects
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# The ffmpeg/AAC combination in Docker is prone to audio quality fluctuations with the default
# settings, so the audio bitrate is raised explicitly here to avoid audible distortion in the
# final render caused by a too-low default.
audio_bitrate = "192k"
fps = 30
# When FFmpeg concatenates/transcodes by frame rate, the final duration can be tens of
# milliseconds shorter than the theoretical duration MoviePy reads. A tiny safety margin is
# added to the video material here so frame rounding does not leave a black screen, a stall or
# a final bit of narration without picture at the end of the audio.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# Messaging apps and some encoders round frame dimensions down; WhatsApp, for example,
# squeezes 9:16 material to 478x850, two pixels short of 480. A hard cutoff at 480 would drop
# all such material and fail the whole task with "no valid materials found". A tiny tolerance is
# kept here that lets through material only slightly below the threshold due to rounding while
# still blocking genuinely low-resolution material.
_MIN_DIMENSION_TOLERANCE = 10
# Clips are processed serially by default so existing users do not see a sudden jump in CPU
# and memory usage after upgrading. Raise the concurrency to speed things up, but note that
# each lane starts its own encoding job.
_CLIP_PROCESSING_CONCURRENCY = 1


def _get_clip_processing_concurrency() -> int:
    try:
        concurrency = int(config.app.get("video_clip_concurrency", _CLIP_PROCESSING_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _CLIP_PROCESSING_CONCURRENCY
    return max(1, min(8, concurrency))
_DEFAULT_VIDEO_CODEC = "libx264"
# There is no stage log while ffmpeg concatenates clips, and `subprocess.run` blocks until the
# process exits, so a long concat looks like "no output" in the logs. Liveness is logged at an
# interval here to tell encoding apart from a hang.
_FFMPEG_CONCAT_HEARTBEAT_SECONDS = 30.0
# MoviePy likewise logs nothing while writing the final video, and a one-minute portrait video
# takes several minutes to encode on CPU. Follow the concat stage and log liveness at an
# interval.
_STAGE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS = 3600
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()
# MoviePy pipes sRGB frames, and ffmpeg's default RGB→YUV matrix is BT.601 with no color tags,
# while players and YouTube decode untagged HD as BT.709 and shift the colors. Convert with the
# BT.709 matrix and tag the stream; setparams writes the tags the -color_* flags alone do not.
# The pixel format stays the encoder's choice: MoviePy leaves odd-sized frames to libx264.
_BT709_VIDEO_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_FFMPEG_PARAMS = ["-vf", _BT709_VIDEO_FILTER]


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """Return the scale factor the subtitle bounce animation uses at the given time."""
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale a subtitle frame or its transparency mask around the center while keeping the canvas
    size unchanged.

    MoviePy stores the subtitle color frame and its transparency mask separately. The bounce
    animation must apply exactly the same scaling and cropping to both, otherwise the first
    animation frame composites the transparent area onto the video as a black text outline. A
    2-D array is a mask with values 0~1; a 3-D array is an RGB/RGBA color frame.
    """
    if frame.ndim not in (2, 3):
        raise ValueError("subtitle frame must be a 2D mask or 3D color frame")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # MoviePy masks use 0~1 floats while Pillow's L mode uses 0~255; convert and then
        # restore the original dtype and range so CompositeVideoClip's transparency semantics
        # stay unchanged.
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """Scale both the subtitle color frame and its mask so the bounce animation has no black first frame."""
    animation_duration = min(
        _SUBTITLE_SPRING_DURATION_SECONDS,
        max(0.0, subtitle_duration),
    )
    if animation_duration <= 0:
        return clip

    def transform_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        scale = _get_subtitle_spring_scale(time_seconds, animation_duration)
        if scale == 1.0:
            return frame
        return _scale_subtitle_frame_on_canvas(frame, scale)

    # apply_to=["mask"] is the key to the fix: MoviePy only transforms the color frame by
    # default, so the old implementation briefly kept the full-size mask as each subtitle
    # appeared and showed a black text outline.
    return clip.transform(transform_frame, apply_to=["mask"])


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Return the target duration for concatenating video material.

    Use case: when composing the video, the material must cover the narration audio. If it only
    "exactly equals" the audio duration, FFmpeg may make the final video slightly shorter due to
    frame-rate rounding, so a small margin is always added. The function is kept separate for
    testing and so the margin can be tuned later based on real feedback.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Decide whether the material resolution is sufficient for composition.

    The nominal minimum is 480x480, but up to `_MIN_DIMENSION_TOLERANCE` pixels below it are
    allowed to accommodate dimensions rounded down by encoders/messaging apps (for example
    WhatsApp's 478x850).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
) -> List[SubClippedVideoClip]:
    """
    Prefer each source material appearing only once, reducing the chance of the same material
    repeating in the final video.

    Online material often comes as "one long video cut into several short segments". The old
    logic shuffled all short segments directly in random mode, so several slices of the same
    source video could land both at the start and in the middle, which users perceive as
    repetition. This function only reorders segments: the longest segment of each source file
    goes first, the remaining segments serve as a fallback; when the total material duration is
    insufficient, later segments are still allowed to fill the audio length so the generation
    success rate is not hurt. Choosing the longest segment first avoids randomly picking tiny
    leftover segments from the end of a video and reusing material too early despite having
    enough of it.
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        if source_usage is None:
            return subclipped_items
        if not source_groups:
            return sorted(
                subclipped_items,
                key=lambda item: source_usage.get(item.source_file_path, 0),
            )
        # Keep keyword rounds in order while rotating candidates within each keyword.
        groups = {}
        for item in subclipped_items:
            key = source_groups.get(item.source_file_path, item.source_file_path)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        return [
            item
            for row in itertools.zip_longest(*groups.values())
            for item in row
            if item is not None
        ]

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    if source_usage is not None:
        # Stable sorting retains randomness among equally used sources.
        primary_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        overflow_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Compatibility for callers that historically read the FFmpeg path from the video service.

    The real resolution logic has moved to `app.utils.utils.get_ffmpeg_binary()`; video, voice
    and any future pipelines should share the same precedence. This thin wrapper is kept so
    external scripts or old tests importing `app.services.video.get_ffmpeg_binary` directly do
    not hit an AttributeError.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Read the user-configured video encoder.

    This setting is for advanced users trying to enable hardware encoding such as
    NVENC/AMF/QSV/VideoToolbox. Only a fixed allowlist is deliberately accepted here, so that
    opening up arbitrary FFmpeg parameters cannot let a typo make the output format
    uncontrollable or make the generation task fail only in a later stage.
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Check whether the current FFmpeg declares support for the given encoder.

    This only proves FFmpeg was built with that encoder, not that the hardware and drivers on
    this machine actually work. So a real encoding failure still falls back to libx264.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Return the video encoder actually used for this run.

    When the user picks a hardware encoder, first check the FFmpeg encoder list; if an actual
    encoding failure already happened in this process, fall back directly so every clip in a
    task does not fail again.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    Retry with libx264 after a hardware encoding failure, and disable the hardware encoder
    only if the retry succeeds.

    FFmpeg failures on Windows have many causes: the GPU/driver may be unsupported, or it may
    be a generic IO problem such as a locked output file, directory permissions or antivirus
    interference. Only when libx264 writes successfully can the original failure be attributed
    to the hardware encoder itself with reasonable confidence, so later tasks are not penalized
    by mistake.
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(
    clip, output_file: str, codec: str, atomic_output: bool = False, **kwargs
):
    """
    Write the video with the given encoder, automatically retrying once with libx264 on
    failure.

    Whether a hardware encoder works depends not only on FFmpeg but also on the GPU, drivers
    and the current runtime environment. A generation task must not fail outright because an
    advanced encoder is unavailable, so the fallback is handled centrally here.
    """
    kwargs.setdefault("ffmpeg_params", _BT709_FFMPEG_PARAMS)
    if atomic_output:
        # Final videos can be downloaded by path while they are being rendered.
        # Keep both failed encodes and in-progress writes away from that path.
        output_dir = os.path.dirname(os.path.abspath(output_file))
        descriptor, temp_output = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_file)}.",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=output_dir,
        )
        os.close(descriptor)
        os.unlink(temp_output)
        try:
            used_codec = _write_videofile_with_codec_fallback(
                clip, temp_output, codec, **kwargs
            )
            os.replace(temp_output, output_file)
            return used_codec
        finally:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    f"failed to remove temporary final video: {temp_output}, "
                    f"error: {exc}"
                )

    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # The concat demuxer wraps paths in single quotes, so single quotes in the path must be
    # escaped first.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Produce the path used in the concat demuxer file list.

    The official FFmpeg docs require special characters and spaces in a concat list to be
    escaped, and backslashes in Windows absolute paths are easily parsed as escape characters.
    Paths are normalized to forward slashes here so `C:\\Users\\...` becomes `C:/Users/...`,
    then single quotes are handled, which also works on macOS/Linux.
    """
    absolute_path = os.path.abspath(file_path)
    # Normalize the native separator only. A backslash can be a literal
    # component of a valid POSIX path and must still identify the same file.
    return _escape_ffmpeg_concat_path(absolute_path.replace(os.sep, "/"))


def _describe_concat_output_progress(output_file: str) -> str:
    """Return a readable description of the output file's current size for the concat heartbeat log."""
    try:
        size = os.path.getsize(output_file)
    except OSError:
        # Degrade safely when the output file does not exist yet; this must not affect the concat
        # itself.
        return "output size not available"
    return f"output size: {size / (1024 * 1024):.2f} MB"


@contextmanager
def _stage_heartbeat(description: str):
    """
    Log liveness at an interval during a long stage that produces no logs of its own.

    The heartbeat thread is bound to the thread entering the stage so its logs show up in the
    owning task's WebUI panel. When the stage ends or raises, it stops and waits for the
    heartbeat thread to exit, so "still running" is never written after the stage is over.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_STAGE_HEARTBEAT_SECONDS):
            logger.info(
                f"{description} still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join(timeout=1)


def _report_clip_progress(
    progress_callback: Callable[[float], None] | None,
    covered_duration: float,
    required_duration: float,
) -> None:
    """Convert the covered output duration into a 0~1 fraction and notify the caller."""
    if progress_callback is None:
        return
    fraction = 1.0
    if required_duration > 0:
        fraction = min(1.0, covered_duration / required_duration)
    try:
        progress_callback(fraction)
    except Exception as exc:
        # Progress is display only. A failing callback (e.g. the state backend is temporarily
        # unavailable) must not invalidate clips that were already processed.
        logger.warning(
            "failed to report clip processing progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _run_concat_with_heartbeat(command: list[str], output_file: str):
    """
    Block until ffmpeg finishes, logging liveness at an interval meanwhile.

    ffmpeg produces no stage log while concatenating clips, and `subprocess.run` buffers output
    until the process exits, so a long concat looks like "no output" in the logs. The elapsed
    wait time and output file size are logged to tell ongoing encoding apart from a hang.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_FFMPEG_CONCAT_HEARTBEAT_SECONDS):
            logger.info(
                "ffmpeg concat still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s, "
                f"{_describe_concat_output_progress(output_file)}"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        configured_timeout = config.app.get(
            "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive") from exc
        if (
            isinstance(configured_timeout, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive")

        try:
            # subprocess.run kills and waits for FFmpeg on timeout, so a stalled
            # encoder cannot leave an orphaned child or a permanently active task.
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ffmpeg concat exceeded {timeout_seconds:g} seconds"
            ) from exc
    finally:
        stop_event.set()


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    # Separate renders may share a directory. Each FFmpeg process must keep its
    # own manifest until all codec attempts finish, without overwriting or
    # deleting another render's list.
    concat_list_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ffmpeg-concat-", suffix=".txt",
            dir=output_dir, delete=False,
        ) as fp:
            concat_list_file = fp.name
            for clip_file in clip_files:
                fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")
    except Exception:
        if concat_list_file:
            delete_files(concat_list_file)
        raise

    staged_output = None

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            # Otherwise FFmpeg reads keys from the console it inherits: "q"
            # ends the video early and "c" waits for Enter.
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-vf",
            _BT709_VIDEO_FILTER,
            "-pix_fmt",
            "yuv420p",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(staged_output)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # Use ffmpeg for a single concat and encode, avoiding the repeated re-encoding of
        # MoviePy's segment-by-segment merge and thus reducing quality loss and color shift.
        # The heartbeat log shows the task is still running during the blocking wait.
        result = _run_concat_with_heartbeat(command, staged_output)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        descriptor, staged_output = tempfile.mkstemp(
            prefix=".ffmpeg-concat-",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=os.path.dirname(os.path.abspath(output_file)),
        )
        os.close(descriptor)
        effective_codec = _get_effective_video_codec()
        try:
            result_codec = run_concat(effective_codec)
        except TimeoutError:
            # A hung encoder is not evidence that another codec will work. Do
            # not spend a second timeout period retrying the same input.
            raise
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
        # Failed attempts and in-progress output stay private until FFmpeg has
        # finished. Publication errors must not trigger another codec attempt.
        if os.path.getsize(staged_output) == 0:
            raise RuntimeError("ffmpeg concat produced no output")
        os.replace(staged_output, output_file)
        return result_codec
    finally:
        delete_files([concat_list_file, staged_output])


def _sanitize_image_file(image_path: str) -> str:
    # Some local images open fine in Pillow but make ImageClip raise during parsing because of
    # corrupted EXIF/eXIf metadata. A "clean image" is re-exported here with the bad metadata
    # stripped.
    # Bound the intermediate basename while retaining complete source identity.
    # Different extensions must not overwrite pixels, and valid long filenames
    # must not exceed the filesystem's component limit after adding a suffix.
    source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))

    temp_path = ""
    try:
        with Image.open(image_path) as image:
            with ImageOps.exif_transpose(image) as upright:
                upright.load()
                # Strip metadata after applying the camera's orientation.
                # Palette transparency belongs to pixels, not removable metadata.
                mode = "RGBA" if "A" in upright.getbands() or "transparency" in upright.info else "RGB"
                if upright.mode in {"I;16", "I;16B", "I;16L"}:
                    # 16-bit grayscale samples span0..65535; an RGB cast either
                    # clips them to white or wraps when MoviePy encodes them.
                    with upright.convert("I").point(lambda value: value / 257 + 0.5) as byte_range:
                        cleaned_image = byte_range.convert(mode)
                else:
                    cleaned_image = upright.convert(mode)
                cleaned_image.info.clear()
                source_digest.update(str(cleaned_image.size).encode())
                source_digest.update(cleaned_image.mode.encode())
                source_digest.update(cleaned_image.tobytes())
                sanitized_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{source_digest.hexdigest()}.sanitized.png",
                )
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=".image-sanitize-", suffix=".png",
                    dir=os.path.dirname(os.path.abspath(sanitized_path)),
                )
                os.close(descriptor)
                cleaned_image.save(temp_path)
                cleaned_image.close()
        os.replace(temp_path, sanitized_path)
        temp_path = ""
    finally:
        if temp_path:
            delete_files(temp_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # MoviePy does not apply camera EXIF orientation while decoding an image.
    # CMYK JPEG channels must become RGB; LA must become RGBA so MoviePy
    # creates a proper opacity mask rather than treating alpha as a color band.
    # Bilevel pixels must become byte-range RGB rather than bool values0/1.
    # Ordinary RGB inputs retain the direct path.
    try:
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            image_mode = image.mode
    except Exception:
        orientation = 1
        image_mode = None
    if orientation in range(2, 9) or image_mode in {"CMYK", "LA", "1", "I;16", "I;16B", "I;16L"}:
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path

    # Try opening the original image first; if that fails due to corrupted metadata, fall back
    # to generating a metadata-free copy.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


_moviepy_reader_open_lock = threading.Lock()


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Open a video file quietly, preventing MoviePy 2.1.x from printing ffmpeg probe info
    straight to stdout.

    Background:
    The currently pinned `FFMPEG_VideoReader` contains `print(self.infos)` and
    `print(ffmpeg command)`, which print `audio_found: False` when reading an intermediate
    video without an audio track. That is only input material metadata and does not mean the
    final video has no audio, but it misleads WebUI/terminal users into thinking generation
    failed.

    Implementation:
    1. Redirect stdout only for the short window in which VideoFileClip is opened;
    2. Default to `audio=False`, because the material stage of this project does not keep the
       material's native audio; the final audio is attached in `generate_video()`;
    3. If the library did print something, downgrade it to a debug log for troubleshooting.
    """
    captured_stdout = io.StringIO()
    # redirect_stdout changes process-wide state. Overlapping reader opens can
    # restore each other's capture buffers instead of the original stdout.
    # Serialize this short construction window; clip processing stays parallel.
    with _moviepy_reader_open_lock:
        with redirect_stdout(captured_stdout):
            clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # When looping to fill the video, the same temporary clip path appears several times in the
    # FFmpeg concat list. The concat must keep the duplicates, but cleanup may delete each only
    # once; deduplicate in the original order here so every caller gets idempotent behavior and
    # no stream of FileNotFoundError follows the first successful delete.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # Cleanup tolerates files that no longer exist, e.g. an FFmpeg failure path or a
            # concurrent cleanup already removed them; this is not something the user must deal
            # with and should not pollute the generation log.
            continue
        except OSError as e:
            # Permissions, a read-only filesystem or disk errors leave real temp files behind;
            # keep the warning so the environment problem can be located from the path and
            # system error.
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # bgm_file in an API request is user input; it may only resolve to the user BGM or
            # built-in song directory, preventing MoviePy from reading arbitrary server files
            # such as config or keys.
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # When the background music directory is empty, fall back to "no BGM" directly so
        # random.choice([]) does not raise.
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Resize a clip to an exact canvas using cover/crop or contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "video dimensions must be positive: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Exact aspect-ratio matches do not need either a crop or a background.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil guarantees the resized clip covers the complete canvas despite
        # floating-point rounding. Any excess is removed symmetrically.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain preserves the legacy behavior: show the complete source frame,
    # centered over a black canvas when the aspect ratios differ.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    # Direct callers and persisted tasks can still supply the raw string.
    # Normalize it once for all downstream enum access.
    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_clip = AudioFileClip(audio_file)
    try:
        # Only the narration audio duration is needed here to decide the concat length of the
        # material video; audio_clip is not used afterwards. Close it right after reading so
        # early returns or error paths do not leak the file handle.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # Handle direct API calls that pass no transition mode, so a later .value access does not
    # crash.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # Log the effective value only once; this helps spot out-of-range API params being
        # normalized and avoids repeating the same log in the per-clip hot path.
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration constrains the final playback duration in the output, not how much of
    # the source video is read. MoviePy playing 1.5s of source at 0.5x yields a 3s clip, and
    # playing 6s of source at 2x also yields a 3s clip. So the source duration must be derived
    # from the speed before slicing; if a fixed 3s were still read, slowed down and trimmed,
    # the next segment would start at second 3 of the source and skip 1.5s of footage. The
    # computation also keeps the source timeline continuous and non-overlapping at any speed.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = None
        try:
            clip = _open_video_clip_quietly(video_path)
            clip_duration = float(clip.duration)
            clip_w, clip_h = clip.size
            if (
                not math.isfinite(clip_duration)
                or clip_duration <= 0
                or not all(
                    math.isfinite(float(dimension)) and float(dimension) > 0
                    for dimension in (clip_w, clip_h)
                )
            ):
                raise ValueError("invalid video duration or dimensions")
        except Exception as exc:
            logger.warning(
                f"skipping unreadable video source: path={video_path}, error={exc}"
            )
            continue
        finally:
            close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # Keep every valid segment.
            # This neither drops material whose whole video is shorter than max_clip_duration
            # nor swallows the small tail left at the end of a long video.
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    def process_one_clip(indexed_item):
        """Trim, speed-adjust and apply the transition to one source segment, then write it out; return None on failure."""
        index, subclipped_item = indexed_item
        source_clip = None
        clip = None
        clip_file = None
        try:
            logger.debug(
                f"processing clip {index + 1}: {subclipped_item.width}x{subclipped_item.height}, "
                f"source: {os.path.basename(subclipped_item.source_file_path)}"
            )
            source_clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = source_clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # Playback speed is a property of the material itself and should be applied before
            # the transition, so one-second transitions such as Fade/Slide do not become 0.5s or
            # 2s with the material speed; the later max-duration trim remains a safety net for
            # float error or odd material durations, keeping the final clip within the
            # configured limit.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normalize every source clip before transitions are applied. In cover mode
            # the clip fills the canvas and the excess edges are cropped; contain keeps
            # the complete source frame and uses black bars for the unused area.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "resizing clip, "
                    f"source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
                    f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            # Short footage and faster playback can leave less than one second.
            # Keep the chosen transition inside the rendered clip's timeline.
            transition_duration = min(1.0, clip.duration)
            if transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, transition_duration),
                    lambda c: video_effects.fadeout_transition(c, transition_duration),
                    lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # Write each candidate clip to a unique temporary file. Threads must not
            # share the same output path.
            # Distinct combinations can share a task/output directory. Reserve
            # an owned path so their encoders and cleanup never share a clip.
            with tempfile.NamedTemporaryFile(
                dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
            ) as temporary_clip:
                clip_file = temporary_clip.name
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            clip_duration_saved = clip.duration
            processed_clip = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=subclipped_item.source_file_path,
            )
            clip_file = None
            return processed_clip
        except Exception as exc:
            logger.error(f"failed to process clip: {str(exc)}")
            return None
        finally:
            # The derived clip shares its FFmpeg reader with the source. If
            # subclipping itself failed, close the original source instead.
            close_clip(clip if clip is not None else source_clip)
            # MoviePy may leave a truncated MP4 even when encoding raises. It
            # was never returned, so concat cleanup cannot see it.
            # Close the reader first so Windows can remove the partial file.
            if clip_file:
                delete_files(clip_file)

    # Clips are always processed in a thread pool. Per-clip logs are the only progress info in
    # this stage; only after binding them to the thread that started the composition can the
    # WebUI task log collect them.
    process_clip_in_task_scope = logging_utils.bind_log_scope(process_one_clip)
    clip_processing_workers = 1
    if len(subclipped_items) >= 2:
        clip_processing_workers = min(_get_clip_processing_concurrency(), len(subclipped_items))
    with ThreadPoolExecutor(
        max_workers=clip_processing_workers,
        thread_name_prefix="clip-process",
    ) as executor:
        next_candidate_index = 0
        while (
            next_candidate_index < len(subclipped_items)
            and video_duration < required_video_duration
        ):
            remaining_duration = required_video_duration - video_duration
            batch = []
            batch_duration = 0.0
            candidate_index = next_candidate_index
            while candidate_index < len(subclipped_items) and batch_duration < remaining_duration:
                subclipped_item = subclipped_items[candidate_index]
                source_duration = subclipped_item.end_time - subclipped_item.start_time
                output_duration = min(
                    max_clip_duration,
                    source_duration / normalized_clip_speed,
                )
                batch.append((candidate_index, subclipped_item))
                batch_duration += output_duration
                candidate_index += 1
            if not batch:
                break
            for processed_clip in executor.map(process_clip_in_task_scope, batch):
                if processed_clip is None:
                    continue
                processed_clips.append(processed_clip)
                video_duration += processed_clip.duration
                # Each 4K clip takes over ten seconds to process; report the covered duration per
                # clip so the log and progress bar show this stage is still moving.
                logger.info(
                    f"processed clip {len(processed_clips)}: "
                    f"{video_duration:.1f} of {required_video_duration:.1f}s covered"
                )
                _report_clip_progress(
                    progress_callback, video_duration, required_video_duration
                )
            next_candidate_index = candidate_index
    
    # loop processed clips until the video duration covers the audio duration and the small safety margin.
    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        if video_paths:
            raise RuntimeError("no readable video clips available for merging")
        logger.warning("no clips available for merging")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    try:
        concat_video_clips_with_ffmpeg(
            clip_files=clip_files,
            output_file=combined_video_path,
            threads=threads,
            output_dir=output_dir,
            max_duration=audio_duration,
        )
        if used_video_paths is not None:
            # Exclude safety-margin clips that FFmpeg trims entirely from the output.
            elapsed = 0.0
            for clip in processed_clips:
                if elapsed >= audio_duration:
                    break
                used_video_paths.append(clip.source_file_path)
                elapsed += clip.duration
    finally:
        # FFmpeg failures and timeouts must not strand one encoded MP4 per clip.
        # Repeated clips share a path; delete_files already deduplicates them.
        delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60, *, shaped=None, fallbacks=None):
    # Subtitle wrapping must happen before the TextClip is actually created, otherwise MoviePy
    # computes the render area from the raw text only. PIL measures width here with the
    # current font and size so each line stays within the usable video width, keeping large
    # font sizes or long Chinese sentences from overflowing the frame.
    # Subtitles containing Arabic letters are shaped and drawn as a whole by khatt (see
    # subtitle_shaping); every line produced by manual wrapping (including English translation
    # lines within it) must be measured with khatt.
    if shaped is None:
        shaped = subtitle_shaping.needs_shaping(text)
    if "\n" in text:
        # Hard breaks in SRT text are separate layout lines. Measuring a token
        # across a newline makes Pillow count both lines as one wide string,
        # then character wrapping can split an otherwise fitting word.
        wrapped_lines = [
            wrap_text(
                line, max_width, font=font, fontsize=fontsize, shaped=shaped, fallbacks=fallbacks
            )
            for line in text.split("\n")
        ]
        return "\n".join(line for line, _ in wrapped_lines), sum(
            height for _, height in wrapped_lines
        )

    # Arabic-letter text must be measured with the shaped glyphs and the font actually used
    # for drawing (the fallback font when glyphs are missing), so wrapping and height match the
    # final subtitle; unshaped Pillow measures the widths of disconnected isolated glyphs. Each
    # manually wrapped line is already its own paragraph (e.g. Arabic followed by a line of
    # English translation), while auto-wrapped lines carry the direction mark of the original
    # paragraph; it is included in measurement too so measuring and rendering share the same
    # direction. Characters the fallback font cannot draw (e.g. embedded Chinese) are drawn by
    # the selected font and counted in measurement as well (see
    # subtitle_shaping.fallback_fonts).
    direction_mark = ""
    if shaped:
        selected_font = font
        font, _ = subtitle_shaping.resolve_font(font, text)
        if fallbacks is None:
            fallbacks = subtitle_shaping.fallback_fonts(font, selected_font)
        if text.strip():
            direction_mark = subtitle_shaping.direction_mark(text)
    shaped_font_path = font
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() returns the visible ink height of the current glyphs, not the font line height.
    # English containing only characters without descenders such as A, m, n lacks the descent,
    # and with multiple lines that error accumulates line by line until the last line of the
    # TextClip is clipped by the canvas. ascent + descent come from the font itself, are
    # independent of language and character mix, and match MoviePy's baseline drawing model.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # Normal TrueType/OpenType fonts never get here; keep a diagnostic log and the font-size
        # fallback so a corrupted or unusual font returning bogus metrics does not produce
        # zero-height subtitles.
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        if shaped:
            width = subtitle_shaping.text_width(
                direction_mark + inner_text, shaped_font_path, fontsize, fallbacks
            )
            return width, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox is still right for measuring the actual width needed for wrapping; height must
        # always use the stable font line height.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # SRT entries allow manual line breaks. Even when the text needs no further wrapping by
        # width, the canvas height must still count the existing lines, otherwise the second and
        # later lines get clipped.
        return direction_mark + text, (text.count("\n") + 1) * line_height

    def attached_text_units(token):
        units = []
        for char in token:
            if units and unicodedata.category(char).startswith("M"):
                units[-1] += char
            else:
                units.append(char)
        return units

    def split_long_token(token):
        # When a single token is itself too wide (common for long Chinese sentences without
        # spaces, or very long English words), degrade to character-level splitting. The key
        # point: when the candidate is too wide, commit the previous still-valid current first,
        # then put the current character on the next line; never push the overflowing
        # character back onto the previous line.
        lines = []
        current = ""
        for char in attached_text_units(token):
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            # The last piece does not fill the line yet, so following words should keep trying
            # to append to it; otherwise every overlong word is followed by an extra line.
            pieces = split_long_token(word)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    # Arabic-letter words cannot be split (it breaks letter joining), so skip the punctuation
    # adjustment below.
    for index in range(1, 1 if shaped else len(lines)):
        # When a long Chinese sentence is split by character, the final period, comma or other
        # closing punctuation may land alone on the next line, stretching the subtitle
        # background and looking like a stray dot below the text. Without redesigning the
        # wrapping algorithm, move the last character of the previous line in front of the
        # punctuation line so the punctuation follows the text; this covers common Chinese and
        # English closing punctuation.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        previous_units = attached_text_units(lines[index - 1])
        if len(previous_units) <= 1:
            continue
        last_unit = previous_units[-1]
        candidate = f"{last_unit}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-len(last_unit)]

    result = "\n".join(
        direction_mark + line.strip() for line in lines if line.strip()
    ).strip()
    # Height is based on the final result. An explicit newline in the original text may survive
    # inside a token, in which case the length of the temporary lines list differs from the
    # number of lines MoviePy actually renders.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Match the Pillow color syntax accepted by subtitle preflight.
    # The background clip sets opacity separately from the color value.
    if isinstance(color, str):
        try:
            return ImageColor.getcolor(color, "RGB")
        except (ValueError, TypeError):
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # The new subtitle background is used only when the user explicitly enables it: a rounded,
    # semi-transparent panel is drawn as an RGBA image and handed to MoviePy as a transparent
    # ImageClip for compositing. The default path stays untouched while a softer subtitle look
    # can be tried out cheaply.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Center the TextClip in the background container by its actually visible pixels.

    MoviePy's TextClip creates a transparent canvas from the font line height and baseline. For
    many fonts the visible glyphs are not at the geometric center of that canvas, so a plain
    `with_position("center")` centers the whole transparent canvas and the subtitle looks
    shifted up or down. The TextClip's transparency mask is read here and the offset computed
    only from the bbox of pixels that are really present, so the text the user sees is
    visually centered in the subtitle background.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def validate_subtitle_colors(params: VideoParams) -> None:
    """Reject colors Pillow cannot render, before generation services run."""
    if not params.subtitle_enabled:
        return
    colors = [("text_fore_color", params.text_fore_color)]
    # Rendering truncates the width to an integer; a zero-width stroke is unused.
    if int(params.stroke_width) > 0:
        colors.append(("stroke_color", params.stroke_color))
    if isinstance(params.text_background_color, str):
        colors.append(("text_background_color", params.text_background_color))
    for field, value in colors:
        if value is None:
            continue
        try:
            ImageColor.getcolor(value, "RGBA")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid subtitle color: {field}") from exc


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Check whether subtitle text and background share a color, to warn that subtitles may be unreadable."""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Check whether the font has the glyphs the sample text needs, caching repeated checks."""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # A failed font probe must not block generation; keep the log for diagnosing
        # environment compatibility issues.
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Check whether the font can draw the letters and digits in the text, ignoring whitespace and punctuation."""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    if subtitle_shaping.needs_shaping(sample):
        # Arabic-letter text is drawn by khatt, which switches to a fallback font when the
        # selected font lacks glyphs: check whether the font actually used for drawing can
        # render every letter and digit.
        return subtitle_shaping.supports_text(font_path, sample)
    return _subtitle_font_supports_sample(font_path, sample)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Compose the final video and return whether background music processing succeeded.

    The return value only describes the BGM state: True when no BGM was requested or it was
    mixed successfully; False when BGM was requested but loading, effects or mixing failed.
    Even when BGM fails, a narration-only video is still produced, leaving the task
    orchestration layer to decide whether to show the user a degradation warning.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        # The API entry point pre-checks this, but the WebUI, CLI and internal calls can reach
        # the render layer directly; always verify by real path that the font stays inside
        # resource/fonts, blocking absolute paths, ../ traversal and symlinks pointing outside
        # the directory, before handing it to PIL/MoviePy.
        font_path = file_security.resolve_path_within_directory(
            utils.font_dir(), params.font_name
        )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # Legacy parameter compatibility: in the API `text_background_color` may be a boolean or
        # an actual color string. Normalize it here so True/False is never passed straight to
        # TextClip with unpredictable rendering results.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    # The font for Arabic-letter subtitles is decided once for all subtitles of the video (see
    # where subtitles are loaded), so individual subtitles in one video do not flip between two
    # fonts.
    shaped_font = font_path

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        shaped = subtitle_shaping.needs_shaping(phrase)
        clip_font = shaped_font if shaped else font_path
        clip_fallbacks = subtitle_shaping.fallback_fonts(shaped_font, font_path) if shaped else []
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # The rounded background is sized to the real text width, so side padding should be
        # modest; the old rectangular background keeps a larger safety margin so long subtitles
        # from legacy configs do not touch the edge or get clipped.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # The subtitle background needs clear horizontal padding around the text. Subtract the
        # padding from the usable width before wrapping, so long English or large font sizes
        # that exactly fill 90% of the video width do not press against the background edge and
        # look clipped. Both the plain rectangular and the rounded background use this logic;
        # subtitles without a background keep the original max width.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=clip_font,
            fontsize=params.font_size,
            fallbacks=clip_fallbacks,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow/MoviePy expand the stroke above and below the glyphs and count it in each
        # line's advance height. Adding stroke padding once around the whole subtitle block
        # still lets the error accumulate per line for thick strokes on multi-line text. Here
        # the two-sided stroke space is counted per actual line: the default thin stroke adds
        # only a little height, while "small font + thick stroke + many lines" still renders in
        # full.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # With `method=label` MoviePy shrinks the text box height automatically and, with
        # multi-line subtitles, strokes or a background color, tends to cut off the lower half of
        # the last line. A more conservative height is passed explicitly here, including line
        # spacing and extra vertical padding, so both the background box and the text render
        # completely.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # The rounded background should hug the text width instead of using 90% of the video
            # width. Measure the longest line with PIL first, then add horizontal padding, so
            # short subtitles do not get an overly wide panel.
            try:
                if shaped:
                    # Arabic-letter text is measured with the shaped glyphs and the font actually
                    # used for drawing.
                    text_w = max(
                        subtitle_shaping.text_width(
                            line, clip_font, params.font_size, clip_fallbacks
                        )
                        for line in wrapped_txt.split("\n")
                    )
                else:
                    font = ImageFont.truetype(font_path, params.font_size)
                    text_w = max(
                        int(font.getbbox(line)[2] - font.getbbox(line)[0])
                        for line in wrapped_txt.split("\n")
                    )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            if shaped:
                # khatt sizes the canvas to hold all ink (e.g. Quranic pause marks rising above
                # the font's ascender); fixing the height by the formula above could clip them,
                # so measure the natural height first.
                with subtitle_shaping.text_clip(
                    TextClip,
                    shaped_font=shaped_font,
                    text=wrapped_txt,
                    font=font_path,
                    font_size=params.font_size,
                    stroke_width=params.stroke_width,
                    interline=interline,
                    size=(size[0], None),
                    text_align="center",
                ) as natural:
                    size = (size[0], max(size[1], natural.h))
            _clip = subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)

        # The bounce animation is enabled only when the user explicitly selects it; the default
        # none keeps the original subtitle render path entirely.
        anim_type = getattr(params, "subtitle_animation", "none")
        if anim_type in ("pop_spring", "spring", "pop"):
            _clip = _apply_subtitle_spring_animation(_clip, duration)

        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            # 2/3 from the bottom = 1/3 from the top: y = (video_height - _clip.h) * (1/3)
            y_two_thirds = (video_height - _clip.h) / 3.0
            _clip = _clip.with_position(("center", y_two_thirds))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # MoviePy's CompositeAudioClip.close() does not close child AudioFileClips. An ExitStack
    # explicitly holds every raw file reader here so the FFmpeg subprocesses are released on
    # success, subtitle errors, mixing failures and video write failures alike, which above all
    # avoids locked files on Windows.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(1.0 if params.voice_volume is None else params.voice_volume)]
        )

        def make_textclip(text):
            return subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            subtitles = moviepy_subtitles(subtitle_path)
            shaped_font = subtitle_shaping.video_font(
                font_path, (text for _, text in subtitles)
            )
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitles,
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # Every BGM source shares this short-circuit rule. With volume not above 0, neither
            # random/custom files nor provider-returned files may be resolved or loaded, avoiding
            # pointless IO and mixing.
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # Provider music can be passed in directly as a file by the task orchestration layer.
        # None means fall back to random/custom BGM resolution, and an empty string explicitly
        # disables BGM for this run; but every source must first pass the common volume rule.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [afx.MultiplyVolume(params.bgm_volume)]
                # Random/custom music resolved inside the service may be shorter than the output
                # and needs looping to fill it; a file passed by the task layer via override means
                # the provider already fitted the duration. Looping is decided by the file's
                # origin here, so adding a provider later does not require editing a name
                # allowlist.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                # Fade the complete playback timeline, not each source loop.
                bgm_effects.append(afx.AudioFadeOut(3))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # Log the full traceback with stable context to tell file decoding, MoviePy effect
                # and CompositeAudioClip failures apart; file contents and API keys never reach
                # the log.
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # Explicitly reuse the input audio sample rate; if unavailable, fall back to MoviePy's
        # default 44100Hz. This reduces quality fluctuations from resampling again in different
        # environments, especially Docker.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        with _stage_heartbeat("final video render"):
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=output_file,
                codec=_get_configured_video_codec(),
                atomic_output=True,
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
            )
        return bgm_mix_succeeded


# Zoom mode now renders with sub-pixel interpolation, so the output pixels differ from older
# versions. The file name carries this tag so jittery cached clips scaled to integer sizes by
# older versions are never reused.
_ZOOM_CACHE_TAG = "-smooth"


def _smooth_zoom_clip(clip, zoom_amount: float):
    """
    Produce a slowly zooming clip with sub-pixel precision while keeping the canvas size.

    The old implementation `clip.resized(lambda t: ...)` scaled the image to an integer size
    every frame and then center-cropped by an integer offset. At a very slow zoom (about 3% per
    second) adjacent frames either keep the same size or jump by 1px, and the center offset
    wobbles with rounding, which shows up as jitter/stutter in the output and is more visible
    when upscaled to 1080x1920.

    Here the output size is fixed and each frame is an affine transform about the image center
    (BICUBIC interpolation) with a scale that changes continuously over time. The composition
    matches the old implementation: start from the original image, zoom about the center and
    crop whatever exceeds the canvas. Images with an alpha channel get the same treatment for
    their mask.
    """
    base = Image.fromarray(np.asarray(clip.get_frame(0)).astype("uint8"))
    width, height = base.size
    center_x, center_y = width / 2.0, height / 2.0
    duration = clip.duration or 1.0
    mask_base = None
    if clip.mask is not None:
        mask_frame = np.asarray(clip.mask.get_frame(0)) * 255
        mask_base = Image.fromarray(mask_frame.clip(0, 255).astype("uint8"))

    def warp(image, t, resample):
        progress = min(max(t, 0.0), duration) / duration
        scale = 1.0 + zoom_amount * progress
        # Output pixel (x, y) samples the source at (cx + (x - cx) / s, cy + (y - cy) / s).
        coefficients = (
            1 / scale, 0, center_x - center_x / scale,
            0, 1 / scale, center_y - center_y / scale,
        )
        return image.transform(
            (width, height), Image.AFFINE, coefficients, resample=resample
        )

    zoomed = VideoClip(
        lambda t: np.asarray(warp(base, t, Image.BICUBIC)), duration=clip.duration
    )
    if mask_base is not None:
        mask = VideoClip(
            lambda t: np.asarray(warp(mask_base, t, Image.BILINEAR)) / 255.0,
            is_mask=True,
            duration=clip.duration,
        )
        zoomed = zoomed.with_mask(mask)
    return zoomed.with_position("center")


def render_image_zoom_video(
    image_path: str, clip_duration: int = 5, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Render a single image into a static or slowly zooming mp4 clip and return the output path.

    Local material preprocessing and OpenAI-compatible text-to-image material share this
    "image -> clip" rendering logic: by default the roughly 3%-per-second zoom is kept, and
    static mode only fixes the playback duration. This setting is independent of clip
    transitions. Rendering errors are handled by the caller per its material source's failure
    contract.
    """
    motion = ImageMotion(image_motion)
    clip, _ = _open_image_clip_with_fallback(image_path)
    clip = clip.with_duration(clip_duration).with_position("center")
    temp_path = ""
    try:
        # Zoom mode enlarges frame by frame with sub-pixel precision; static mode keeps the
        # original composition and transparent background handling.
        image_clip = (
            _smooth_zoom_clip(clip, clip_duration * 0.03)
            if motion == ImageMotion.zoom
            else clip
        )

        # Optionally, create a composite video clip containing the zoomed clip.
        # This is useful if you want to add other elements to the video.
        final_clip = CompositeVideoClip([image_clip], bg_color=(0, 0, 0))
        try:
            # The duration changes the rendered content, so it must be part of
            # the output identity. Different tasks may render the same image
            # concurrently; only publish a complete MP4 after MoviePy closes it.
            source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))
            source_frame = clip.get_frame(0)
            source_digest.update(str(source_frame.shape).encode())
            source_digest.update(source_frame.tobytes())
            if clip.mask is not None:
                source_digest.update(clip.mask.get_frame(0).tobytes())
            source_identity = source_digest.hexdigest()
            duration_identity = hashlib.sha256(str(clip_duration).encode()).hexdigest()[:16]
            motion_tag = _ZOOM_CACHE_TAG if motion == ImageMotion.zoom else ""
            video_file = os.path.join(
                os.path.dirname(image_path),
                f"{source_identity}.{motion.value}{motion_tag}-{duration_identity}.mp4",
            )
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".image-zoom-",
                suffix=".mp4",
                dir=os.path.dirname(os.path.abspath(video_file)),
            )
            os.close(descriptor)
            final_clip.write_videofile(
                temp_path, fps=30, logger=None, ffmpeg_params=_BT709_FFMPEG_PARAMS
            )
        finally:
            close_clip(final_clip)
        os.replace(temp_path, video_file)
        temp_path = ""
        return video_file
    finally:
        close_clip(clip)
        if temp_path:
            delete_files(temp_path)


def preprocess_video(
    materials: List[MaterialInfo], clip_duration=4, image_motion: ImageMotion = ImageMotion.zoom
):
    # The WebUI may pass an empty material list in some regeneration scenarios; return an empty
    # result directly here to avoid a NoneType exception.
    if not materials:
        return []

    # Return only materials that pass preprocessing validation, so low-resolution images do not
    # continue into the video composition pipeline.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # Material paths for the local video_source come from API params and must be
            # confined to the dedicated material directory. Users may pass a file name, and
            # absolute paths returned historically are still accepted, but escaping to other
            # system directories is not allowed, preventing arbitrary file reads or probing
            # sensitive local files via MoviePy.
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        is_image = ext in const.FILE_TYPE_IMAGES
        try:
            # Read image material as an image directly, avoiding a VideoFileClip misdetection
            # that triggers the unstable fallback branch.
            if is_image:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # Fall back to image mode on a non-standard extension or failed probe, for
            # compatibility with local image paths passed directly in the past.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
                # The successful decoder determines the material kind, even
                # when the uploaded filename has a video or unknown suffix.
                is_image = True
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # Release the resource right after detecting low-resolution material, and do not
                # return that material to later stages.
                close_clip(clip)
                continue

            if is_image:
                logger.info(f"processing image: {material_source_path}")
                # The material was already opened once to probe its size; release that probe
                # handle first, then render the image clip used for export.
                close_clip(clip)
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration, image_motion=image_motion
                )
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # Ordinary video material only needs its size read for validation; release the
                # handle as soon as validation is done.
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
