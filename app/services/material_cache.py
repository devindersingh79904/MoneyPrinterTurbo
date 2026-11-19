"""Disk cache for online material search results."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils


MATERIAL_SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_FORMAT_VERSION = 2
_CACHE_CLEANUP_INTERVAL_SECONDS = 60 * 60
_CACHE_FILE_PATTERN = re.compile(r"^[0-9a-f]{64}\.json$")
# Saving the cache writes a NamedTemporaryFile(delete=False) first and then publishes it
# with os.replace. When the process is killed (Ctrl+C, container stop, power loss) the
# Python exception handlers never run, leaving an intermediate file that was never
# replaced; it does not match the cache file pattern above and must be recognized
# separately for cleanup, or it accumulates in the cache directory forever.
_CACHE_TEMP_FILE_PATTERN = re.compile(r"^\.[0-9a-f]{64}-[a-z0-9_]+\.tmp$")

# The API allows multiple video tasks to run concurrently by default. A fixed number of
# lock shards lets identical search criteria share one lock while avoiding unbounded
# memory growth from keeping a Lock per keyword forever. It only coalesces concurrent
# requests within this process; cross-process writes still rely on the temp file and
# os.replace for integrity.
_CACHE_LOCKS = tuple(threading.Lock() for _ in range(256))
_cleanup_state_lock = threading.Lock()
_last_cleanup_monotonic: float | None = None


def _safe_public_url(value) -> str | None:
    """Strip query parameters and user credentials from a public page URL so the cache never stores a token by accident."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _cached_source_info(item: MaterialInfo) -> dict | None:
    """
    Build the source info that may be written to disk, using an allowlist.

    The search keyword is already part of the cache key and is not written into the cache
    content in plain text; it is restored from the call arguments on read. The download
    URL is stored separately in ``MaterialInfo.url``; only the public material page, the
    author's public page and stable business identifiers are allowed here, so arbitrary
    extra fields never reach the disk cache.
    """
    source = item.source_info
    if not isinstance(source, dict) or not source:
        return None

    cached: dict = {
        "provider": str(source.get("provider") or item.provider),
    }
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if asset_id not in (None, ""):
        cached["asset_id"] = str(asset_id)
    if source_page:
        cached["source_page"] = source_page

    raw_creator = source.get("creator")
    if isinstance(raw_creator, dict):
        creator = {}
        creator_id = raw_creator.get("id")
        creator_name = raw_creator.get("name")
        creator_page = _safe_public_url(raw_creator.get("profile_page"))
        if creator_id not in (None, ""):
            creator["id"] = str(creator_id)
        if creator_name not in (None, ""):
            creator["name"] = str(creator_name)
        if creator_page:
            creator["profile_page"] = creator_page
        if creator:
            cached["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            cached["rendition"] = rendition
    return cached


def _cache_dir() -> Path:
    """
    Return the material search cache directory shared by every entry point.

    The cache must live under ``storage`` rather than in the WebUI session or process
    memory so the WebUI, API, CLI and tasks after a Docker restart all reuse the same
    results.
    """
    return Path(utils.storage_dir("cache_material_search", create=True))


def _cache_key(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> str:
    """
    Build a stable file name from the business parameters that affect search results.

    The API key only authenticates and does not change public search results, so it must
    not go into the cache key or content. SHA-256 keeps the keyword out of the file name
    while keeping the path length fixed.
    """
    aspect_value = getattr(video_aspect, "value", video_aspect)
    cache_key = json.dumps(
        {
            "provider": str(provider).strip().lower(),
            "search_term": str(search_term).strip(),
            "minimum_duration": int(minimum_duration),
            "video_aspect": str(aspect_value),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()


def _cache_path(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> Path:
    digest = _cache_key(
        provider=provider,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )
    return _cache_dir() / f"{digest}.json"


def get_material_search_cache_lock(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> threading.Lock:
    """Return the in-process lock shard for the current search criteria."""
    digest = _cache_key(
        provider=provider,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )
    return _CACHE_LOCKS[int(digest[:8], 16) % len(_CACHE_LOCKS)]


def _is_same_cache_file(cache_path: Path, expected: os.stat_result) -> bool:
    """Avoid removing a file replaced since the scan/read snapshot."""
    try:
        current = cache_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and (
        current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size
    ) == (expected.st_dev, expected.st_ino, expected.st_mtime_ns, expected.st_size)


def _remove_invalid_cache(
    cache_path: Path, expected: os.stat_result | None = None
) -> None:
    """Delete a single expired or unparsable cache file; a failure does not affect the main material search flow."""
    try:
        if expected is not None and not _is_same_cache_file(cache_path, expected):
            return
        cache_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            f"failed to remove invalid material search cache: "
            f"file={cache_path.name}, error={exc}"
        )


def load_material_search_cache(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
    *,
    now: float | None = None,
) -> list[MaterialInfo] | None:
    """
    Read material search results still within the 24-hour validity window.

    ``None`` means a cache miss and the remote API must be called; an empty list is never
    returned as a valid cache entry, so a network error or upstream failure cached by
    mistake cannot keep blocking later tasks.
    """
    if str(provider).strip().lower() == "coverr":
        # Coverr download URLs contain a signed JWT bound to the API key. It is only for the
        # current request and must not enter the disk cache; when the same criteria are
        # queried, also delete any cache an older version may have left behind.
        try:
            _remove_invalid_cache(
                _cache_path(
                    provider=provider,
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            )
        except Exception as exc:
            logger.warning(
                "failed to remove disabled Coverr material search cache: "
                f"error={type(exc).__name__}, detail={exc}"
            )
        return None

    try:
        cache_path = _cache_path(
            provider=provider,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
    except Exception as exc:
        # Errors creating the cache directory or resolving paths must not block the remote
        # material search. Keep the full exception type and message to diagnose permission or
        # mount problems, and continue the main flow as a cache miss.
        logger.warning(
            "failed to prepare material search cache: "
            f"operation=read, error={type(exc).__name__}, detail={exc}"
        )
        return None
    try:
        stat_result = cache_path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning(
            f"failed to inspect material search cache: "
            f"file={cache_path.name}, error={exc}"
        )
        return None

    current_time = time.time() if now is None else now
    cache_age = current_time - stat_result.st_mtime
    # After a clock rollback or copying the file from another machine, mtime may lie in
    # the future. The cache must not then count as fresh indefinitely; invalidating it and
    # querying the remote again is more reliable.
    if cache_age < 0 or cache_age >= MATERIAL_SEARCH_CACHE_TTL_SECONDS:
        _remove_invalid_cache(cache_path, expected=stat_result)
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)

        if (
            not isinstance(payload, dict)
            or payload.get("version") != _CACHE_FORMAT_VERSION
            or not isinstance(payload.get("items"), list)
            or not payload["items"]
        ):
            raise ValueError("invalid cache payload")

        items = []
        for raw_item in payload["items"]:
            if not isinstance(raw_item, dict):
                raise ValueError("invalid material item")
            item_provider = raw_item.get("provider")
            item_url = raw_item.get("url")
            item_duration = raw_item.get("duration")
            source_info = raw_item.get("source_info")
            if (
                not isinstance(item_provider, str)
                or not item_provider
                or not isinstance(item_url, str)
                or not item_url
                or isinstance(item_duration, bool)
                or not isinstance(item_duration, (int, float))
                or (isinstance(item_duration, float) and not math.isfinite(item_duration))
                or item_duration < 1
                or not isinstance(source_info, dict)
                or not source_info
            ):
                raise ValueError("invalid material fields")
            source_info = dict(source_info)
            source_info["search_term"] = search_term
            items.append(
                MaterialInfo(
                    provider=item_provider,
                    url=item_url,
                    duration=int(item_duration),
                    source_info=source_info,
                )
            )
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            f"failed to load material search cache: file={cache_path.name}, error={exc}"
        )
        _remove_invalid_cache(cache_path, expected=stat_result)
        return None

    logger.info(
        f"material search cache hit: provider={provider}, "
        f"term={search_term!r}, items={len(items)}"
    )
    return items


def save_material_search_cache(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
    items: Iterable[MaterialInfo],
) -> bool:
    """
    Atomically save one successful, non-empty material search result.

    Several tasks may search the same keyword concurrently. Writing to a unique temp file
    in the same directory and publishing it with ``os.replace`` guarantees readers only
    ever see the complete old file or the complete new file; even if two writers finish
    at once, the final content is a valid result for the same cache key either way.
    """
    if str(provider).strip().lower() == "coverr":
        return False

    temp_path = None
    try:
        serialized_items = []
        for item in items:
            source_info = _cached_source_info(item)
            if not item.url or item.duration <= 0 or not source_info:
                continue
            serialized_items.append(
                {
                    "provider": item.provider,
                    "url": item.url,
                    "duration": int(item.duration),
                    "source_info": source_info,
                }
            )
        if not serialized_items:
            return False

        cache_path = _cache_path(
            provider=provider,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        cleanup_expired_material_search_cache()
        payload = {
            "version": _CACHE_FORMAT_VERSION,
            "items": serialized_items,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, cache_path)
        return True
    except Exception as exc:
        logger.warning(
            "failed to save material search cache: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def cleanup_expired_material_search_cache(
    *,
    now: float | None = None,
    force: bool = False,
) -> int:
    """
    Occasionally clean up expired search caches that were never queried again, plus temp
    files left behind by interrupted writes.

    The normal write path scans the directory at most once per hour so each search does
    not cause a linear directory walk; ``force`` is for tests or explicit maintenance
    only. Only SHA-256-named JSON cache files and the ``.tmp`` intermediate files this
    module creates are deleted; other files users place in the directory are untouched.
    Both use the same expiry test, so a temp file still being written is never removed
    by a concurrent cleanup.
    """
    global _last_cleanup_monotonic

    monotonic_now = time.monotonic()
    with _cleanup_state_lock:
        if (
            not force
            and _last_cleanup_monotonic is not None
            and monotonic_now - _last_cleanup_monotonic
            < _CACHE_CLEANUP_INTERVAL_SECONDS
        ):
            return 0
        _last_cleanup_monotonic = monotonic_now

    try:
        cache_dir = _cache_dir()
        entries = os.scandir(cache_dir)
    except Exception as exc:
        logger.warning(
            "failed to scan material search cache: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return 0

    current_time = time.time() if now is None else now
    deleted_count = 0
    failed_count = 0
    with entries:
        for entry in entries:
            if not (
                _CACHE_FILE_PATTERN.fullmatch(entry.name)
                or _CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
            ):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                # DirEntry.stat() reports st_dev/st_ino as 0 on Windows, which
                # would never match the identity check in _is_same_cache_file().
                expected = os.stat(entry.path, follow_symlinks=False)
                cache_age = current_time - expected.st_mtime
                if _CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name):
                    # An active write may look future-dated after a clock
                    # rollback. Only remove temp files demonstrably old enough.
                    if cache_age < MATERIAL_SEARCH_CACHE_TTL_SECONDS:
                        continue
                elif 0 <= cache_age < MATERIAL_SEARCH_CACHE_TTL_SECONDS:
                    continue
                if not _is_same_cache_file(Path(entry.path), expected):
                    continue
                os.unlink(entry.path)
                deleted_count += 1
            except OSError as exc:
                failed_count += 1
                logger.warning(
                    "failed to delete material search cache file: "
                    f"file={entry.name}, error={exc}"
                )

    if deleted_count or failed_count:
        logger.info(
            "finished cleaning material search cache: "
            f"deleted={deleted_count}, failed={failed_count}"
        )
    return deleted_count
