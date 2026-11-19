import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.models.schema import MaterialInfo, VideoAspect
from app.services import material, material_cache


class TestMaterialSearchCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir_patch = patch(
            "app.services.material_cache.utils.storage_dir",
            return_value=self.temp_dir.name,
        )
        self.cache_dir_patch.start()

    def tearDown(self):
        self.cache_dir_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _item(url: str = "https://example.com/video.mp4") -> MaterialInfo:
        return MaterialInfo(
            provider="pixabay",
            url=url,
            duration=12,
            source_info={
                "provider": "pixabay",
                "search_term": "nature",
                "asset_id": "123",
                "source_page": "https://pixabay.com/videos/example-123/",
                "creator": {
                    "id": "456",
                    "name": "Creator",
                    "profile_page": "https://pixabay.com/users/creator-456/",
                },
                "rendition": {
                    "id": "large",
                    "width": 1080,
                    "height": 1920,
                },
            },
        )

    def _cache_path(self) -> Path:
        return material_cache._cache_path(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

    def test_cache_round_trip_preserves_material_fields(self):
        """
        The disk cache must restore every field MaterialInfo needs across processes; caching
        only the URL and losing provider or duration would change later download and
        duration calculation behavior.
        """
        saved = material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertTrue(saved)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].provider, "pixabay")
        self.assertEqual(loaded[0].url, "https://example.com/video.mp4")
        self.assertEqual(loaded[0].duration, 12)
        self.assertEqual(loaded[0].source_info["search_term"], "nature")
        self.assertEqual(loaded[0].source_info["asset_id"], "123")
        self.assertEqual(
            loaded[0].source_info["source_page"],
            "https://pixabay.com/videos/example-123/",
        )
        self.assertEqual(
            loaded[0].source_info["creator"]["profile_page"],
            "https://pixabay.com/users/creator-456/",
        )

    def test_expired_cache_is_removed_and_treated_as_miss(self):
        """
        Pixabay allows search results to be reused for at most 24 hours. Expired files must
        be invalidated and deleted immediately so old material URLs are not reused forever
        and the cache directory does not keep accumulating stale JSON.
        """
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        cache_path = self._cache_path()
        now = 2_000_000_000.0
        expired_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(cache_path, (expired_mtime, expired_mtime))

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            now=now,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_future_dated_cache_is_removed_and_treated_as_miss(self):
        """A broken system clock must not let a future timestamp bypass the 24-hour validity."""
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        cache_path = self._cache_path()
        now = 2_000_000_000.0
        future_mtime = now + 60
        os.utime(cache_path, (future_mtime, future_mtime))

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            now=now,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_corrupted_cache_is_removed_without_breaking_search(self):
        """
        A process crash, disk failure or manual edit can leave a corrupt file. A read failure
        must fall back to the remote search and clean up the bad file; one cache entry must
        never block material generation permanently.
        """
        cache_path = self._cache_path()
        cache_path.write_text("{invalid-json", encoding="utf-8")

        with patch("app.services.material_cache.logger.warning") as warning:
            loaded = material_cache.load_material_search_cache(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
                # Keep freshness deterministic despite filesystem/clock precision.
                now=cache_path.stat().st_mtime + 1,
            )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())
        self.assertTrue(warning.called)

    def test_empty_results_are_not_cached(self):
        """
        The current provider interface uses [] for both "no results" and "request failed".
        Caching an empty list would freeze a Cloudflare block or a transient network failure
        for 24 hours, so only non-empty successful results may be cached.
        """
        saved = material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[],
        )

        self.assertFalse(saved)
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_cache_file_does_not_contain_search_parameters_or_credentials(self):
        """
        The cache file name uses a digest and the content holds only material fields. Even if
        the user shares the storage directory, no keyword, API key or other request config
        may appear in the file.
        """
        item = self._item()
        item.source_info["source_page"] += "?token=drop"
        item.source_info["creator"]["profile_page"] += "?key=drop"
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="private search term",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[item],
        )
        cache_files = list(Path(self.temp_dir.name).glob("*.json"))

        self.assertEqual(len(cache_files), 1)
        self.assertNotIn("private search term", cache_files[0].name)
        raw_payload = cache_files[0].read_text(encoding="utf-8")
        payload = json.loads(raw_payload)
        self.assertEqual(set(payload), {"version", "items"})
        self.assertNotIn("private search term", raw_payload)
        self.assertNotIn("token=drop", raw_payload)

    def test_coverr_signed_urls_are_never_cached(self):
        """
        Coverr download URLs contain a signed JWT and must not enter the long-lived disk cache.
        """
        item = self._item(
            "https://storage.coverr.co/video/download?token=signed-jwt"
        )
        item.provider = "coverr"
        item.source_info["provider"] = "coverr"

        saved = material_cache.save_material_search_cache(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[item],
        )

        self.assertFalse(saved)
        self.assertEqual(list(Path(self.temp_dir.name).glob("*.json")), [])

    def test_coverr_cache_load_removes_legacy_signed_url(self):
        """
        Accessing Coverr must clean up signed download URL caches an older version may have
        left.
        """
        cache_path = material_cache._cache_path(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "coverr",
                            "url": "https://storage.coverr.co/video?token=signed-jwt",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = material_cache.load_material_search_cache(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_version_one_cache_is_invalidated(self):
        """
        An old cache without source info must be re-queried after upgrade instead of producing
        incomplete task records.
        """
        cache_path = self._cache_path()
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "pixabay",
                            "url": "https://example.com/old.mp4",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_cache_key_separates_provider_duration_and_aspect(self):
        """
        Material source, minimum duration and aspect ratio all change the remote search
        results; any change in these parameters must use a separate cache so material that
        does not fit the current task is never returned to video generation.
        """
        base_path = material_cache._cache_path(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        paths = {
            base_path,
            material_cache._cache_path(
                provider="pexels",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            ),
            material_cache._cache_path(
                provider="pixabay",
                search_term="nature",
                minimum_duration=10,
                video_aspect=VideoAspect.portrait,
            ),
            material_cache._cache_path(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.landscape,
            ),
        }

        self.assertEqual(len(paths), 4)

    def test_search_wrapper_reuses_cache_across_calls(self):
        """
        The first call searches remotely and writes the cache; a second call with the same
        parameters must reuse the disk result directly. This is the core behavior that
        reduces Pixabay API calls and the chance of triggering Cloudflare bot protection.
        """
        remote_search = Mock(return_value=[self._item()])

        first = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        second = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(remote_search.call_count, 1)
        self.assertEqual(first, second)

    def test_search_wrapper_refreshes_mixed_orientation_cache(self):
        """
        A pre-upgrade cache may contain material of other orientations. Returning only the
        few filtered entries would reduce material diversity, so any orientation mismatch
        must trigger a new request that replaces the whole candidate set.
        """
        portrait_item = self._item("https://example.com/old-portrait.mp4")
        landscape_item = self._item("https://example.com/old-landscape.mp4")
        landscape_item.source_info["rendition"] = {
            "id": "large",
            "width": 1920,
            "height": 1080,
        }
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[portrait_item, landscape_item],
        )

        refreshed_item = self._item("https://example.com/refreshed-portrait.mp4")
        remote_search = Mock(return_value=[refreshed_item])
        results = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(remote_search.call_count, 1)
        self.assertEqual(
            [item.url for item in results],
            ["https://example.com/refreshed-portrait.mp4"],
        )
        cached_items = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        self.assertEqual(
            [item.url for item in cached_items],
            ["https://example.com/refreshed-portrait.mp4"],
        )

    def test_square_search_reuses_crop_compatible_cache(self):
        """
        Square tasks must keep reusing the croppable material cache instead of re-requesting
        because the original orientation differs.
        """
        landscape_item = self._item("https://example.com/landscape.mp4")
        landscape_item.source_info["rendition"] = {
            "id": "large",
            "width": 1920,
            "height": 1080,
        }
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.square,
            items=[landscape_item],
        )
        remote_search = Mock(return_value=[])

        results = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.square,
        )

        self.assertEqual(remote_search.call_count, 0)
        self.assertEqual(
            [item.url for item in results],
            ["https://example.com/landscape.mp4"],
        )

    def test_search_wrapper_retries_after_empty_result(self):
        """
        Empty results are not cached; the next call must hit the remote again so a transient
        failure retries automatically.
        """
        remote_search = Mock(return_value=[])

        for _ in range(2):
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )
            self.assertEqual(results, [])

        self.assertEqual(remote_search.call_count, 2)

    def test_cache_read_failure_falls_back_to_remote_search(self):
        """
        A cache read error may only degrade to a miss and must not block the remote material
        search.
        """
        remote_items = [self._item()]
        remote_search = Mock(return_value=remote_items)

        with patch.object(
            material_cache,
            "load_material_search_cache",
            side_effect=RuntimeError("cache read failed"),
        ), patch.object(material_cache.logger, "warning") as warning:
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, remote_items)
        self.assertEqual(remote_search.call_count, 1)
        self.assertTrue(warning.called)

    def test_cache_write_failure_keeps_remote_results(self):
        """
        After a successful remote search, usable material must still be returned even if the
        cache write fails.
        """
        remote_items = [self._item()]
        remote_search = Mock(return_value=remote_items)

        with patch.object(
            material_cache,
            "load_material_search_cache",
            return_value=None,
        ), patch.object(
            material_cache,
            "save_material_search_cache",
            side_effect=RuntimeError("cache write failed"),
        ), patch.object(material_cache.logger, "warning") as warning:
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, remote_items)
        self.assertEqual(remote_search.call_count, 1)
        self.assertTrue(warning.called)

    def test_invalid_cache_item_does_not_raise(self):
        """
        A malformed material object must not let the optional cache write break the caller's
        main flow.
        """
        with patch.object(material_cache.logger, "warning") as warning:
            saved = material_cache.save_material_search_cache(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
                items=[None],
            )

        self.assertFalse(saved)
        self.assertTrue(warning.called)

    def test_concurrent_identical_searches_share_remote_request(self):
        """
        The API service allows concurrent tasks. On the first search with identical
        conditions, later threads must wait for the first thread to write the cache instead
        of spending third-party API quota again.
        """
        remote_started = threading.Event()
        allow_remote_finish = threading.Event()
        remote_call_lock = threading.Lock()
        remote_call_count = 0
        results = []

        def remote_search(**_kwargs):
            nonlocal remote_call_count
            with remote_call_lock:
                remote_call_count += 1
            remote_started.set()
            self.assertTrue(allow_remote_finish.wait(timeout=2))
            return [self._item()]

        def run_search():
            results.append(
                material._search_videos_with_cache(
                    provider="pixabay",
                    search_videos=remote_search,
                    search_term="shared nature",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
            )

        first_thread = threading.Thread(target=run_search)
        second_thread = threading.Thread(target=run_search)
        first_thread.start()
        self.assertTrue(remote_started.wait(timeout=2))
        second_thread.start()
        # Give the second thread time to enter the cache lock wait so the test covers a real
        # concurrent miss.
        time.sleep(0.05)
        allow_remote_finish.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(remote_call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])

    def test_cleanup_removes_expired_entries_only(self):
        """
        The low-frequency cleanup deletes only expired cache entries and must not touch valid
        entries or the user's other files.
        """
        stale_path = self._cache_path()
        stale_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "pixabay",
                            "url": "https://example.com/stale.mp4",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        fresh_path = material_cache._cache_path(
            provider="pexels",
            search_term="fresh",
            minimum_duration=5,
            video_aspect=VideoAspect.landscape,
        )
        fresh_path.write_text("{}", encoding="utf-8")
        unrelated_path = Path(self.temp_dir.name) / "notes.json"
        unrelated_path.write_text("keep", encoding="utf-8")

        now = 2_000_000_000.0
        stale_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(stale_path, (stale_mtime, stale_mtime))
        os.utime(fresh_path, (now - 60, now - 60))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(stale_path.exists())
        self.assertTrue(fresh_path.exists())
        self.assertTrue(unrelated_path.exists())

    def test_cleanup_removes_orphaned_temp_files(self):
        """
        When a process is killed, the intermediate file left by NamedTemporaryFile(delete=False)
        never gets consumed by os.replace and the exception-path cleanup never runs. It must be
        reclaimed together with expired cache entries, otherwise every abnormal interruption
        leaves one more permanent leftover file in the cache directory.
        """
        orphan_path = Path(self.temp_dir.name) / (
            f".{self._cache_path().stem}-m429jzwe.tmp"
        )
        orphan_path.write_text('{"version":2,"items":[]}', encoding="utf-8")
        unrelated_path = Path(self.temp_dir.name) / "notes.tmp"
        unrelated_path.write_text("keep", encoding="utf-8")

        now = 2_000_000_000.0
        stale_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(orphan_path, (stale_mtime, stale_mtime))
        os.utime(unrelated_path, (stale_mtime, stale_mtime))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(orphan_path.exists())
        # Other files whose prefix does not match belong to the user and must not be deleted by
        # the cleanup.
        self.assertTrue(unrelated_path.exists())

    def test_cleanup_keeps_recent_temp_files(self):
        """
        During concurrent searches another process may be writing a temporary file. Cleanup
        must reuse the cache expiry rule and leave not-yet-expired intermediate files alone,
        so a file waiting for os.replace is not deleted.
        """
        in_flight_path = Path(self.temp_dir.name) / (
            f".{self._cache_path().stem}-m429jzwe.tmp"
        )
        in_flight_path.write_text('{"version":2,"items":', encoding="utf-8")

        now = 2_000_000_000.0
        os.utime(in_flight_path, (now - 1, now - 1))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 0)
        self.assertTrue(in_flight_path.exists())


if __name__ == "__main__":
    unittest.main()
