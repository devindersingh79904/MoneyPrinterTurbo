from unittest import mock

import pytest

from app.config import config
from app.models.schema import VideoAspect
from app.services import google_veo


class FakeResponse:
    def __init__(self, status_code=200, body=None, chunks=(), headers=None):
        self.status_code = status_code
        self._body = body
        self._chunks = chunks
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_content(self, _size):
        yield from self._chunks

    def close(self):
        pass


@pytest.fixture
def veo_config(monkeypatch):
    monkeypatch.setattr(config, "app", {"google_veo_api_key": "secret-key"})
    monkeypatch.setattr(config, "proxy", {})


def test_key_falls_back_to_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert google_veo.get_api_key({"gemini_api_key": " g "}) == "g"
    assert google_veo.get_api_key({"google_veo_api_key": "v", "gemini_api_key": "g"}) == "v"
    assert not google_veo.is_enabled({})


@pytest.mark.parametrize(
    "requested,model,expected",
    [(3, "veo-3.1-fast-generate-preview", 4), (5, "veo-3.0-generate-001", 6),
     (12, "veo-3.1-generate-preview", 8), (3, "veo-2.0-generate-001", 5)],
)
def test_clip_seconds_snaps_to_model_lengths(requested, model, expected):
    assert google_veo.clip_seconds(requested, {"google_veo_model": model}) == expected


def test_rejects_square_aspect(veo_config, tmp_path):
    with pytest.raises(google_veo.GoogleVeoError):
        google_veo.generate_video(
            search_term="city", video_aspect=VideoAspect.square,
            clip_duration=5, output_dir=str(tmp_path),
        )


def test_generate_video_submits_polls_and_downloads(veo_config, tmp_path, monkeypatch):
    operation = "models/veo-3.1-fast-generate-preview/operations/op1"
    uri = "https://generativelanguage.googleapis.com/v1beta/files/f1:download?alt=media"
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            return FakeResponse(body={"name": operation})
        if url.endswith(operation):
            done = sum(1 for c in calls if c[1].endswith(operation)) > 1
            body = {"name": operation, "done": done}
            if done:
                body["response"] = {"generateVideoResponse": {
                    "generatedSamples": [{"video": {"uri": uri}}]}}
            return FakeResponse(body=body)
        if url == uri:
            return FakeResponse(status_code=302, headers={"Location": "https://storage.example/v.mp4"})
        return FakeResponse(chunks=[b"video-bytes"])

    monkeypatch.setattr(google_veo, "_request", fake_request)
    monkeypatch.setattr(google_veo.time, "sleep", lambda _s: None)
    monkeypatch.setattr(google_veo.material_upload, "_validate_video", lambda _p: None)
    fake_clip = mock.MagicMock()
    fake_clip.__enter__.return_value.duration = 6.0
    fake_clip.__enter__.return_value.size = (720, 1280)
    with mock.patch("moviepy.VideoFileClip", return_value=fake_clip):
        item = google_veo.generate_video(
            search_term="stock market", video_aspect=VideoAspect.portrait,
            clip_duration=5, output_dir=str(tmp_path),
        )

    method, url, kwargs = calls[0]
    assert url.endswith("/models/veo-3.1-fast-generate-preview:predictLongRunning")
    params = kwargs["json"]["parameters"]
    assert params["aspectRatio"] == "9:16" and params["durationSeconds"] == 6
    assert "stock market" in kwargs["json"]["instances"][0]["prompt"]
    # The key goes to Google's host but never to the redirected storage host.
    storage_call = calls[-1]
    assert storage_call[1] == "https://storage.example/v.mp4"
    assert storage_call[2]["headers"] == {}
    assert item.provider == "google_veo" and item.duration == 6
    assert item.source_info["asset_id"] == operation
    assert item.url.endswith(".mp4")


def test_server_error_on_submit_is_unconfirmed(veo_config, tmp_path, monkeypatch):
    monkeypatch.setattr(google_veo, "_request", lambda *a, **k: FakeResponse(status_code=503))
    with pytest.raises(google_veo.GoogleVeoUnconfirmedError):
        google_veo.generate_video(
            search_term="city", video_aspect=VideoAspect.landscape,
            clip_duration=8, output_dir=str(tmp_path),
        )


def test_safety_filter_reports_reason(veo_config):
    body = {"done": True, "response": {"generateVideoResponse": {
        "raiMediaFilteredReasons": ["blocked"]}}}
    with pytest.raises(google_veo.GoogleVeoFilteredError, match="blocked"):
        google_veo._video_uri(body, "op", "secret-key")


def test_on_demand_skips_filtered_keywords_and_stops_when_covered(monkeypatch, tmp_path):
    from app.models.schema import MaterialInfo
    from app.services import material

    calls = []

    def fake_generate(*, search_term, video_aspect, clip_duration, output_dir):
        calls.append(search_term)
        if search_term == "blocked":
            raise google_veo.GoogleVeoFilteredError("filtered", "op-x")
        return MaterialInfo(provider="google_veo", url=f"{search_term}.mp4", duration=6)

    monkeypatch.setattr(google_veo, "generate_video", fake_generate)
    monkeypatch.setattr(material.utils, "task_dir", lambda _t: str(tmp_path))
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)
    paths = material._download_videos_google_veo_on_demand(
        task_id="t", search_terms=["blocked", "a", "b", "c"],
        video_aspect=VideoAspect.portrait, audio_duration=10, max_clip_duration=6,
    )
    assert paths == ["a.mp4", "b.mp4"]
    assert calls == ["blocked", "a", "b"]


def test_on_demand_fails_when_every_keyword_is_filtered(monkeypatch, tmp_path):
    from app.services import material

    def fake_generate(**_kwargs):
        raise google_veo.GoogleVeoFilteredError("filtered")

    monkeypatch.setattr(google_veo, "generate_video", fake_generate)
    monkeypatch.setattr(material.utils, "task_dir", lambda _t: str(tmp_path))
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)
    with pytest.raises(google_veo.GoogleVeoError, match="no clips"):
        material._download_videos_google_veo_on_demand(
            task_id="t", search_terms=["x"], video_aspect=VideoAspect.portrait,
            audio_duration=10, max_clip_duration=6,
        )
