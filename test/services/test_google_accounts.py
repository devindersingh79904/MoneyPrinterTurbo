import pytest

from app.services import google_accounts as accounts


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


SETTINGS = {
    "google_accounts": ["personal", "Work", "bad name!", "personal"],
    "google_account_personal_api_key": " p-key ",
    "google_account_work_api_key": "w-key",
    "gemini_api_key": "legacy-gemini",
    "google_veo_api_key": "legacy-veo",
}


def test_account_names_are_normalized_and_deduplicated():
    assert accounts.account_names(SETTINGS) == ["personal", "work"]


def test_each_usage_selects_its_own_account():
    settings = {**SETTINGS, "google_veo_account": "work", "gemini_account": "personal"}
    assert accounts.resolve_api_key("veo", settings) == "w-key"
    assert accounts.resolve_api_key("gemini", settings) == "p-key"


def test_no_selection_keeps_legacy_keys():
    assert accounts.resolve_api_key("veo", SETTINGS) == "legacy-veo"
    assert accounts.resolve_api_key("gemini", SETTINGS) == "legacy-gemini"
    assert accounts.resolve_api_key("veo", {"gemini_api_key": "g"}) == "g"


def test_environment_is_last_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert accounts.resolve_api_key("gemini", {}) == "env-key"


@pytest.mark.parametrize("selection", ["missing", "keyless"])
def test_bad_selection_never_falls_back_to_another_account(selection):
    settings = {**SETTINGS, "google_accounts": ["keyless"], "gemini_account": selection}
    with pytest.raises(accounts.GoogleAccountError):
        accounts.resolve_api_key("gemini", settings)
    assert not accounts.has_api_key("gemini", settings)


def test_names_are_safe_config_keys():
    assert accounts.api_key_config_key("My-Work") == "google_account_my_work_api_key"
    for bad in ("", "../x", "a" * 33, "_lead"):
        with pytest.raises(accounts.GoogleAccountError):
            accounts.normalize_name(bad)


def test_unknown_usage_is_a_programming_error():
    with pytest.raises(ValueError):
        accounts.resolve_api_key("imagen", {})
