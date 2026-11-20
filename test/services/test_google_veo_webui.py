"""Render the Google Veo source and the shared Google account manager in the real WebUI."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config

WEBUI = Path(__file__).parents[2] / "webui/Main.py"


def _open_veo_settings(values):
    app = AppTest.from_file(str(WEBUI), default_timeout=60)
    app.session_state["ui_language"] = "en"
    app.session_state["settings_dialog_open"] = True
    app.session_state["settings_dialog_target_tab"] = "video"
    app.session_state["settings_material_providers"] = {"ai_video": "google_veo"}
    app.run()
    assert not app.exception
    return app


def test_veo_source_offers_model_lengths_and_requires_charge_confirmation():
    values = {"llm_provider": "openai", "video_source": "google_veo", "gemini_api_key": "g"}
    with (
        patch.object(config, "app", values),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.run()
        assert not app.exception
        assert app.selectbox(key="video_clip_duration_select_en").options == ["4", "6", "8"]
        assert not app.checkbox(key="google_veo_confirm_charge").value
        assert any("Google Veo bills" in warning.value for warning in app.warning)


def test_add_select_and_remove_google_accounts():
    values = {"llm_provider": "gemini", "gemini_api_key": "legacy"}
    with (
        patch.object(config, "app", values),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = _open_veo_settings(values)
        # With no accounts the legacy single key stays editable.
        assert app.text_input(key="google_veo_api_key_input")
        app.text_input(key="google_account_new_name_input").set_value("Work")
        app.text_input(key="google_account_new_api_key_input").set_value("work-key")
        app.button(key="google_account_add_button").click().run()
        assert not app.exception
        assert values["google_accounts"] == ["work"]
        assert values["google_account_work_api_key"] == "work-key"

        app.selectbox(key="google_veo_account_select").select("work").run()
        assert not app.exception
        assert values["google_veo_account"] == "work"
        # A selected account replaces the single-key field.
        assert not any(item.key == "google_veo_api_key_input" for item in app.text_input)

        app.session_state["settings_dialog_target_tab"] = "llm"
        app.run()
        assert not app.exception
        app.selectbox(key="gemini_account_select").select("work").run()
        assert values["gemini_account"] == "work"
        assert values["gemini_api_key"] == "legacy"

        app.session_state["settings_dialog_target_tab"] = "video"
        app.run()
        app.button(key="google_account_work_api_key_remove").click().run()
        assert not app.exception
        assert values["google_accounts"] == []
        assert "google_account_work_api_key" not in values
        assert values["google_veo_account"] == "" and values["gemini_account"] == ""
