"""Named Google (Gemini API) accounts shared by Veo, the Gemini LLM and Gemini TTS.

Accounts are stored as flat config keys so the existing key backup/restore and
secret redaction keep working without special cases::

    google_accounts = ["personal", "work"]
    google_account_personal_api_key = "..."
    google_account_work_api_key = "..."

    google_veo_account = "work"       # account used for Veo clips
    gemini_account = "personal"       # account used for script writing and Gemini TTS

Each usage picks an account independently. An empty selection keeps the legacy
single-key behaviour, so existing configs need no migration. A selection that
names a missing or keyless account is an error rather than a silent fallback,
so a typo can never bill a different Google account.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.config import config

ACCOUNTS_KEY = "google_accounts"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


@dataclass(frozen=True)
class Usage:
    """Where a feature reads its account selection and its pre-accounts fallback keys."""

    selection_key: str
    legacy_keys: tuple[str, ...]


USAGES = {
    "veo": Usage("google_veo_account", ("google_veo_api_key", "gemini_api_key")),
    "gemini": Usage("gemini_account", ("gemini_api_key",)),
}


@dataclass(frozen=True)
class GoogleAccount:
    name: str
    api_key: str = field(repr=False)


class GoogleAccountError(ValueError):
    """The configured account selection cannot be satisfied."""


def _settings(settings: Mapping | None) -> Mapping:
    return config.app if settings is None else settings


def normalize_name(name: str) -> str:
    """Lowercase and validate an account name; names become part of a config key."""
    value = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not _NAME_PATTERN.match(value):
        raise GoogleAccountError(
            "Account names use 1-32 letters, digits or underscores and start with a letter or digit"
        )
    return value


def api_key_config_key(name: str) -> str:
    return f"google_account_{normalize_name(name)}_api_key"


def account_names(settings: Mapping | None = None) -> list[str]:
    raw = _settings(settings).get(ACCOUNTS_KEY, [])
    names = []
    for item in raw if isinstance(raw, list) else []:
        try:
            name = normalize_name(item)
        except GoogleAccountError:
            continue
        if name not in names:
            names.append(name)
    return names


def list_accounts(settings: Mapping | None = None) -> list[GoogleAccount]:
    values = _settings(settings)
    return [
        GoogleAccount(name, str(values.get(api_key_config_key(name), "") or "").strip())
        for name in account_names(values)
    ]


def selected_account(usage: str, settings: Mapping | None = None) -> str:
    return str(_settings(settings).get(_usage(usage).selection_key, "") or "").strip().lower()


def resolve_api_key(usage: str, settings: Mapping | None = None) -> str:
    """Return the API key for ``usage`` ("veo" or "gemini"); empty when nothing is configured."""
    values = _settings(settings)
    spec = _usage(usage)
    selected = selected_account(usage, values)
    if selected:
        account = next((a for a in list_accounts(values) if a.name == selected), None)
        if account is None:
            raise GoogleAccountError(f"Google account {selected!r} is not configured")
        if not account.api_key:
            raise GoogleAccountError(f"Google account {selected!r} has no API key")
        return account.api_key
    for key in spec.legacy_keys:
        value = str(values.get(key, "") or "").strip()
        if value:
            return value
    return next((os.getenv(k, "").strip() for k in _ENV_KEYS if os.getenv(k, "").strip()), "")


def has_api_key(usage: str, settings: Mapping | None = None) -> bool:
    try:
        return bool(resolve_api_key(usage, settings))
    except GoogleAccountError:
        return False


def _usage(usage: str) -> Usage:
    try:
        return USAGES[usage]
    except KeyError:
        raise ValueError(f"unknown Google account usage: {usage!r}") from None
