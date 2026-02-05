"""Tests for build_chat_service (the Azure wiring helper).

These construct the service with dummy credentials - AzureChatCompletion does not make
a network call at construction time, so no live Azure is needed.
"""
import pytest
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from sk_cookbook.service import (
    DEFAULT_API_VERSION,
    ENV_API_KEY,
    ENV_API_VERSION,
    ENV_CHAT_DEPLOYMENT,
    ENV_ENDPOINT,
    MissingConfigError,
    build_chat_service,
)

DUMMY = {
    ENV_ENDPOINT: "https://example.openai.azure.com/",
    ENV_API_KEY: "dummy-key",
    ENV_CHAT_DEPLOYMENT: "gpt-4.1-mini",
}


def _set_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    env = {**DUMMY, **overrides}
    for key in (ENV_ENDPOINT, ENV_API_KEY, ENV_CHAT_DEPLOYMENT, ENV_API_VERSION):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_missing_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (ENV_ENDPOINT, ENV_API_KEY, ENV_CHAT_DEPLOYMENT):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(MissingConfigError):
        build_chat_service()


def test_partial_config_names_missing_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    monkeypatch.delenv(ENV_CHAT_DEPLOYMENT, raising=False)
    with pytest.raises(MissingConfigError, match=ENV_CHAT_DEPLOYMENT):
        build_chat_service()


def test_builds_service_with_all_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    service = build_chat_service()
    assert isinstance(service, AzureChatCompletion)


def test_uses_default_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    service = build_chat_service()
    # The underlying async client must carry the pinned version, not SK's older default.
    assert service.client._api_version == DEFAULT_API_VERSION


def test_explicit_api_version_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, **{ENV_API_VERSION: "2099-09-09"})
    service = build_chat_service(api_version="2024-10-21")
    assert service.client._api_version == "2024-10-21"


def test_env_api_version_used_when_no_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, **{ENV_API_VERSION: "2024-12-01-preview"})
    service = build_chat_service()
    assert service.client._api_version == "2024-12-01-preview"
