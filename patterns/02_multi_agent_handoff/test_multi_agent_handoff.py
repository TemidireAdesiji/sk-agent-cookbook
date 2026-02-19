"""Tests for pattern 02: multi-agent handoff."""
import pytest
from multi_agent_handoff import (
    Coordinator,
    build_specialist,
    keyword_router,
)

from sk_cookbook.testing import ScriptedChatService


def _coordinator() -> Coordinator:
    billing = build_specialist(
        ScriptedChatService(replies=["Your invoice is attached."]), "billing", "Handle billing."
    )
    technical = build_specialist(
        ScriptedChatService(replies=["Try restarting the service."]), "technical", "Handle tech."
    )
    router = keyword_router(
        {
            "billing": ("invoice", "payment", "refund"),
            "technical": ("error", "crash", "bug"),
        },
        default="technical",
    )
    return Coordinator({"billing": billing, "technical": technical}, router, default="technical")


def test_routes_billing_keyword() -> None:
    assert _coordinator().route("I need a refund on my invoice") == "billing"


def test_routes_technical_keyword() -> None:
    assert _coordinator().route("the app crashed with an error") == "technical"


def test_unmatched_message_uses_default() -> None:
    assert _coordinator().route("hello there") == "technical"


async def test_handle_returns_specialist_reply() -> None:
    result = await _coordinator().handle("I need a refund")
    assert result.specialist == "billing"
    assert result.reply == "Your invoice is attached."


async def test_handle_routes_technical() -> None:
    result = await _coordinator().handle("there is a bug")
    assert result.specialist == "technical"
    assert "restarting" in result.reply


def test_unknown_router_result_falls_back_to_default() -> None:
    billing = build_specialist(ScriptedChatService(replies=["x"]), "billing", "b")
    coordinator = Coordinator(
        {"billing": billing},
        router=lambda _msg: "nonexistent",  # never a valid key
        default="billing",
    )
    assert coordinator.route("anything") == "billing"


def test_default_must_be_a_known_specialist() -> None:
    billing = build_specialist(ScriptedChatService(replies=["x"]), "billing", "b")
    with pytest.raises(ValueError, match="default"):
        Coordinator({"billing": billing}, router=lambda _m: "billing", default="missing")


async def test_each_specialist_receives_only_its_messages() -> None:
    billing_svc = ScriptedChatService(replies=["billing reply"])
    tech_svc = ScriptedChatService(replies=["tech reply"])
    coordinator = Coordinator(
        {
            "billing": build_specialist(billing_svc, "billing", "b"),
            "technical": build_specialist(tech_svc, "technical", "t"),
        },
        keyword_router({"billing": ("invoice",)}, default="technical"),
        default="technical",
    )

    await coordinator.handle("about my invoice")
    assert billing_svc.call_count == 1
    assert tech_svc.call_count == 0
