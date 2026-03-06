"""Tests for pattern 04: cost-aware routing."""
from cost_aware_routing import (
    CostAwareRouter,
    Tier,
    build_tier_agent,
    heuristic_classifier,
)

from sk_cookbook.testing import ScriptedChatService


def _router() -> tuple[CostAwareRouter, ScriptedChatService, ScriptedChatService]:
    cheap_svc = ScriptedChatService(replies=["cheap answer"])
    capable_svc = ScriptedChatService(replies=["capable answer"])
    router = CostAwareRouter(
        build_tier_agent(cheap_svc, "Cheap", "Answer briefly."),
        build_tier_agent(capable_svc, "Capable", "Answer thoroughly."),
        heuristic_classifier(),
    )
    return router, cheap_svc, capable_svc


def test_short_simple_message_is_cheap() -> None:
    classify = heuristic_classifier()
    assert classify("What time is it in Lagos?") == Tier.CHEAP


def test_keyword_forces_capable() -> None:
    classify = heuristic_classifier()
    assert classify("Explain why the sky is blue") == Tier.CAPABLE


def test_long_message_is_capable() -> None:
    classify = heuristic_classifier(long_threshold=50)
    assert classify("x" * 60) == Tier.CAPABLE


def test_custom_keywords() -> None:
    classify = heuristic_classifier(complex_keywords=("summarise",))
    assert classify("summarise this report") == Tier.CAPABLE
    assert classify("hello") == Tier.CHEAP


async def test_handle_routes_simple_to_cheap() -> None:
    router, cheap_svc, capable_svc = _router()
    result = await router.handle("hi there")

    assert result.tier == Tier.CHEAP
    assert result.reply == "cheap answer"
    assert cheap_svc.call_count == 1
    assert capable_svc.call_count == 0


async def test_handle_routes_complex_to_capable() -> None:
    router, cheap_svc, capable_svc = _router()
    result = await router.handle("Please analyze the trade-off here step by step")

    assert result.tier == Tier.CAPABLE
    assert result.reply == "capable answer"
    assert capable_svc.call_count == 1
    assert cheap_svc.call_count == 0


async def test_routed_counts_and_cheap_fraction() -> None:
    router, _, _ = _router()
    await router.handle("hi")               # cheap
    await router.handle("hello")            # cheap
    await router.handle("explain why x")    # capable

    assert router.routed_counts[Tier.CHEAP] == 2
    assert router.routed_counts[Tier.CAPABLE] == 1
    assert router.cheap_fraction == 2 / 3


def test_cheap_fraction_zero_when_no_requests() -> None:
    router, _, _ = _router()
    assert router.cheap_fraction == 0.0
