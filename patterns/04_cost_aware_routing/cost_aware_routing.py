"""Pattern 04: Cost-aware model routing.

Most requests to an assistant are easy and a small, cheap model answers them perfectly.
A minority are genuinely hard and need a capable (expensive) model. Sending everything to
the big model wastes money; sending everything to the small one wastes quality. This
pattern routes each request to the right tier based on an estimated complexity.

The classifier here is a transparent heuristic so routing is testable and explainable.
Swap in a small-model classifier when heuristics stop being good enough.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase


class Tier(StrEnum):
    CHEAP = "cheap"
    CAPABLE = "capable"


# A classifier maps a message to the tier that should handle it.
Classifier = Callable[[str], "Tier"]


@dataclass
class RoutingResult:
    tier: Tier
    reply: str


# Signals that a request likely needs the capable model.
DEFAULT_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "explain why",
    "analyse",
    "analyze",
    "step by step",
    "prove",
    "design",
    "compare and contrast",
    "trade-off",
    "tradeoff",
    "debug",
    "refactor",
)


def heuristic_classifier(
    *,
    long_threshold: int = 280,
    complex_keywords: tuple[str, ...] = DEFAULT_COMPLEX_KEYWORDS,
) -> Classifier:
    """Return a classifier that flags a request as CAPABLE when it is long or contains a
    reasoning keyword, otherwise CHEAP.

    :param long_threshold: messages longer than this many characters go to the capable tier.
    :param complex_keywords: substrings (matched case-insensitively) that force the capable tier.
    """

    def classify(message: str) -> Tier:
        lowered = message.lower()
        if len(message) > long_threshold:
            return Tier.CAPABLE
        if any(keyword in lowered for keyword in complex_keywords):
            return Tier.CAPABLE
        return Tier.CHEAP

    return classify


class CostAwareRouter:
    """Routes each request to the cheap or capable agent based on a classifier."""

    def __init__(
        self,
        cheap_agent: ChatCompletionAgent,
        capable_agent: ChatCompletionAgent,
        classifier: Classifier,
    ) -> None:
        self._agents = {Tier.CHEAP: cheap_agent, Tier.CAPABLE: capable_agent}
        self._classifier = classifier
        self.routed_counts = {Tier.CHEAP: 0, Tier.CAPABLE: 0}

    def classify(self, message: str) -> Tier:
        return self._classifier(message)

    async def handle(self, message: str) -> RoutingResult:
        tier = self.classify(message)
        self.routed_counts[tier] += 1
        response = await self._agents[tier].get_response(messages=message)
        return RoutingResult(tier=tier, reply=str(response.message.content))

    @property
    def cheap_fraction(self) -> float:
        """Share of requests sent to the cheap tier - your cost-savings proxy."""
        total = sum(self.routed_counts.values())
        return self.routed_counts[Tier.CHEAP] / total if total else 0.0


def build_tier_agent(
    service: ChatCompletionClientBase, name: str, instructions: str
) -> ChatCompletionAgent:
    """Build an agent for one tier. In production give each tier a different deployment."""
    return ChatCompletionAgent(service=service, name=name, instructions=instructions)
