"""Pattern 02: Multi-agent handoff.

One generalist agent rarely does everything well. This pattern uses a coordinator that
routes each request to the right specialist agent (billing, technical, ...) and returns
that specialist's answer, recording which one handled it.

This is the explicit, debuggable form of handoff: routing is a function you can read and
test, not an opaque LLM decision. Semantic Kernel also ships ``HandoffOrchestration`` for
LLM-driven handoff - see the README for when to prefer that.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase


@dataclass
class HandoffResult:
    """The outcome of a coordinated request."""

    specialist: str
    reply: str


# A router maps a user message to the name of the specialist that should handle it.
Router = Callable[[str], str]


class Coordinator:
    """Routes each request to a named specialist agent.

    :param specialists: mapping of specialist name -> agent.
    :param router: decides which specialist handles a message. Must return a key
        present in ``specialists``; if it returns an unknown key, ``default`` is used.
    :param default: specialist used when the router returns an unknown name.
    """

    def __init__(
        self,
        specialists: dict[str, ChatCompletionAgent],
        router: Router,
        default: str,
    ) -> None:
        if default not in specialists:
            raise ValueError(f"default {default!r} is not among specialists")
        self._specialists = specialists
        self._router = router
        self._default = default

    def route(self, message: str) -> str:
        """Return the specialist name that will handle ``message``."""
        chosen = self._router(message)
        return chosen if chosen in self._specialists else self._default

    async def handle(self, message: str) -> HandoffResult:
        """Route the message to a specialist and return its reply."""
        name = self.route(message)
        agent = self._specialists[name]
        response = await agent.get_response(messages=message)
        return HandoffResult(specialist=name, reply=str(response.message.content))


def keyword_router(keywords: dict[str, tuple[str, ...]], default: str) -> Router:
    """Build a simple router that matches keywords to specialist names.

    :param keywords: specialist name -> tuple of trigger words (matched case-insensitively).
    :param default: returned when no keyword matches.
    """

    def route(message: str) -> str:
        lowered = message.lower()
        for name, triggers in keywords.items():
            if any(trigger in lowered for trigger in triggers):
                return name
        return default

    return route


def build_specialist(
    service: ChatCompletionClientBase, name: str, instructions: str
) -> ChatCompletionAgent:
    """Build a specialist agent."""
    return ChatCompletionAgent(service=service, name=name, instructions=instructions)
