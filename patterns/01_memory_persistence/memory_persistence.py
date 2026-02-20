"""Pattern 01: Memory persistence across sessions.

A ChatCompletionAgent only remembers a conversation for the life of its thread. When
the process restarts, the thread is gone. This pattern persists the conversation's
ChatHistory to a JSON store keyed by conversation id, so a later session can reload it
and continue where it left off.

The store is abstracted behind ``ConversationStore`` - a ``FileConversationStore`` is
provided here; swap in Redis, Cosmos DB, or Azure Blob by implementing the same two
methods.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents import ChatHistory

AGENT_NAME = "MemoryAgent"
AGENT_INSTRUCTIONS = (
    "You are a helpful assistant. Use the prior conversation to stay consistent "
    "and avoid asking for information the user already gave you."
)


class ConversationStore(Protocol):
    """Persistence boundary for a conversation's serialized ChatHistory."""

    def load(self, conversation_id: str) -> ChatHistory | None: ...

    def save(self, conversation_id: str, history: ChatHistory) -> None: ...


class FileConversationStore:
    """Stores each conversation as ``<root>/<conversation_id>.json``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self._root / f"{conversation_id}.json"

    def load(self, conversation_id: str) -> ChatHistory | None:
        path = self._path(conversation_id)
        if not path.exists():
            return None
        return ChatHistory.restore_chat_history(path.read_text(encoding="utf-8"))

    def save(self, conversation_id: str, history: ChatHistory) -> None:
        self._path(conversation_id).write_text(history.serialize(), encoding="utf-8")


class PersistentChatSession:
    """Wraps a ChatCompletionAgent so each turn is persisted and resumable.

    On construction the prior history (if any) is loaded from the store. After each
    ``send`` the updated history is saved, so the conversation survives a restart.
    """

    def __init__(
        self,
        agent: ChatCompletionAgent,
        store: ConversationStore,
        conversation_id: str,
    ) -> None:
        self._agent = agent
        self._store = store
        self._conversation_id = conversation_id
        self._history = store.load(conversation_id) or ChatHistory()
        self._thread = ChatHistoryAgentThread(chat_history=self._history)

    @property
    def message_count(self) -> int:
        return len(self._history.messages)

    async def send(self, message: str) -> str:
        """Send a user message, persist the updated conversation, return the reply."""
        response = await self._agent.get_response(messages=message, thread=self._thread)
        self._store.save(self._conversation_id, self._history)
        return str(response.message.content)


def build_agent(service: ChatCompletionClientBase) -> ChatCompletionAgent:
    """Build the memory agent. Pass ``build_chat_service()`` in real use."""
    return ChatCompletionAgent(
        service=service,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
    )


def export_history(store: ConversationStore, conversation_id: str) -> str:
    """Return the stored conversation as pretty JSON, or ``{}`` if none exists."""
    history = store.load(conversation_id)
    if history is None:
        return "{}"
    return json.dumps(json.loads(history.serialize()), indent=2)
