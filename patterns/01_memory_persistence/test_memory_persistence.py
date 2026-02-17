"""Tests for pattern 01: memory persistence."""
from pathlib import Path

from memory_persistence import (
    FileConversationStore,
    PersistentChatSession,
    build_agent,
    export_history,
)

from sk_cookbook.testing import ScriptedChatService


async def test_conversation_survives_new_session(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path)
    svc = ScriptedChatService(replies=["Hi Ada!", "You said your name is Ada."])
    agent = build_agent(svc)

    # Session 1
    session1 = PersistentChatSession(agent, store, conversation_id="conv-1")
    await session1.send("My name is Ada.")

    # Session 2 - a fresh session object, as if the process restarted
    session2 = PersistentChatSession(agent, store, conversation_id="conv-1")
    assert session2.message_count == 2  # the prior user+assistant turn was reloaded

    await session2.send("What is my name?")
    # The reloaded history was sent to the service - the prior user message is present
    full_history = svc.received_histories[-1]
    contents = [str(m.content) for m in full_history.messages]
    assert any("My name is Ada." in c for c in contents)


async def test_separate_conversations_are_isolated(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path)
    svc = ScriptedChatService(replies=["ok"])
    agent = build_agent(svc)

    await PersistentChatSession(agent, store, "conv-a").send("hello from A")
    session_b = PersistentChatSession(agent, store, "conv-b")

    # conv-b starts empty - it must not see conv-a's messages
    assert session_b.message_count == 0


async def test_store_round_trip_persists_to_disk(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path)
    svc = ScriptedChatService(replies=["reply"])
    agent = build_agent(svc)

    await PersistentChatSession(agent, store, "conv-x").send("remember this")

    # The file exists and a brand-new store instance can read it back
    assert (tmp_path / "conv-x.json").exists()
    reloaded = FileConversationStore(tmp_path).load("conv-x")
    assert reloaded is not None
    assert len(reloaded.messages) == 2


async def test_message_count_grows_each_turn(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path)
    svc = ScriptedChatService(replies=["a", "b"])
    session = PersistentChatSession(build_agent(svc), store, "conv-grow")

    await session.send("turn one")
    assert session.message_count == 2
    await session.send("turn two")
    assert session.message_count == 4


def test_export_history_empty_for_unknown_conversation(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path)
    assert export_history(store, "does-not-exist") == "{}"


async def test_export_history_returns_json(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path)
    svc = ScriptedChatService(replies=["hi"])
    await PersistentChatSession(build_agent(svc), store, "conv-e").send("hello")

    exported = export_history(store, "conv-e")
    assert "hello" in exported
    assert exported.startswith("{")
