"""Tests for pattern 05: structured output."""
import pytest
from structured_output_agent import (
    Priority,
    StructuredAgent,
    StructuredOutputError,
    SupportTicket,
    build_structured_agent,
)

from sk_cookbook.testing import ScriptedChatService


def _agent_returning(raw_reply: str) -> StructuredAgent[SupportTicket]:
    svc = ScriptedChatService(replies=[raw_reply])
    return build_structured_agent(svc, SupportTicket)


async def test_valid_json_parses_into_model() -> None:
    raw = '{"category": "billing", "priority": "high", "summary": "Double charged"}'
    ticket = await _agent_returning(raw).extract("I was charged twice!")

    assert isinstance(ticket, SupportTicket)
    assert ticket.category == "billing"
    assert ticket.priority is Priority.HIGH
    assert ticket.summary == "Double charged"


async def test_malformed_json_raises_structured_error() -> None:
    agent = _agent_returning("this is not json at all")
    with pytest.raises(StructuredOutputError):
        await agent.extract("anything")


async def test_missing_required_field_raises() -> None:
    raw = '{"category": "billing", "summary": "no priority field"}'
    agent = _agent_returning(raw)
    with pytest.raises(StructuredOutputError):
        await agent.extract("anything")


async def test_invalid_enum_value_raises() -> None:
    raw = '{"category": "billing", "priority": "URGENT", "summary": "bad enum"}'
    agent = _agent_returning(raw)
    with pytest.raises(StructuredOutputError):
        await agent.extract("anything")


async def test_error_message_names_the_model() -> None:
    agent = _agent_returning("{}")
    with pytest.raises(StructuredOutputError, match="SupportTicket"):
        await agent.extract("anything")


async def test_works_with_arbitrary_model() -> None:
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    svc = ScriptedChatService(replies=['{"name": "Ada", "age": 36}'])
    agent = build_structured_agent(svc, Person)
    person = await agent.extract("Ada is 36")

    assert person.name == "Ada"
    assert person.age == 36
