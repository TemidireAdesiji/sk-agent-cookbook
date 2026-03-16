"""Pattern 05: Structured output.

When an agent's reply feeds another system - a ticketing API, a database, a workflow - free
text is a liability. You need a typed, validated object. This pattern asks the model for
JSON matching a Pydantic schema (via ``response_format``) and then validates the reply into
that model, raising a clear error if the output does not conform.

Validation is defence-in-depth: even with ``response_format`` set, you still parse and
validate the returned text, because a model can return malformed or incomplete JSON.
"""
from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

TModel = TypeVar("TModel", bound=BaseModel)


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SupportTicket(BaseModel):
    """Example target schema: a structured support ticket."""

    category: str
    priority: Priority
    summary: str


class StructuredOutputError(ValueError):
    """Raised when the model's reply cannot be validated into the target schema."""


class StructuredAgent(Generic[TModel]):
    """Wraps a ChatCompletionAgent to return a validated Pydantic object.

    :param agent: the underlying agent.
    :param model: the Pydantic model the reply must conform to.
    """

    def __init__(self, agent: ChatCompletionAgent, model: type[TModel]) -> None:
        self._agent = agent
        self._model = model

    async def extract(self, message: str) -> TModel:
        """Run the agent and validate its reply into the target model.

        :raises StructuredOutputError: if the reply is not valid JSON for the schema.
        """
        response = await self._agent.get_response(messages=message)
        raw = str(response.message.content)
        try:
            return self._model.model_validate_json(raw)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"Model reply did not match {self._model.__name__}: {exc}"
            ) from exc


def build_structured_agent(
    service: ChatCompletionClientBase,
    model: type[TModel],
    name: str = "Extractor",
    instructions: str | None = None,
) -> StructuredAgent[TModel]:
    """Build a StructuredAgent whose execution settings request JSON for ``model``.

    Setting ``response_format=model`` tells Azure OpenAI to constrain output to the
    schema; the StructuredAgent then validates it, so both guards are in place.
    """
    settings = OpenAIChatPromptExecutionSettings(response_format=model)
    agent = ChatCompletionAgent(
        service=service,
        name=name,
        instructions=instructions
        or f"Extract the requested information as JSON matching the {model.__name__} schema.",
        arguments=KernelArguments(settings=settings),
    )
    return StructuredAgent(agent, model)
