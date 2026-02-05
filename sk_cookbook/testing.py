"""Fake chat completion services for credential-free testing.

These stand in for AzureChatCompletion so pattern tests run in CI without a live
Azure OpenAI endpoint. They subclass the real ``ChatCompletionClientBase`` so an
agent treats them exactly like a real service.

- ``ScriptedChatService`` returns a fixed reply, or a queue of replies for
  multi-turn conversations, and records every ChatHistory it was asked to complete
  so a test can assert what the agent actually sent.
"""
from collections.abc import AsyncGenerator
from typing import Any

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent, StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


class ScriptedChatService(ChatCompletionClientBase):
    """A chat service that replays scripted assistant replies.

    :param replies: the assistant messages to return, in order. Each call to the
        agent consumes the next one; once exhausted, the last reply repeats.
    """

    # Pydantic model: declare the extra fields the base model does not know about.
    replies: list[str] = []
    call_count: int = 0
    received_histories: list[ChatHistory] = []

    def __init__(self, replies: list[str] | str | None = None, **kwargs: Any) -> None:
        if isinstance(replies, str):
            replies = [replies]
        # The extra fields below are declared on this subclass; the base __init__
        # accepts them via pydantic, but its type stub does not list them.
        super().__init__(  # type: ignore[call-arg]
            ai_model_id=kwargs.pop("ai_model_id", "fake-model"),
            replies=replies or ["This is a scripted reply."],
            call_count=0,
            received_histories=[],
            **kwargs,
        )

    def _next_reply(self) -> str:
        index = min(self.call_count, len(self.replies) - 1)
        return self.replies[index]

    async def _inner_get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> list[ChatMessageContent]:
        self.received_histories.append(chat_history)
        reply = self._next_reply()
        self.call_count += 1
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=reply)]

    async def _inner_get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        function_invoke_attempt: int = 0,
        **kwargs: Any,
    ) -> AsyncGenerator[list[StreamingChatMessageContent], None]:
        self.received_histories.append(chat_history)
        reply = self._next_reply()
        self.call_count += 1
        yield [
            StreamingChatMessageContent(
                role=AuthorRole.ASSISTANT, content=reply, choice_index=0
            )
        ]

    def get_prompt_execution_settings_class(self) -> type[PromptExecutionSettings]:
        return PromptExecutionSettings

    @property
    def last_user_message(self) -> str | None:
        """The most recent user message the service was asked to complete."""
        if not self.received_histories:
            return None
        for message in reversed(self.received_histories[-1].messages):
            if message.role == AuthorRole.USER:
                return str(message.content)
        return None
