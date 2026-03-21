"""Pattern 06: Retrieval-augmented generation (RAG) with Azure AI Search.

An agent that answers from its training data hallucinates about your private documents.
RAG fixes this: retrieve the most relevant documents for the question, inject them into the
prompt, and instruct the agent to answer only from that context, with citations.

Retrieval is abstracted behind ``DocumentRetriever``. ``InMemoryRetriever`` is a dependency-
free implementation for demos and tests; ``AzureAISearchRetriever`` is the production one
(needs the ``[search]`` extra). The agent code does not change between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase

GROUNDING_INSTRUCTIONS = (
    "You answer strictly from the provided context. If the context does not contain the "
    "answer, say you do not know. Cite the source id in square brackets after each claim."
)


class Document(BaseModel):
    """A retrieved document chunk."""

    id: str
    content: str
    source: str = "unknown"


@dataclass
class GroundedAnswer:
    answer: str
    sources: list[Document]


class DocumentRetriever(Protocol):
    """Retrieval boundary. Implement this for any backing store."""

    async def retrieve(self, query: str, top_k: int) -> list[Document]: ...


class InMemoryRetriever:
    """Naive keyword-overlap retriever over an in-memory document set.

    Good enough for demos and tests; not a real search engine. Scores documents by the
    number of query words they contain and returns the top ``top_k``.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    async def retrieve(self, query: str, top_k: int) -> list[Document]:
        query_words = {w for w in query.lower().split() if len(w) > 2}

        def score(doc: Document) -> int:
            content = doc.content.lower()
            return sum(1 for word in query_words if word in content)

        ranked = sorted(self._documents, key=score, reverse=True)
        return [doc for doc in ranked[:top_k] if score(doc) > 0]


class AzureAISearchRetriever:
    """Production retriever backed by Azure AI Search.

    Requires the ``[search]`` extra (``azure-search-documents``). The import is deferred so
    this module loads without the dependency installed.
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        api_key: str,
        content_field: str = "content",
    ) -> None:
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "AzureAISearchRetriever requires the [search] extra: "
                'pip install "sk-agent-cookbook[search]"'
            ) from exc

        self._client = SearchClient(endpoint, index_name, AzureKeyCredential(api_key))
        self._content_field = content_field

    # pragma: no cover - needs a live Azure AI Search service
    async def retrieve(self, query: str, top_k: int) -> list[Document]:
        results = self._client.search(search_text=query, top=top_k)
        documents: list[Document] = []
        for result in results:
            result_any: dict[str, Any] = dict(result)
            documents.append(
                Document(
                    id=str(result_any.get("id", result_any.get("@search.score", "?"))),
                    content=str(result_any.get(self._content_field, "")),
                    source=str(result_any.get("source", "azure-ai-search")),
                )
            )
        return documents


def build_context_block(documents: list[Document]) -> str:
    """Render retrieved documents into a context block for the prompt."""
    if not documents:
        return "(no relevant context found)"
    return "\n\n".join(f"[{doc.id}] (source: {doc.source})\n{doc.content}" for doc in documents)


class RagAgent:
    """Retrieves context, grounds the prompt, and returns an answer plus its sources."""

    def __init__(
        self,
        agent: ChatCompletionAgent,
        retriever: DocumentRetriever,
        top_k: int = 3,
    ) -> None:
        self._agent = agent
        self._retriever = retriever
        self._top_k = top_k

    async def answer(self, question: str) -> GroundedAnswer:
        documents = await self._retriever.retrieve(question, self._top_k)
        context = build_context_block(documents)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = await self._agent.get_response(messages=prompt)
        return GroundedAnswer(answer=str(response.message.content), sources=documents)


def build_rag_agent(
    service: ChatCompletionClientBase,
    retriever: DocumentRetriever,
    top_k: int = 3,
) -> RagAgent:
    """Build a RAG agent grounded in the given retriever."""
    agent = ChatCompletionAgent(
        service=service,
        name="GroundedAssistant",
        instructions=GROUNDING_INSTRUCTIONS,
    )
    return RagAgent(agent, retriever, top_k=top_k)
