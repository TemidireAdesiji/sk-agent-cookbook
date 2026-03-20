"""Tests for pattern 06: RAG with Azure AI Search (using the in-memory retriever)."""
from rag_with_azure_ai_search import (
    Document,
    InMemoryRetriever,
    RagAgent,
    build_context_block,
    build_rag_agent,
)

from sk_cookbook.testing import ScriptedChatService

DOCS = [
    Document(id="d1", content="The refund window is 30 days from purchase.", source="policy"),
    Document(id="d2", content="Premium plans include priority support.", source="plans"),
    Document(id="d3", content="Data is encrypted at rest using AES-256.", source="security"),
]


def _rag(reply: str = "Grounded answer [d1].") -> tuple[RagAgent, ScriptedChatService]:
    svc = ScriptedChatService(replies=[reply])
    return build_rag_agent(svc, InMemoryRetriever(DOCS), top_k=2), svc


async def test_retriever_finds_relevant_doc() -> None:
    retriever = InMemoryRetriever(DOCS)
    results = await retriever.retrieve("how long is the refund window", top_k=2)
    assert results[0].id == "d1"


async def test_retriever_returns_nothing_for_irrelevant_query() -> None:
    retriever = InMemoryRetriever(DOCS)
    results = await retriever.retrieve("xyzzy nonsense", top_k=3)
    assert results == []


async def test_answer_returns_sources() -> None:
    rag, _ = _rag()
    result = await rag.answer("What is the refund window?")
    assert any(doc.id == "d1" for doc in result.sources)
    assert result.answer == "Grounded answer [d1]."


async def test_retrieved_context_is_injected_into_prompt() -> None:
    rag, svc = _rag()
    await rag.answer("Tell me about the refund window")

    # The agent must have received the document content in its prompt
    sent = svc.last_user_message or ""
    assert "refund window is 30 days" in sent
    assert "Context:" in sent


async def test_no_relevant_docs_still_answers_with_empty_sources() -> None:
    rag, svc = _rag(reply="I don't know.")
    result = await rag.answer("completely unrelated quantum gibberish")

    assert result.sources == []
    sent = svc.last_user_message or ""
    assert "no relevant context found" in sent


def test_context_block_formats_sources() -> None:
    block = build_context_block(DOCS[:2])
    assert "[d1]" in block
    assert "source: policy" in block
    assert "priority support" in block


def test_context_block_empty() -> None:
    assert build_context_block([]) == "(no relevant context found)"


async def test_top_k_limits_sources() -> None:
    svc = ScriptedChatService(replies=["ok"])
    rag = build_rag_agent(svc, InMemoryRetriever(DOCS), top_k=1)
    # A query matching multiple docs should still return at most top_k
    result = await rag.answer("support refund encrypted")
    assert len(result.sources) <= 1
