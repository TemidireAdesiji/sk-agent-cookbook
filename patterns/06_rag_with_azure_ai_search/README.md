# Pattern 06: RAG with Azure AI Search

## The problem

An agent answering questions about *your* documents - internal policies, product docs, a
knowledge base - has none of that in its training data. Ask it anyway and it confidently
makes things up. You need it to answer from your content, and to tell you when your content
does not contain the answer.

## The pattern

Retrieval-augmented generation: retrieve the most relevant documents for the question,
inject them into the prompt as context, and instruct the agent to answer only from that
context with citations.

```python
from sk_cookbook import build_chat_service
from rag_with_azure_ai_search import AzureAISearchRetriever, build_rag_agent

retriever = AzureAISearchRetriever(
    endpoint="https://your-search.search.windows.net",
    index_name="docs",
    api_key="...",
)
rag = build_rag_agent(build_chat_service(), retriever, top_k=3)

result = await rag.answer("How long is the refund window?")
print(result.answer)    # "The refund window is 30 days [d1]."
print(result.sources)   # the Document objects that grounded the answer
```

Retrieval sits behind the `DocumentRetriever` protocol. Tests and demos use
`InMemoryRetriever` (no Azure needed); production uses `AzureAISearchRetriever`. The agent
code is identical either way - you only swap the retriever.

## When to use it

- Question answering over private or frequently-changing documents.
- Anywhere hallucination about your own content is unacceptable (support, compliance, docs).
- When answers must be traceable to a source.

## Gotchas

- **Retrieval quality is the ceiling.** If the right document is not retrieved, the agent
  cannot answer correctly - it can only honestly say it does not know. Invest in chunking,
  embeddings, and hybrid (keyword + vector) search before blaming the model.
- **Citations are not proof.** The model can cite a source that does not actually support
  the claim. For high-stakes use, verify cited spans against the retrieved text.
- **Context window limits `top_k`.** Stuffing 50 documents into the prompt is slow, costly,
  and dilutes the relevant ones. Retrieve few, high-quality chunks.
- **The in-memory retriever is a toy.** Its keyword overlap is for demos and tests only. Do
  not ship it - use Azure AI Search (ideally with vector or hybrid queries).

## Run the test

```bash
pytest patterns/06_rag_with_azure_ai_search
```
