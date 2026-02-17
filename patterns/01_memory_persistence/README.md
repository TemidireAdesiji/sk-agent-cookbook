# Pattern 01: Memory persistence across sessions

## The problem

A `ChatCompletionAgent` remembers a conversation only for the life of its thread. The
moment your process restarts - a new request, a redeployed container, a fresh worker - the
thread is gone and the agent has amnesia. Users notice immediately: they have to
re-introduce themselves and repeat context every session.

## The pattern

Persist the conversation's `ChatHistory` to durable storage keyed by a conversation id.
On the next session, reload it into a fresh thread and continue.

`PersistentChatSession` wraps the agent and does this automatically:

```python
from sk_cookbook import build_chat_service
from memory_persistence import FileConversationStore, PersistentChatSession, build_agent

agent = build_agent(build_chat_service())
store = FileConversationStore("./conversations")

session = PersistentChatSession(agent, store, conversation_id="user-42")
print(await session.send("My name is Ada."))
# ... process restarts ...
session = PersistentChatSession(agent, store, conversation_id="user-42")
print(await session.send("What is my name?"))   # remembers "Ada"
```

The key calls are `ChatHistory.serialize()` to write and `ChatHistory.restore_chat_history()`
to read. Because `ChatHistoryAgentThread(chat_history=...)` appends to the history object you
pass in, the wrapper keeps a reference and saves it after each turn.

## When to use it

- Multi-session assistants where users return over hours or days.
- Stateless web workers where any request may land on a different process.
- Anywhere "the agent forgot what I told it" is a real complaint.

## Gotchas

- **Unbounded growth.** A long-running conversation grows without limit and eventually
  blows the model's context window. Use `ChatHistoryAgentThread.reduce()` (history
  reducers) to truncate or summarise before persisting. This pattern keeps the full
  history for clarity - add reduction before production.
- **Swap the store, not the pattern.** `FileConversationStore` is fine for a single box.
  For real deployments implement `ConversationStore` against Redis, Cosmos DB, or Azure
  Blob - the agent code does not change.
- **PII.** A persisted conversation is persisted user data. Encrypt at rest and apply your
  retention policy; do not keep transcripts forever by default.

## Run the test

```bash
pytest patterns/01_memory_persistence
```
