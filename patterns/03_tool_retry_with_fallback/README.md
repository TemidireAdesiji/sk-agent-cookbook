# Pattern 03: Tool retry with fallback

## The problem

Agent tools call unreliable things: third-party APIs, networks, rate-limited endpoints. When
a tool raises, the whole agent turn fails and the user sees an error - even though the call
would likely have succeeded on a second try a moment later. Worse, an unhandled tool
exception can abort a multi-step plan halfway through.

## The pattern

Wrap the tool's work in `call_with_retry`: bounded attempts, exponential backoff, and a
graceful fallback. The agent always receives a usable string instead of an exception.

```python
from tool_retry_with_fallback import call_with_retry, ResilientToolPlugin

# Reusable core - wrap any flaky async operation:
result = await call_with_retry(
    lambda: external_api.fetch(query),
    max_attempts=3,
    base_delay=0.5,                       # 0.5s, 1s, 2s backoff
    retry_on=(ConnectionError, TimeoutError),
    fallback=lambda: "Service unavailable, please try again shortly.",
)

# As an agent tool:
plugin = ResilientToolPlugin(external_api.fetch, max_attempts=3)
agent = ChatCompletionAgent(service=service, name="Support", plugins=[plugin])
```

`retry_on` is deliberately explicit: only transient errors should be retried. A
`ValueError` from bad input will never succeed on retry, so it propagates immediately
instead of wasting attempts.

## When to use it

- Any tool that hits the network or a rate-limited service.
- Operations where a stale-but-present answer beats a hard failure (search, enrichment).
- Plans where one flaky step should not abort the whole agent run.

## Gotchas

- **Don't retry non-idempotent writes blindly.** Retrying "charge the card" can double-charge.
  Restrict `retry_on`, or make the operation idempotent (idempotency keys) before retrying.
- **Backoff matters under load.** Retrying instantly hammers an already-struggling service.
  The exponential backoff here is the floor; add jitter for high-concurrency callers.
- **Fallbacks must be honest.** A fallback that fabricates data is worse than an error. Return
  a clear "unavailable" message (as here) so the agent and user know the result is degraded.
- **Set a budget.** `max_attempts` × max backoff is your worst-case latency. Keep it under the
  caller's timeout.

## Run the test

```bash
pytest patterns/03_tool_retry_with_fallback
```
