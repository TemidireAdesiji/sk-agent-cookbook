# Pattern 05: Structured output

## The problem

When an agent's reply is consumed by code - a ticketing API, a database insert, the next
step of a workflow - free text is a liability. You end up writing brittle regex to pull
fields out of prose, and a single reworded reply breaks the parse. You want the agent to
return a typed, validated object.

## The pattern

Ask the model for JSON matching a Pydantic schema via `response_format`, then validate the
reply into that model. If the reply does not conform, raise a clear error instead of
passing malformed data downstream.

```python
from pydantic import BaseModel
from sk_cookbook import build_chat_service
from structured_output_agent import build_structured_agent

class SupportTicket(BaseModel):
    category: str
    priority: str
    summary: str

agent = build_structured_agent(build_chat_service(), SupportTicket)
ticket = await agent.extract("I was double-charged on my last invoice!")
# ticket is a validated SupportTicket, not a string
ticket.category   # "billing"
```

Two guards are stacked: `response_format=SupportTicket` constrains the model's output at
the API level, and `model_validate_json` validates it again on the way in. The second guard
matters because models can still emit malformed or incomplete JSON, and your code should
never trust unvalidated output.

## When to use it

- The reply feeds another system that expects fixed fields.
- You need enums/ranges enforced (priority must be one of low/medium/high).
- Extraction, classification, and form-filling tasks.

## Gotchas

- **Validate even with `response_format`.** Structured outputs reduce malformed replies but do
  not eliminate them, especially on older models or under truncation. Keep the Pydantic
  validation; this pattern raises `StructuredOutputError` rather than returning bad data.
- **Keep schemas flat-ish.** Deeply nested or huge schemas raise the model's error rate.
  Extract in stages if the target object is large.
- **Enums need exact values.** The model will occasionally return `"URGENT"` for a `"high"`
  enum. The validation catches it - decide whether to retry, map synonyms, or surface the error.
- **Handle the error.** A raised `StructuredOutputError` is your signal to retry the call or
  fall back, not a reason to crash the request.

## Run the test

```bash
pytest patterns/05_structured_output_agent
```
