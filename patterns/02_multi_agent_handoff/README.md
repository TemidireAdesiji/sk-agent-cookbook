# Pattern 02: Multi-agent handoff

## The problem

A single agent with one giant system prompt that tries to handle billing, technical
support, and account changes does all of them mediocrely. You want focused specialist
agents - but then something has to decide which specialist handles each request and pass
control to it.

## The pattern

A `Coordinator` holds named specialist agents and a `Router`. The router inspects the
incoming message and returns the name of the specialist to handle it; the coordinator
delegates and returns that specialist's reply, tagged with who handled it.

```python
from sk_cookbook import build_chat_service
from multi_agent_handoff import Coordinator, build_specialist, keyword_router

service = build_chat_service()
specialists = {
    "billing": build_specialist(service, "Billing", "You handle billing and refunds."),
    "technical": build_specialist(service, "Technical", "You handle technical issues."),
}
router = keyword_router(
    {"billing": ("invoice", "refund", "payment"), "technical": ("error", "crash", "bug")},
    default="technical",
)
coordinator = Coordinator(specialists, router, default="technical")

result = await coordinator.handle("I need a refund on my invoice")
print(result.specialist)  # "billing"
print(result.reply)
```

Routing here is an ordinary, testable function. You can assert exactly where a message goes
without running a model, and you can see in logs which specialist answered.

## When to use it

- Distinct domains with different instructions, tools, or compliance rules per domain.
- You need the routing decision to be auditable and deterministic.
- Cost control: a cheap router (keywords or a small model) gates expensive specialists.

## Framework-native alternative

Semantic Kernel ships `HandoffOrchestration` and `OrchestrationHandoffs`, where the LLM
itself decides when to hand off between agents. Prefer that when handoff conditions are
fuzzy and hard to express as rules, and you are willing to trade determinism for
flexibility. Prefer the explicit coordinator here when you want routing you can test and
explain.

## Gotchas

- **Context does not transfer by default.** Each specialist here answers the single routed
  message. If a specialist needs the conversation so far, pass a shared thread (see
  pattern 01) or include a summary in the handoff.
- **Router quality is the ceiling.** Keyword routing is brittle for natural language. Start
  with keywords, measure misroutes, and graduate to a small-model classifier router when
  the data justifies it.
- **One hop only.** This coordinator does a single handoff. For chains (A -> B -> C), let
  the chosen specialist itself be a coordinator, or use the framework orchestration.

## Run the test

```bash
pytest patterns/02_multi_agent_handoff
```
