# Pattern 04: Cost-aware model routing

## The problem

Most requests to an assistant are easy - a small model like `gpt-4o-mini` answers them
perfectly at a fraction of the price. A minority are genuinely hard and need `gpt-4o`.
Routing everything to the capable model can cost 15-20x more than necessary; routing
everything to the small model tanks quality on the hard requests. You want to pay for
capability only when capability is needed.

## The pattern

A `CostAwareRouter` holds a cheap agent and a capable agent and a classifier. The
classifier estimates request complexity; the router sends the request to the matching
tier and tracks the split so you can see your savings.

```python
from sk_cookbook import build_chat_service
from cost_aware_routing import CostAwareRouter, build_tier_agent, heuristic_classifier

cheap = build_tier_agent(cheap_service, "Cheap", "Answer briefly.")
capable = build_tier_agent(capable_service, "Capable", "Reason carefully.")
router = CostAwareRouter(cheap, capable, heuristic_classifier())

result = await router.handle("What time is it in Lagos?")     # -> Tier.CHEAP
result = await router.handle("Explain why this query is slow") # -> Tier.CAPABLE

print(router.cheap_fraction)   # e.g. 0.82 -> 82% of traffic stayed cheap
```

The heuristic classifier routes to the capable tier when the message is long or contains a
reasoning keyword (`explain why`, `analyse`, `step by step`, `debug`, ...). It is a plain
function, so you can unit-test exactly where each message goes.

## Pairs well with `openai-cost-guard`

Routing decides *which* model runs; measuring *what it cost* is a separate concern. Wrap the
tier agents' calls with [`openai-cost-guard`](../../../openai-cost-guard) to turn
`cheap_fraction` into actual dollars saved.

## When to use it

- High request volume where the easy/hard split is real (support, Q&A, classification).
- A clear cheap vs capable deployment pair available in your Azure OpenAI resource.
- You can tolerate occasional misroutes (a hard question landing on the cheap model).

## Gotchas

- **Misroute cost is asymmetric.** A hard question routed to the cheap model gives a worse
  answer; an easy one routed to the capable model just costs more. Tune the classifier to
  fail toward `CAPABLE` when the stakes of a bad answer are high.
- **Heuristics drift.** Keyword/length rules are a starting point. Log routing decisions and
  outcomes; when misroutes climb, replace the heuristic with a small-model classifier
  (which is itself a cheap call).
- **Two deployments, two prices.** Make sure the cheap tier really is a cheaper deployment -
  routing to two `gpt-4o` deployments saves nothing.

## Run the test

```bash
pytest patterns/04_cost_aware_routing
```
