# Pattern 07: Human in the loop

## The problem

You want an agent to be useful enough to *act* - delete records, issue refunds, send
emails - but some of those actions are irreversible. Letting the model fire them
autonomously is a risk; removing them entirely makes the agent useless. You need the agent
to propose the action and a human to approve it before it executes.

## The pattern

Register a Semantic Kernel **function-invocation filter** that intercepts calls to
designated functions, consults an approval callback, and blocks the call if it is denied.
The agent still decides to call the function; the filter is the gate.

```python
from semantic_kernel import Kernel
from human_in_the_loop import add_human_approval, requires_named

kernel = Kernel()
kernel.add_plugin(AdminPlugin(), plugin_name="admin")

def approve(function_name: str, args: dict) -> bool:
    # CLI prompt, Slack message, web approval queue - anything returning yes/no
    answer = input(f"Allow {function_name}({args})? [y/N] ")
    return answer.strip().lower() == "y"

add_human_approval(kernel, approve, requires_approval=requires_named("delete_account"))
```

When the agent invokes `delete_account`, the filter pauses and calls `approve`. Approve and
the function runs; deny and it never executes - the agent receives a denial result instead.
The approval callback can be sync or async, so a web approval queue works as naturally as a
terminal prompt.

## When to use it

- Irreversible or high-impact tool calls (delete, pay, send, deploy).
- Compliance requirements that mandate human sign-off.
- Rolling out a new agent: gate everything first, then relax as you build trust.

## Gotchas

- **Gate by capability, not by name alone.** `requires_named(...)` is the simple case. For
  real systems, also inspect the arguments - approve `refund(amount=5)` automatically but
  require sign-off for `refund(amount=5000)`.
- **Denial is a result, not a crash.** The agent gets the denial message back and continues.
  Write the denial text so the model handles it gracefully ("tell the user this needs
  approval"), not so it retries the same call in a loop.
- **Approval latency.** A blocking `input()` is fine for a CLI; a web agent needs an async
  callback that parks the request on a queue and resumes when a human responds. The async
  callback support here is for exactly that.
- **Audit everything.** Log every approve/deny decision with who, what, and the arguments.
  The filter logs at INFO/WARNING - route those to your audit sink.

## Run the test

```bash
pytest patterns/07_human_in_the_loop
```
