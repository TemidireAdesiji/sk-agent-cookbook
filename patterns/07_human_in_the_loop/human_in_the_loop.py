"""Pattern 07: Human in the loop.

Some actions an agent can take are irreversible or sensitive - deleting data, sending money,
emailing a customer. You want the agent to propose the action but a human to approve it
before it runs. This pattern uses a Semantic Kernel function-invocation filter to intercept
calls to designated functions, ask an approval callback, and block the call if it is denied.

The approval callback is injected, so it can be a CLI prompt, a Slack message, a web
approval queue - anything that returns a yes/no.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from semantic_kernel import Kernel
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext
from semantic_kernel.functions import FunctionResult

logger = logging.getLogger(__name__)

# An approval callback receives (function_name, arguments) and returns True to allow.
# It may be sync or async.
ApprovalCallback = Callable[[str, dict[str, object]], bool | Awaitable[bool]]

# Decides whether a given function invocation needs human approval.
RequiresApproval = Callable[[FunctionInvocationContext], bool]


def requires_named(*function_names: str) -> RequiresApproval:
    """Build a predicate that requires approval for the named functions."""
    targets = set(function_names)

    def predicate(context: FunctionInvocationContext) -> bool:
        return context.function.name in targets

    return predicate


def add_human_approval(
    kernel: Kernel,
    approve: ApprovalCallback,
    requires_approval: RequiresApproval,
    denial_message: str = "Action denied by human reviewer.",
) -> None:
    """Register a function-invocation filter that gates sensitive functions on approval.

    When a function matching ``requires_approval`` is invoked, ``approve`` is consulted with
    the function name and its arguments. If it returns False, the function does not run and
    its result is set to ``denial_message``.

    :param kernel: the kernel whose function invocations are gated.
    :param approve: callback returning True to allow the call (sync or async).
    :param requires_approval: predicate selecting which invocations need approval.
    :param denial_message: the result returned when approval is refused.
    """

    async def approval_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        if not requires_approval(context):
            await next(context)
            return

        arguments = dict(context.arguments) if context.arguments else {}
        decision = approve(context.function.name, arguments)
        approved = await decision if isinstance(decision, Awaitable) else decision

        if approved:
            logger.info("Approved: %s(%s)", context.function.name, arguments)
            await next(context)
        else:
            logger.warning("Denied: %s(%s)", context.function.name, arguments)
            context.result = FunctionResult(
                function=context.function.metadata, value=denial_message
            )

    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, approval_filter)
