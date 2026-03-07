"""Pattern 03: Tool retry with fallback.

Agent tools (plugin functions) often call flaky things - third-party APIs, networks,
rate-limited services. A single transient failure should not break the whole agent turn.
This pattern wraps a tool operation with bounded retries and exponential backoff, then a
graceful fallback so the agent always gets a usable answer instead of an exception.

``call_with_retry`` is the reusable core. ``ResilientToolPlugin`` shows how to expose a
resilient operation as a ``@kernel_function`` an agent can call.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ToolUnavailableError(RuntimeError):
    """Raised when an operation exhausts its retries and no fallback is configured."""


async def call_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    fallback: Callable[[], Awaitable[T] | T] | None = None,
) -> T:
    """Call ``operation`` with retries, exponential backoff, and an optional fallback.

    :param operation: zero-arg async callable performing the work.
    :param max_attempts: total attempts before giving up (>= 1).
    :param base_delay: seconds for the first backoff; doubles each retry.
    :param retry_on: exception types that trigger a retry. Others propagate immediately.
    :param fallback: zero-arg callable (sync or async) producing a fallback result when all
        attempts fail. If None, the last exception is wrapped in ToolUnavailableError.
    :raises ToolUnavailableError: if all attempts fail and no fallback is given.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except retry_on as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    logger.error("All %d attempts failed; last error: %s", max_attempts, last_exc)
    if fallback is not None:
        result = fallback()
        if isinstance(result, Awaitable):
            return await result
        return result
    raise ToolUnavailableError(f"Operation failed after {max_attempts} attempts") from last_exc


class ResilientToolPlugin:
    """A plugin whose tool wraps a flaky operation in retry + fallback.

    :param fetch: the underlying async operation, ``fetch(query) -> str``. Inject the real
        API call in production; inject a fake in tests.
    :param max_attempts: retries passed to ``call_with_retry``.
    :param base_delay: backoff base passed to ``call_with_retry`` (set 0 in tests).
    """

    def __init__(
        self,
        fetch: Callable[[str], Awaitable[str]],
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
    ) -> None:
        self._fetch = fetch
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    @kernel_function(name="lookup", description="Look up information for a query.")
    async def lookup(self, query: str) -> str:
        """Resilient tool: retries the fetch, then returns a safe fallback message."""
        return await call_with_retry(
            lambda: self._fetch(query),
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            fallback=lambda: f"Information for '{query}' is temporarily unavailable.",
        )
