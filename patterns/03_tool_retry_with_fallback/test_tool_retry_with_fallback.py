"""Tests for pattern 03: tool retry with fallback."""
import pytest
from tool_retry_with_fallback import (
    ResilientToolPlugin,
    ToolUnavailableError,
    call_with_retry,
)


class FlakyOperation:
    """Fails the first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times: int, result: str = "ok") -> None:
        self.fail_times = fail_times
        self.result = result
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError(f"transient failure {self.calls}")
        return self.result


async def test_succeeds_first_try() -> None:
    op = FlakyOperation(fail_times=0)
    result = await call_with_retry(op, base_delay=0)
    assert result == "ok"
    assert op.calls == 1


async def test_retries_then_succeeds() -> None:
    op = FlakyOperation(fail_times=2)
    result = await call_with_retry(op, max_attempts=3, base_delay=0)
    assert result == "ok"
    assert op.calls == 3  # failed twice, succeeded on the third


async def test_exhausts_retries_and_uses_fallback() -> None:
    op = FlakyOperation(fail_times=10)
    result = await call_with_retry(
        op, max_attempts=3, base_delay=0, fallback=lambda: "fallback value"
    )
    assert result == "fallback value"
    assert op.calls == 3


async def test_async_fallback_is_awaited() -> None:
    async def afallback() -> str:
        return "async fallback"

    op = FlakyOperation(fail_times=10)
    result = await call_with_retry(op, max_attempts=2, base_delay=0, fallback=afallback)
    assert result == "async fallback"


async def test_raises_when_no_fallback() -> None:
    op = FlakyOperation(fail_times=10)
    with pytest.raises(ToolUnavailableError):
        await call_with_retry(op, max_attempts=2, base_delay=0)


async def test_non_retryable_exception_propagates_immediately() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        await call_with_retry(op, max_attempts=5, base_delay=0, retry_on=(ConnectionError,))
    assert calls == 1  # not retried


async def test_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        await call_with_retry(FlakyOperation(0), max_attempts=0)


async def test_plugin_returns_value_on_success() -> None:
    async def fetch(query: str) -> str:
        return f"data for {query}"

    plugin = ResilientToolPlugin(fetch, base_delay=0)
    assert await plugin.lookup(query="weather") == "data for weather"


async def test_plugin_returns_fallback_message_on_persistent_failure() -> None:
    async def fetch(query: str) -> str:
        raise TimeoutError("upstream down")

    plugin = ResilientToolPlugin(fetch, max_attempts=2, base_delay=0)
    result = await plugin.lookup(query="weather")
    assert "temporarily unavailable" in result
    assert "weather" in result


async def test_plugin_retries_underlying_fetch() -> None:
    attempts = 0

    async def fetch(query: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("blip")
        return "recovered"

    plugin = ResilientToolPlugin(fetch, max_attempts=3, base_delay=0)
    assert await plugin.lookup(query="x") == "recovered"
    assert attempts == 2
