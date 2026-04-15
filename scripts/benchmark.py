"""Benchmark the app-independent hot paths of the cookbook.

This is a library of agent patterns, so a meaningful end-to-end benchmark is not
possible without a live model (and would mostly measure Azure latency, not this code).
Instead this measures the parts that are entirely ours and deterministic:

1. ``heuristic_classifier`` from pattern 04 (cost-aware routing) - the cheap/capable
   routing decision over representative inputs.
2. ``keyword_router`` from pattern 02 (multi-agent handoff) - the specialist routing
   decision over representative inputs.
3. ChatHistory serialize/restore round-trip from pattern 01 (memory persistence) - the
   cost of persisting and reloading a conversation.

Where an agent call is needed, the fake ``ScriptedChatService`` is used, so this script
needs NO live Azure credentials and makes NO network calls.

Output is a Markdown table written to STDOUT only.

Requirements:
    pip install -e ".[dev]"

Run:
    python scripts/benchmark.py
    # pipe straight into the README benchmarks section:
    python scripts/benchmark.py | python scripts/inject_readme_section.py --section benchmarks
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Make the shared package and the pattern modules importable when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "patterns" / "01_memory_persistence"))
sys.path.insert(0, str(REPO_ROOT / "patterns" / "02_multi_agent_handoff"))
sys.path.insert(0, str(REPO_ROOT / "patterns" / "04_cost_aware_routing"))

from cost_aware_routing import heuristic_classifier  # noqa: E402
from multi_agent_handoff import keyword_router  # noqa: E402
from semantic_kernel.contents import ChatHistory, ChatMessageContent  # noqa: E402
from semantic_kernel.contents.utils.author_role import AuthorRole  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# Representative routing inputs (mix of simple and complex).
SAMPLE_MESSAGES: tuple[str, ...] = (
    "What time is it in Lagos?",
    "Explain why the sky is blue, step by step.",
    "Reset my password please.",
    "Analyze the trade-off between latency and cost for this design.",
    "Thanks, that worked!",
    "Debug this stack trace and refactor the handler.",
    "I was charged twice on my invoice this month.",
    "Compare and contrast these two approaches.",
)


def _time_calls(fn: Callable[[str], object], inputs: tuple[str, ...], iterations: int) -> float:
    """Return operations per second for ``fn`` over ``inputs`` repeated ``iterations`` times."""
    start = time.perf_counter()
    for _ in range(iterations):
        for message in inputs:
            fn(message)
    elapsed = time.perf_counter() - start
    total_ops = iterations * len(inputs)
    return total_ops / elapsed if elapsed else float("inf")


def _bench_heuristic_classifier(iterations: int = 20_000) -> float:
    classify = heuristic_classifier()
    return _time_calls(classify, SAMPLE_MESSAGES, iterations)


def _bench_keyword_router(iterations: int = 20_000) -> float:
    route = keyword_router(
        {
            "billing": ("invoice", "charged", "refund", "payment"),
            "technical": ("password", "stack trace", "debug", "error"),
        },
        default="general",
    )
    return _time_calls(route, SAMPLE_MESSAGES, iterations)


def _build_history(turns: int) -> ChatHistory:
    history = ChatHistory()
    for i in range(turns):
        history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content=f"User message number {i}.")
        )
        history.add_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, content=f"Assistant reply number {i}.")
        )
    return history


def _bench_history_round_trip(turns: int = 25, iterations: int = 2_000) -> float:
    """Operations per second for one serialize + restore round-trip of a conversation."""
    history = _build_history(turns)
    start = time.perf_counter()
    for _ in range(iterations):
        blob = history.serialize()
        ChatHistory.restore_chat_history(blob)
    elapsed = time.perf_counter() - start
    return iterations / elapsed if elapsed else float("inf")


def main() -> None:
    """Run the benchmarks and print a Markdown table to STDOUT."""
    logger.info("Running benchmarks (no live Azure required)...")

    classifier_ops = _bench_heuristic_classifier()
    router_ops = _bench_keyword_router()
    round_trip_ops = _bench_history_round_trip()

    rows = [
        (
            "Pattern 04: heuristic_classifier (route decision)",
            f"{classifier_ops:,.0f} ops/s",
        ),
        (
            "Pattern 02: keyword_router (route decision)",
            f"{router_ops:,.0f} ops/s",
        ),
        (
            "Pattern 01: ChatHistory serialize + restore (25-turn convo)",
            f"{round_trip_ops:,.0f} round-trips/s",
        ),
    ]

    # STDOUT only: the Markdown table.
    print("| Operation | Throughput |")
    print("|---|---|")
    for label, value in rows:
        print(f"| {label} | {value} |")
    print()
    print(
        "_Measured locally on pure-Python / fake-backed paths; no live Azure calls. "
        "Numbers vary by machine._"
    )


if __name__ == "__main__":
    main()
