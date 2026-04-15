"""Generate the high-level architecture diagram for sk-agent-cookbook.

This models the real structure of the cookbook:

- A host application / agent uses ``sk_cookbook.build_chat_service`` to build an
  ``AzureChatCompletion`` service, which talks to Azure OpenAI.
- The patterns build ``ChatCompletionAgent`` instances on top of that service.
- Pattern 06 (RAG) additionally uses a ``DocumentRetriever`` - either the in-memory
  fake (demos/tests) or Azure AI Search (production).
- Tests substitute ``ScriptedChatService`` for the live service, so no Azure is needed.

Requirements:
    pip install diagrams
    # plus the Graphviz "dot" binary, e.g.:
    #   Debian/Ubuntu:  sudo apt-get install graphviz
    #   macOS:          brew install graphviz

Run:
    python scripts/architecture.py
    # writes assets/architecture.png

The output path is fixed at <repo>/assets/architecture.png regardless of the
working directory you run it from.
"""
from __future__ import annotations

import logging
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.azure.ml import CognitiveServices
from diagrams.azure.web import Search
from diagrams.onprem.client import User
from diagrams.programming.language import Python

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
OUTPUT_STEM = ASSETS_DIR / "architecture"


def build_diagram() -> None:
    """Render the component diagram to assets/architecture.png."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    graph_attr = {"fontsize": "20", "bgcolor": "white", "pad": "0.4"}

    with Diagram(
        "sk-agent-cookbook",
        filename=str(OUTPUT_STEM),
        show=False,
        outformat="png",
        direction="LR",
        graph_attr=graph_attr,
    ):
        host = User("Host app / agent")

        with Cluster("sk_cookbook (shared layer)"):
            chat_service = Python("build_chat_service\n-> AzureChatCompletion")
            test_fake = Python("ScriptedChatService\n(tests, no Azure)")

        with Cluster("patterns/ (01-07)"):
            agents = Python("ChatCompletionAgent(s)")
            retriever = Python("DocumentRetriever\n(InMemory | Azure AI Search)")

        azure_openai = CognitiveServices("Azure OpenAI")
        azure_search = Search("Azure AI Search")

        host >> Edge(label="compose") >> agents
        agents >> Edge(label="uses") >> chat_service
        chat_service >> Edge(label="REST (pinned api-version)") >> azure_openai

        test_fake >> Edge(style="dashed", label="substitutes in tests") >> agents

        agents >> Edge(label="pattern 06") >> retriever
        retriever >> Edge(style="dashed", label="production") >> azure_search

    logger.info("Wrote %s.png", OUTPUT_STEM)


if __name__ == "__main__":
    build_diagram()
