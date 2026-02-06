"""Build a Semantic Kernel chat completion service from environment variables.

Every pattern uses this so the Azure wiring lives in one place. Real runs read
credentials from the environment; tests inject a fake service instead (see
``sk_cookbook.testing``).
"""
import os

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

ENV_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_CHAT_DEPLOYMENT = "AZURE_OPENAI_CHAT_DEPLOYMENT"
ENV_API_VERSION = "AZURE_OPENAI_API_VERSION"

# Pin the Azure OpenAI REST API version explicitly. Semantic Kernel's built-in
# default can lag behind newer models (e.g. gpt-4.1-mini), and an older api-version
# makes the deployment path return 404 Resource not found. This value is recent
# enough to serve current models; override with AZURE_OPENAI_API_VERSION if needed.
DEFAULT_API_VERSION = "2025-01-01-preview"


class MissingConfigError(RuntimeError):
    """Raised when a required Azure OpenAI environment variable is not set."""


def build_chat_service(
    service_id: str = "default",
    api_version: str | None = None,
) -> AzureChatCompletion:
    """Create an AzureChatCompletion from environment variables.

    Required environment variables:
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT
    Optional:
        AZURE_OPENAI_API_VERSION (else DEFAULT_API_VERSION is used)

    :param service_id: logical id for the service, used when a kernel holds more
        than one (e.g. the cost-aware routing pattern registers two).
    :param api_version: Azure OpenAI REST API version. Precedence: this argument >
        AZURE_OPENAI_API_VERSION env var > DEFAULT_API_VERSION. Pinning this avoids
        404s when SK's built-in default predates the model you deployed.
    :raises MissingConfigError: if any required variable is absent.
    """
    endpoint = os.environ.get(ENV_ENDPOINT)
    api_key = os.environ.get(ENV_API_KEY)
    deployment = os.environ.get(ENV_CHAT_DEPLOYMENT)
    resolved_api_version = api_version or os.environ.get(ENV_API_VERSION) or DEFAULT_API_VERSION

    missing = [
        name
        for name, value in (
            (ENV_ENDPOINT, endpoint),
            (ENV_API_KEY, api_key),
            (ENV_CHAT_DEPLOYMENT, deployment),
        )
        if not value
    ]
    if missing:
        raise MissingConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    return AzureChatCompletion(
        service_id=service_id,
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        api_version=resolved_api_version,
    )
