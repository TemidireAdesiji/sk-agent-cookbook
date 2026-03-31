"""Tests for pattern 07: human in the loop."""
from human_in_the_loop import ApprovalCallback, add_human_approval, requires_named
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments, kernel_function


class AdminPlugin:
    """A plugin with one sensitive function and one safe function."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.reads = 0

    @kernel_function(name="delete_account", description="Delete a user account.")
    async def delete_account(self, user_id: str) -> str:
        self.deleted.append(user_id)
        return f"Deleted {user_id}"

    @kernel_function(name="read_account", description="Read a user account.")
    async def read_account(self, user_id: str) -> str:
        self.reads += 1
        return f"Account {user_id}"


def _kernel_with_plugin() -> tuple[Kernel, AdminPlugin]:
    kernel = Kernel()
    plugin = AdminPlugin()
    kernel.add_plugin(plugin, plugin_name="admin")
    return kernel, plugin


def _gate(kernel: Kernel, approve: ApprovalCallback, **kwargs: object) -> None:
    """Register approval gating for the sensitive delete_account function."""
    add_human_approval(
        kernel,
        approve=approve,
        requires_approval=requires_named("delete_account"),
        **kwargs,  # type: ignore[arg-type]
    )


async def _invoke(kernel: Kernel, function_name: str, user_id: str) -> object:
    return await kernel.invoke(
        plugin_name="admin",
        function_name=function_name,
        arguments=KernelArguments(user_id=user_id),
    )


async def test_denied_action_does_not_run() -> None:
    kernel, plugin = _kernel_with_plugin()
    _gate(kernel, approve=lambda name, args: False)

    result = await _invoke(kernel, "delete_account", "u1")

    assert plugin.deleted == []  # the function body never ran
    assert "denied" in str(result).lower()


async def test_approved_action_runs() -> None:
    kernel, plugin = _kernel_with_plugin()
    _gate(kernel, approve=lambda name, args: True)

    result = await _invoke(kernel, "delete_account", "u1")

    assert plugin.deleted == ["u1"]
    assert "Deleted u1" in str(result)


async def test_non_sensitive_function_is_not_gated() -> None:
    kernel, plugin = _kernel_with_plugin()
    # Deny everything that asks - but read_account should never ask
    _gate(kernel, approve=lambda name, args: False)

    await _invoke(kernel, "read_account", "u9")
    assert plugin.reads == 1  # ran without approval


async def test_approval_callback_receives_function_name_and_args() -> None:
    kernel, plugin = _kernel_with_plugin()
    seen: dict[str, object] = {}

    def approve(name: str, args: dict[str, object]) -> bool:
        seen["name"] = name
        seen["args"] = args
        return True

    _gate(kernel, approve=approve)
    await _invoke(kernel, "delete_account", "u42")

    assert seen["name"] == "delete_account"
    assert seen["args"]["user_id"] == "u42"  # type: ignore[index]


async def test_async_approval_callback() -> None:
    kernel, plugin = _kernel_with_plugin()

    async def approve(name: str, args: dict[str, object]) -> bool:
        return True

    _gate(kernel, approve=approve)
    await _invoke(kernel, "delete_account", "u7")
    assert plugin.deleted == ["u7"]


async def test_custom_denial_message() -> None:
    kernel, plugin = _kernel_with_plugin()
    _gate(kernel, approve=lambda name, args: False, denial_message="Needs manager sign-off.")

    result = await _invoke(kernel, "delete_account", "u1")
    assert "manager sign-off" in str(result)
