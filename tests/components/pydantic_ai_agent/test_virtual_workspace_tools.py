"""Test virtual workspace tools."""

from typing import Any, cast

import pytest

from custom_components.pydantic_ai_agent.const import CONF_VIRTUAL_WORKSPACE_ENABLED
from custom_components.pydantic_ai_agent.virtual_workspace import (
    virtual_workspace_enabled,
)
from custom_components.pydantic_ai_agent.virtual_workspace import (
    workspace as workspace_module,
)
from custom_components.pydantic_ai_agent.virtual_workspace.const import (
    MAX_COMMAND_BYTES,
)
from custom_components.pydantic_ai_agent.virtual_workspace.tools import (
    build_virtual_workspace_toolset,
)
from custom_components.pydantic_ai_agent.virtual_workspace.workspace import (
    VirtualWorkspace,
)


async def test_bash_restores_previous_cwd() -> None:
    """Test bash commands run in requested cwd without leaking cwd changes."""
    workspace = VirtualWorkspace()
    workspace.create_directory("/workspace/sub")
    await workspace._tool.execute("cd /tmp")

    result = await workspace.bash("pwd", working_directory="/workspace/sub")

    assert result["ok"] is True
    assert result["stdout"].strip() == "/workspace/sub"
    assert workspace._tool.shell_state().cwd == "/tmp"


async def test_bash_rolls_back_when_command_exceeds_workspace_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test shell-created files cannot bypass the aggregate workspace cap."""
    workspace = VirtualWorkspace()
    monkeypatch.setattr(workspace_module, "MAX_WORKSPACE_BYTES", 5)

    result = await workspace.bash("printf 123456 > big.txt")

    assert result["ok"] is False
    assert "workspace size limit" in result["error"]
    assert workspace.read_file("big.txt")["ok"] is False


async def test_bash_rolls_back_when_execution_raises() -> None:
    """Test failed shell execution restores partial VFS mutations."""
    workspace = VirtualWorkspace()

    class ExecuteFailingTool:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            self.call_count = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        async def execute(self, commands: str, on_output: Any | None = None) -> Any:
            self.call_count += 1
            if self.call_count == 1:
                self.wrapped.write_file("/workspace/partial.txt", "content")
                raise RuntimeError("boom")
            return await self.wrapped.execute(commands, on_output=on_output)

    workspace._tool = cast(Any, ExecuteFailingTool(workspace._tool))

    result = await workspace.bash("boom")

    assert result["ok"] is False
    assert result["error"] == "boom"
    assert workspace.read_file("partial.txt")["ok"] is False


async def test_bash_rejects_oversize_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test command size is enforced before shell execution."""
    workspace = VirtualWorkspace()
    monkeypatch.setattr(workspace_module, "MAX_COMMAND_BYTES", 3)

    result = await workspace.bash("1234")

    assert result["ok"] is False
    assert "command exceeds" in result["error"]


def test_read_file_truncates_with_utf8_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test byte-limit truncation does not silently drop split UTF-8 bytes."""
    workspace = VirtualWorkspace()
    workspace.write_file("unicode.txt", "é")
    monkeypatch.setattr(workspace_module, "MAX_READ_BYTES", 1)

    result = workspace.read_file("unicode.txt")

    assert result["truncated"] is True
    assert result["content"] == "�"


def test_workspace_constructor_uses_no_mounts_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Bashkit is created without host mounts or network access."""
    calls: list[dict[str, Any]] = []

    class FakeBashTool:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def mkdir(self, path: str, *, recursive: bool = False) -> None:
            assert path == "/workspace"
            assert recursive is True

    monkeypatch.setattr(
        "custom_components.pydantic_ai_agent.virtual_workspace.workspace.BashTool",
        FakeBashTool,
    )

    VirtualWorkspace()

    assert calls == [{"max_memory": 16777216, "timeout_seconds": 10.0, "network": None}]


def test_file_operations_require_confirmation_for_overwrite() -> None:
    """Test file writes default to create-only and confirm destructive writes."""
    workspace = VirtualWorkspace()

    created = workspace.write_file("notes.txt", "one")
    blocked = workspace.write_file("notes.txt", "two", overwrite=True)
    replaced = workspace.write_file("notes.txt", "two", overwrite=True, confirm=True)
    read = workspace.read_file("notes.txt")

    assert created["ok"] is True
    assert blocked["ok"] is False
    assert "confirm=true" in blocked["error"]
    assert replaced["ok"] is True
    assert read["content"] == "two"


def test_write_file_enforces_total_workspace_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test repeated writes cannot exceed the aggregate workspace cap."""
    monkeypatch.setattr(workspace_module, "MAX_WORKSPACE_BYTES", 5)
    workspace = VirtualWorkspace()

    first = workspace.write_file("one.txt", "123")
    blocked = workspace.write_file("two.txt", "123")

    assert first["ok"] is True
    assert blocked["ok"] is False
    assert "workspace size limit" in blocked["error"]
    assert workspace.read_file("two.txt")["ok"] is False


def test_directory_pagination_is_name_sorted() -> None:
    """Test directory reads page entries by stable name sort."""
    workspace = VirtualWorkspace()
    workspace.write_file("b.txt", "b")
    workspace.write_file("a.txt", "a")
    workspace.write_file("c.txt", "c")

    first = workspace.read_directory("/workspace", limit=2)
    second = workspace.read_directory("/workspace", cursor=first["nextCursor"], limit=2)

    assert [entry["name"] for entry in first["entries"]] == ["a.txt", "b.txt"]
    assert first["nextCursor"] == "2"
    assert [entry["name"] for entry in second["entries"]] == ["c.txt"]


def test_remove_refuses_unconfirmed_and_protected_paths() -> None:
    """Test remove requires confirmation and refuses workspace roots."""
    workspace = VirtualWorkspace()
    workspace.write_file("delete.txt", "content")

    unconfirmed = workspace.remove("delete.txt")
    protected = workspace.remove("/workspace", recursive=True, confirm=True)
    removed = workspace.remove("delete.txt", confirm=True)

    assert unconfirmed["ok"] is False
    assert protected["ok"] is False
    assert removed["ok"] is True


def test_copy_and_move_require_confirmed_overwrite() -> None:
    """Test copy and move enforce destination overwrite gates."""
    workspace = VirtualWorkspace()
    workspace.write_file("source.txt", "source")
    workspace.write_file("destination.txt", "destination")

    blocked = workspace.copy("source.txt", "destination.txt", overwrite=True)
    copied = workspace.copy(
        "source.txt", "destination.txt", overwrite=True, confirm=True
    )
    moved = workspace.move("source.txt", "moved.txt", confirm=True)

    assert blocked["ok"] is False
    assert "confirm=true" in blocked["error"]
    assert copied["ok"] is True
    assert moved["ok"] is True
    assert workspace.read_file("moved.txt")["content"] == "source"


def test_move_requires_confirmation_without_overwrite() -> None:
    """Test moving a path is destructive and requires confirmation."""
    workspace = VirtualWorkspace()
    workspace.write_file("source.txt", "source")

    result = workspace.move("source.txt", "moved.txt")

    assert result["ok"] is False
    assert "confirm=true" in result["error"]
    assert workspace.read_file("source.txt")["content"] == "source"
    assert workspace.read_file("moved.txt")["ok"] is False


def test_copy_overwrite_failure_restores_destination() -> None:
    """Test copy restores an existing destination if backend copy fails."""
    workspace = VirtualWorkspace()
    workspace.write_file("source.txt", "source")
    workspace.write_file("destination.txt", "destination")

    class FailingCopyTool:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        def fs(self) -> Any:
            class FailingCopyFS:
                def __init__(self, wrapped: Any) -> None:
                    self.wrapped = wrapped

                def __getattr__(self, name: str) -> Any:
                    return getattr(self.wrapped, name)

                def copy(self, source: str, destination: str) -> None:
                    raise RuntimeError("copy failed")

            return FailingCopyFS(self.wrapped.fs())

    workspace._tool = cast(Any, FailingCopyTool(workspace._tool))

    result = workspace.copy(
        "source.txt", "destination.txt", overwrite=True, confirm=True
    )

    assert result["ok"] is False
    assert result["error"] == "copy failed"
    assert workspace.read_file("source.txt")["content"] == "source"
    assert workspace.read_file("destination.txt")["content"] == "destination"


def test_move_overwrite_failure_restores_source_and_destination() -> None:
    """Test move restores source and destination if backend rename fails."""
    workspace = VirtualWorkspace()
    workspace.write_file("source.txt", "source")
    workspace.write_file("destination.txt", "destination")

    class FailingRenameTool:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        def fs(self) -> Any:
            class FailingRenameFS:
                def __init__(self, wrapped: Any) -> None:
                    self.wrapped = wrapped

                def __getattr__(self, name: str) -> Any:
                    return getattr(self.wrapped, name)

                def rename(self, source: str, destination: str) -> None:
                    raise RuntimeError("rename failed")

            return FailingRenameFS(self.wrapped.fs())

    workspace._tool = cast(Any, FailingRenameTool(workspace._tool))

    result = workspace.move(
        "source.txt", "destination.txt", overwrite=True, confirm=True
    )

    assert result["ok"] is False
    assert result["error"] == "rename failed"
    assert workspace.read_file("source.txt")["content"] == "source"
    assert workspace.read_file("destination.txt")["content"] == "destination"


def test_copy_enforces_total_workspace_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test copying files cannot exceed the aggregate workspace cap."""
    monkeypatch.setattr(workspace_module, "MAX_WORKSPACE_BYTES", 5)
    workspace = VirtualWorkspace()
    workspace.write_file("source.txt", "123")

    copied = workspace.copy("source.txt", "copy.txt")

    assert copied["ok"] is False
    assert "workspace size limit" in copied["error"]
    assert workspace.read_file("copy.txt")["ok"] is False


def test_copy_and_move_missing_source_preserve_destination() -> None:
    """Test confirmed overwrite does not delete destination when source is invalid."""
    workspace = VirtualWorkspace()
    workspace.write_file("destination.txt", "destination")

    copied = workspace.copy(
        "missing.txt", "destination.txt", overwrite=True, confirm=True
    )
    moved = workspace.move(
        "missing.txt", "destination.txt", overwrite=True, confirm=True
    )

    assert copied["ok"] is False
    assert moved["ok"] is False
    assert workspace.read_file("destination.txt")["content"] == "destination"


def test_copy_and_move_same_path_preserve_source() -> None:
    """Test same-path operations fail without deleting the source."""
    workspace = VirtualWorkspace()
    workspace.write_file("same.txt", "content")

    copied = workspace.copy("same.txt", "same.txt", overwrite=True, confirm=True)
    moved = workspace.move("same.txt", "same.txt", overwrite=True, confirm=True)

    assert copied["ok"] is False
    assert moved["ok"] is False
    assert workspace.read_file("same.txt")["content"] == "content"


def test_copy_and_move_reject_directory_descendant_destination() -> None:
    """Test directory operations cannot target their own descendants."""
    workspace = VirtualWorkspace()
    workspace.create_directory("dir")
    workspace.write_file("dir/file.txt", "content")

    copied = workspace.copy("dir", "dir/sub")
    moved = workspace.move("dir", "dir/sub", confirm=True)

    assert copied["ok"] is False
    assert moved["ok"] is False
    assert workspace.read_file("dir/file.txt")["content"] == "content"


def test_tool_wrapper_requires_boolean_confirm_values() -> None:
    """Test malformed string confirmations do not satisfy destructive gates."""
    workspace = VirtualWorkspace()
    workspace.write_file("delete.txt", "content")
    toolset = build_virtual_workspace_toolset(workspace)

    remove = cast(Any, toolset.tools["remove"].function)
    result = remove(path="delete.txt", confirm="false")

    assert result["ok"] is False
    assert workspace.read_file("delete.txt")["content"] == "content"


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (True, True),
        (False, False),
        (None, False),
        ("true", False),
        ("false", False),
        (1, False),
    ],
)
def test_virtual_workspace_enabled_requires_literal_true(
    value: object,
    enabled: bool,
) -> None:
    """Test persisted truthy non-bool values do not enable workspace tools."""
    assert virtual_workspace_enabled({CONF_VIRTUAL_WORKSPACE_ENABLED: value}) is enabled
    assert virtual_workspace_enabled({}) is False


def test_read_directory_tool_returns_structured_limit_error() -> None:
    """Test malformed pagination limits return structured tool failures."""
    workspace = VirtualWorkspace()
    toolset = build_virtual_workspace_toolset(workspace)

    read_directory = cast(Any, toolset.tools["readDirectory"].function)
    result = read_directory(path="/workspace", limit="abc")

    assert result == {
        "ok": False,
        "path": "/workspace",
        "entries": [],
        "error": "limit must be an integer",
    }


def test_toolset_exposes_exact_camel_case_tool_schemas() -> None:
    """Test public virtual workspace tools use requested names and schemas."""
    toolset = build_virtual_workspace_toolset(VirtualWorkspace())

    assert list(toolset.tools) == [
        "bash",
        "readFile",
        "writeFile",
        "createDirectory",
        "getMetadata",
        "readDirectory",
        "remove",
        "copy",
        "move",
        "applyPatch",
    ]
    assert set(toolset.tools["bash"].function_schema.json_schema["properties"]) == {
        "command",
        "workingDirectory",
    }
    assert (
        toolset.tools["bash"].function_schema.json_schema["properties"]["command"][
            "maxLength"
        ]
        == MAX_COMMAND_BYTES
    )
    assert set(toolset.tools["readFile"].function_schema.json_schema["properties"]) == {
        "path"
    }
    assert set(
        toolset.tools["applyPatch"].function_schema.json_schema["properties"]
    ) == {
        "patch",
        "confirm",
    }
    assert toolset.tools["move"].function_schema.json_schema["required"] == [
        "source",
        "destination",
        "confirm",
    ]
