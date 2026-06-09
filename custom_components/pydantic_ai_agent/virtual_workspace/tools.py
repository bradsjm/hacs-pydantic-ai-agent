"""Pydantic AI tools for the virtual workspace."""

from collections.abc import Callable
from typing import Any

from pydantic_ai import Tool, ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from .const import MAX_COMMAND_BYTES, TOOL_RETURN_METADATA_SOURCE
from .patch import apply_patch
from .workspace import VirtualWorkspace


def build_virtual_workspace_toolset(
    workspace: VirtualWorkspace,
) -> FunctionToolset[None]:
    """Return the virtual workspace tools as an exact-schema toolset."""
    toolset: FunctionToolset[None] = FunctionToolset(sequential=True)
    for tool in (
        _tool(
            "bash",
            "Run a command in the in-memory virtual bash workspace.",
            _bash_schema(),
            _bash(workspace),
        ),
        _tool(
            "readFile",
            "Read a file from the virtual workspace.",
            _read_file_schema(),
            _read_file(workspace),
        ),
        _tool(
            "writeFile",
            "Write a file in the virtual workspace.",
            _write_file_schema(),
            _write_file(workspace),
        ),
        _tool(
            "createDirectory",
            "Create a directory in the virtual workspace.",
            _create_directory_schema(),
            _create_directory(workspace),
        ),
        _tool(
            "getMetadata",
            "Get metadata for a virtual workspace path.",
            _metadata_schema(),
            _metadata(workspace),
        ),
        _tool(
            "readDirectory",
            "List a virtual workspace directory.",
            _read_directory_schema(),
            _read_directory(workspace),
        ),
        _tool(
            "remove",
            "Remove a virtual workspace path after confirmation.",
            _remove_schema(),
            _remove(workspace),
        ),
        _tool(
            "copy", "Copy a virtual workspace path.", _copy_schema(), _copy(workspace)
        ),
        _tool(
            "move",
            "Move a virtual workspace path after confirmation.",
            _move_schema(),
            _move(workspace),
        ),
        _tool(
            "applyPatch",
            "Apply a Codex-style patch to the virtual workspace.",
            _apply_patch_schema(),
            _apply_patch(workspace),
        ),
    ):
        toolset.add_tool(tool)
    return toolset


def _tool(
    name: str,
    description: str,
    schema: dict[str, Any],
    function: Callable[..., object],
) -> Tool[None]:
    return Tool.from_schema(
        function,
        name=name,
        description=description,
        json_schema=schema,
        takes_ctx=False,
    )


def _bash(workspace: VirtualWorkspace) -> Callable[..., object]:
    async def execute(**tool_args: object) -> object:
        return await workspace.bash(
            _string_arg(tool_args, "command"),
            working_directory=_optional_string_arg(tool_args, "workingDirectory"),
        )

    return execute


def _read_file(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.read_file(_string_arg(tool_args, "path"))

    return execute


def _write_file(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.write_file(
            _string_arg(tool_args, "path"),
            _string_arg(tool_args, "content"),
            overwrite=_bool_arg(tool_args, "overwrite"),
            confirm=_bool_arg(tool_args, "confirm"),
        )

    return execute


def _create_directory(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.create_directory(
            _string_arg(tool_args, "path"),
            parents=_bool_arg(tool_args, "parents"),
        )

    return execute


def _metadata(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.metadata(_string_arg(tool_args, "path"))

    return execute


def _read_directory(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        try:
            limit = _int_arg(tool_args, "limit", 100)
        except ValueError as err:
            return {
                "ok": False,
                "path": _string_arg(tool_args, "path"),
                "entries": [],
                "error": str(err),
            }
        return workspace.read_directory(
            _string_arg(tool_args, "path"),
            cursor=_optional_string_arg(tool_args, "cursor"),
            limit=limit,
        )

    return execute


def _remove(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.remove(
            _string_arg(tool_args, "path"),
            recursive=_bool_arg(tool_args, "recursive"),
            confirm=_bool_arg(tool_args, "confirm"),
        )

    return execute


def _copy(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.copy(
            _string_arg(tool_args, "source"),
            _string_arg(tool_args, "destination"),
            overwrite=_bool_arg(tool_args, "overwrite"),
            confirm=_bool_arg(tool_args, "confirm"),
        )

    return execute


def _move(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return workspace.move(
            _string_arg(tool_args, "source"),
            _string_arg(tool_args, "destination"),
            overwrite=_bool_arg(tool_args, "overwrite"),
            confirm=_bool_arg(tool_args, "confirm"),
        )

    return execute


def _apply_patch(workspace: VirtualWorkspace) -> Callable[..., object]:
    def execute(**tool_args: object) -> object:
        return ToolReturn(
            apply_patch(
                workspace,
                _string_arg(tool_args, "patch"),
                confirm=_bool_arg(tool_args, "confirm"),
            ),
            metadata={"source": TOOL_RETURN_METADATA_SOURCE},
        )

    return execute


def _string_arg(tool_args: dict[str, object], key: str) -> str:
    value = tool_args.get(key, "")
    return value if isinstance(value, str) else ""


def _optional_string_arg(tool_args: dict[str, object], key: str) -> str | None:
    value = tool_args.get(key)
    return value if isinstance(value, str) else None


def _bool_arg(tool_args: dict[str, object], key: str) -> bool:
    return tool_args.get(key) is True


def _int_arg(tool_args: dict[str, object], key: str, default: int) -> int:
    value = tool_args.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        raise ValueError(f"{key} must be an integer") from err


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description: str, *, max_length: int | None = None) -> dict[str, str | int]:
    schema: dict[str, str | int] = {"type": "string", "description": description}
    if max_length is not None:
        schema["maxLength"] = max_length
    return schema


def _boolean(description: str) -> dict[str, str | bool]:
    return {"type": "boolean", "description": description}


def _bash_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "command": _string(
                "Command to execute in the virtual bash.",
                max_length=MAX_COMMAND_BYTES,
            ),
            "workingDirectory": _string(
                "Virtual working directory. Defaults to /workspace."
            ),
        },
        ["command"],
    )


def _read_file_schema() -> dict[str, Any]:
    return _object_schema({"path": _string("Virtual file path to read.")}, ["path"])


def _write_file_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "path": _string("Virtual file path to write."),
            "content": _string("File content to write."),
            "overwrite": _boolean("Set true to replace an existing file."),
            "confirm": _boolean("Required with overwrite=true."),
        },
        ["path", "content"],
    )


def _create_directory_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "path": _string("Virtual directory path to create."),
            "parents": _boolean("Create missing parent directories."),
        },
        ["path"],
    )


def _metadata_schema() -> dict[str, Any]:
    return _object_schema({"path": _string("Virtual path to inspect.")}, ["path"])


def _read_directory_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "path": _string("Virtual directory path to list."),
            "cursor": _string("Pagination cursor from a previous readDirectory call."),
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        ["path"],
    )


def _remove_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "path": _string("Virtual path to remove."),
            "recursive": _boolean("Remove directories recursively."),
            "confirm": _boolean("Must be true to remove anything."),
        },
        ["path", "confirm"],
    )


def _copy_schema() -> dict[str, Any]:
    return _copy_move_schema("Copy")


def _move_schema() -> dict[str, Any]:
    return _copy_move_schema("Move", confirm_required=True)


def _copy_move_schema(action: str, *, confirm_required: bool = False) -> dict[str, Any]:
    required = ["source", "destination"]
    if confirm_required:
        required.append("confirm")
    return _object_schema(
        {
            "source": _string(f"Virtual source path to {action.lower()}."),
            "destination": _string(
                f"Virtual destination path for the {action.lower()}."
            ),
            "overwrite": _boolean("Set true to replace an existing destination."),
            "confirm": _boolean(
                "Required for every move."
                if confirm_required
                else "Required with overwrite=true."
            ),
        },
        required,
    )


def _apply_patch_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "patch": _string("Codex-style patch envelope to apply."),
            "confirm": _boolean(
                "Required for updates, deletes, moves, and overwrites."
            ),
        },
        ["patch"],
    )
