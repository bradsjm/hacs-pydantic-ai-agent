# Virtual Workspace

`virtual_workspace` provides optional per-run tools that let an LLM work inside a
temporary in-memory filesystem. The workspace is backed by `bashkit.BashTool`,
has `/workspace` as the default working directory, has no host filesystem mounts,
and has no network access.

All state is discarded after the model run that created the workspace.

## Runtime Flow

- Runtime code checks `virtual_workspace_enabled(data)` on a conversation or AI
  task subentry.
- `virtual_workspace_parts()` creates a fresh `VirtualWorkspace` and a Pydantic
  AI `FunctionToolset` for one run.
- The toolset is registered as `sequential=True` so Pydantic AI does not run
  mutating workspace tools in parallel.
- Tool calls mutate only per-run virtual state, including the in-memory virtual
  filesystem and shell state.
- The run ends and the workspace object is discarded.

## Tools

- `bash` - run a command in the virtual shell.
- `readFile` - read a virtual file.
- `writeFile` - write a virtual file, with overwrite confirmation when needed.
- `createDirectory` - create a virtual directory.
- `getMetadata` - inspect virtual path metadata.
- `readDirectory` - list a virtual directory with cursor pagination.
- `remove` - remove a virtual path after confirmation.
- `copy` - copy a virtual path, with overwrite confirmation when needed.
- `move` - move a virtual path after confirmation.
- `applyPatch` - apply a Codex-style patch atomically.

## Modules

- `__init__.py` - public factory, feature flag, and LLM instructions string.
- `const.py` - size, timeout, paging, and metadata constants.
- `workspace.py` - core `VirtualWorkspace` implementation and Bashkit wrapper.
- `paths.py` - lexical path normalization and protected path checks.
- `tools.py` - exact JSON schemas and Pydantic AI tool construction.
- `patch.py` - Codex patch parsing, add, update, delete, move, and rollback.
- `models.py` - TypedDict result contracts.
- `errors.py` - expected internal exceptions.

## Limits

- Command timeout: 10 seconds.
- Command input: 64 KiB.
- Stdout and stderr output: 64 KiB each.
- File read: 256 KiB.
- File write: 256 KiB.
- Patch input: 512 KiB.
- Directory page size: 100 by default and 500 maximum.
- Total workspace size: 16 MiB.

## Safety Behavior

- Paths are normalized inside `/` and relative paths resolve from `/workspace`.
- NUL bytes and attempts to escape above `/` are rejected.
- `/` and `/workspace` cannot be removed, moved, or replaced.
- Overwrites require `overwrite=true` and `confirm=true`.
- Remove and move require `confirm=true`.
- Patch updates, deletes, moves, and overwrites require `confirm=true`.
- Bash rolls back filesystem mutations on Bashkit exceptions and workspace size
  limit failures. Copy, move, and patch operations use snapshots to roll back
  operation errors.

## Testing

- `scripts/test -k virtual_workspace`
- `scripts/test -k test_virtual_workspace_tools`
- `scripts/test -k test_virtual_workspace_patch`
- `scripts/test -k test_virtual_workspace_paths`

Runtime wiring is also covered by conversation and AI task tests under
`tests/components/pydantic_ai_agent/`.
