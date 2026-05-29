# Virtual Workspace Agent Instructions

## Scope

These instructions apply to
`custom_components/pydantic_ai_agent/virtual_workspace`.

## Agent Focus

- Treat this package as a per-run, in-memory workspace for LLM tool use.
- Preserve the sandbox boundary: no host filesystem mounts and no network.
- Keep every tool result JSON-serializable.
- Keep destructive behavior explicitly gated by `confirm=true`.
- Do not add persistent storage, host command execution, package installation,
  Git access, or network access without an explicit design and user approval.

## Read First

- `__init__.py` - feature flag check and per-run toolset factory.
- `const.py` - command, read, write, patch, output, directory, and workspace size
  limits.
- `workspace.py` - Bashkit-backed virtual filesystem helpers and rollback
  behavior.
- `paths.py` - virtual path normalization and protected replacement paths.
- `tools.py` - Pydantic AI `FunctionToolset` and exact JSON schemas.
- `patch.py` - Codex-style patch parser and atomic patch application.
- `models.py` - JSON-serializable tool result shapes.
- `errors.py` - expected internal exception types.

## Invariants

- `virtual_workspace_parts()` must create a fresh `VirtualWorkspace` for each
  model run. Workspace state must not persist across runs.
- `build_virtual_workspace_toolset()` must keep `FunctionToolset(sequential=True)`
  because tools mutate shared in-memory state.
- `BashTool` must be constructed without host mounts and with `network=None`.
- `/workspace` is the default working directory, and `/` plus `/workspace` are
  protected from removal or replacement.
- `normalize_vfs_path()` must reject empty paths, NUL bytes, and `..` escapes
  beyond the virtual root.
- Bash commands must restore cwd and roll back filesystem mutations on Bashkit
  exceptions or workspace size limit failures.
- Write, copy, move, remove, and patch operations must enforce the limits in
  `const.py` and return structured error results instead of leaking exceptions.
- `applyPatch` returns `ToolReturn` metadata with
  `TOOL_RETURN_METADATA_SOURCE`; downstream stream handling depends on it.

## High-Risk Changes

- Relaxing confirmation gates can let the model delete or overwrite virtual
  files without explicit acknowledgement.
- Relaxing path handling can allow writes outside the virtual root if Bashkit
  changes its filesystem behavior.
- Removing rollback can leave partial state after Bashkit exceptions, workspace
  size limit failures, copy errors, move errors, or patch errors. A normal
  nonzero shell exit can still leave virtual files changed.
- Parallelizing the toolset can corrupt workspace state because operations share
  one `VirtualWorkspace` instance.
- Increasing limits affects Home Assistant memory use because the integration
  runs in the HA process.

## Validation

- Run `scripts/test -k virtual_workspace` for package changes.
- Run `scripts/test -k test_virtual_workspace_patch` for patch parser changes.
- Run `scripts/test -k test_virtual_workspace_paths` for path handling changes.
- Run `scripts/test -k "conversation or ai_task"` when toolset wiring or
  `virtual_workspace_enabled()` changes.
- Run `scripts/lint-check` when changing imports, schemas, or TypedDicts.
