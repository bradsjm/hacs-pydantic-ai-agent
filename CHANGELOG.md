# Changelog

## 0.5.0

- Upgrade runtime and development dependencies, including Pydantic AI 2.0, Home Assistant 2026.6.4, Logfire 4.37.0, and related provider SDKs.
- Update the OpenAI-compatible adapter for Pydantic AI 2.0 model profile APIs.
- Rebuild the strategic unit test suite across models, agent runtime, MCP, virtual workspace, config-flow helpers, and observability logic.
- Refresh the smoke-test hook for the new logical test layout.

## 0.4.0

- Add model-aware context management for conversation and task runs.
- Add web search support with a DuckDuckGo fallback provider.
- Add MCP tool mode configuration and update all-mode defaults.
- Improve MCP and model discovery flows with progress reporting and simplified setup validation.
- Persist OpenAI-compatible model capability profiles and restore saved model pricing in edit forms.
- Redact secret HTTP headers during provider reconfiguration.

## 0.3.0

- Add MCP call-result caching plus a last-MCP-tool sensor for workspace runtime visibility.
- Add secret header controls for provider and MCP configuration so only marked header values are redacted in diagnostics and shared outputs.
- Refresh workflow and local toolchain dependencies used by validation and development.

## 0.2.1

- Add a conversation streaming toggle that can disable progressive responses per agent.
- Move the conversation streaming toggle into Run settings while keeping existing agents streaming by default.
- Fix setup validation, repair handling, and runtime execution so conversation streaming settings are applied consistently.

## 0.2.0

- Add provider-owned model profile management, discovery, validation, and fallback selection flows.
- Add Home Assistant AI task entities with structured output modes, attachments, and optional todo workspace tools.
- Add native Skill subentries plus optional Web fetch support for conversation agents and AI tasks.
- Add runtime metrics, diagnostics, system health, provider repair issues, and optional Logfire tracing.
- Add Anthropic, Google Gemini, and OpenAI-compatible Responses provider support alongside the in-repo OpenAI-compatible clients.

## 0.1.0

- Initial HACS custom-repository
