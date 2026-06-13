# Changelog

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
