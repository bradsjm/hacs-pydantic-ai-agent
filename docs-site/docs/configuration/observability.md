---
title: Configure observability
---

Workspace settings can enable Logfire tracing with a token. Prompt and completion content capture is disabled by default; enable it only when that data handling is acceptable. Traces include Home Assistant identifiers such as entry, subentry, entity, model, and conversation IDs.

Logfire configuration is process-wide in Home Assistant. The first loaded workspace with a token wins; a later different token does not emit traces and produces a repair warning. Runtime diagnostics and in-memory metrics are also available through the [debug services](../reference/services). See [`observability/logfire_support.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/observability/logfire_support.py).
