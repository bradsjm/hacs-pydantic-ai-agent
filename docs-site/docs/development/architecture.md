---
title: Architecture
---

Home Assistant owns config-entry lifecycle and forwards Conversation and AI task platforms. Provider subentries build models through [`models/provider.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/models/provider.py). The shared entity runner resolves profiles, converts `ChatLog` history, assembles HA/MCP/Skill/web/workspace tools, and records metrics and diagnostics.

The flow is intentionally per workspace and per subentry: runtime state is not a global agent or shared memory store. MCP discovery and calls live under `mcp/`; optional Bashkit tools live under `virtual_workspace/`; observability is under `observability/`. Setup and unload are owned by [`__init__.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/__init__.py).
