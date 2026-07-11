---
title: Context, web tools, and virtual workspace
---

### Context

Choose `context_manager`, `sliding_window`, or `off` for model-request context handling. The stored history remains intact.

### Web tools

Web fetch and web search are independent options. Search uses provider-native capability when available or the integration's DuckDuckGo fallback through Pydantic AI. Enable these only for agents that need outbound web access.

### Virtual workspace

The optional per-run workspace is an in-memory Bashkit filesystem rooted at `/workspace`, with no host mounts and no network access. It provides shell and file tools; destructive operations require explicit confirmation and overwrites require an overwrite flag. Source: [`virtual_workspace/tools.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/virtual_workspace/tools.py).
