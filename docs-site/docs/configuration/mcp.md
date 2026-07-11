---
title: Configure remote MCP
---

MCP subentries support **remote Streamable HTTP only**. Configure the server URL, optional headers, timeout, caching, deferred loading, and tool exposure mode. Tool modes are `all`, `specified`, and `disabled`; the specified mode uses an allowlist.

The integration validates URLs and headers, constrains redirects to the validated origin, and prefixes tool names per server. It does not launch local MCP processes or install packages at runtime. See [`mcp/validation.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/mcp/validation.py) and [`mcp/client.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/mcp/client.py).
