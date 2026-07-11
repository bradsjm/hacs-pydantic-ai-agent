---
title: Configuration defaults
---

Defaults are defined in [`const.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/const.py) and can be overridden by subentry/profile settings.

| Setting | Default |
| --- | --- |
| MCP timeout | 10 seconds |
| MCP call-cache TTL | 300 seconds |
| Context window fallback | 32,768 tokens |
| General model timeout | 60 seconds |
| Tool retries | 3 |
| AI task iterations | 30 unless a profile overrides it |

MCP tool mode, context mode, and structured output mode are explicit enumerations documented in the relevant configuration guides.
