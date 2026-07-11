---
title: Configure AI tasks
---

Add an AI task subentry for Home Assistant data generation. An AI task can return plain text or validate structured output requested by Home Assistant. It can select the same model profiles, Home Assistant APIs, MCP servers, Skills, web tools, and virtual workspace as a conversation agent.

Structured output resolves to the highest supported strategy in this order: `tool`, `native`, then `prompted`. AI tasks default to 30 request iterations unless the selected profile overrides that value. The entity platform is implemented in [`ai_task.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/ai_task.py).
