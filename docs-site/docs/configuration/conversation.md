---
title: Configure a conversation agent
---

Add a Conversation subentry and configure its name, model profile, instruction prompt, and optional capabilities. Select a Home Assistant LLM API to expose Home Assistant control tools. Select remote MCP servers, Skills, web fetch/search, or a virtual workspace only when the agent needs them.

Conversation entities are exposed as `conversation.*` and are registered by [`conversation.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/conversation.py). Model settings include limits, timeout, retries, streaming, and provider-supported capability controls.
