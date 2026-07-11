---
title: Agents and tool execution
---

Conversation and AI task entities run a Pydantic AI agent with a resolved model profile and a `ChatLog` request history. Tools are opt-in: Home Assistant LLM APIs, remote MCP servers, Skills, web fetch/search, and the virtual workspace are selected per subentry.

Conversation streaming is controlled by the subentry's `streaming_enabled` setting. The conversation platform passes that setting to the shared runtime; tool selection itself does not establish a separate streaming or provider-compatibility guarantee. See [`conversation.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/conversation.py) and [`entity.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/entity.py). There is no process-global agent or shared conversation memory between entries.
