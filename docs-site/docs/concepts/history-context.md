---
title: History and context management
---

`ChatLog` is the canonical conversation history. Automatic context management changes the model request, not the stored Assist history: long conversations can be summarized or windowed without deleting the original record.

The three context modes are `context_manager`, `sliding_window`, and `off`. Conversation agents default to context-manager behavior and AI tasks to sliding-window behavior; both can be disabled. The request conversion and context logic live in [`agent/context_management.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/agent/context_management.py).
