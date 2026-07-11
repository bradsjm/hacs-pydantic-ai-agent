---
title: Workspaces and subentries
---

A workspace is the config entry that owns shared runtime coordination and optional Logfire settings. It can contain five subentry types:

| Type | Purpose |
| --- | --- |
| Provider | Credentials, mode, headers, and model profiles |
| Conversation | An Assist conversation entity |
| AI task | An AI task data-generation entity |
| MCP server | A remote Streamable HTTP server |
| Skill | Reusable model guidance |

Conversation and AI task subentries reference provider-owned profiles with workspace-local refs shaped like `<provider_subentry_id>:<model_profile_id>`. The flow implementation defines the supported subentry types in [`workspace_flow.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/config_flows/workspace_flow.py).
