---
title: Services
---

The integration registers seven services. Exact fields and selectors are authoritative in [`services.yaml`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/services.yaml).

| Service | Purpose |
| --- | --- |
| `list_mcp_tools` | List cached tools, discovering them when no cache exists |
| `refresh_mcp_tools` | Reconnect and refresh MCP catalogs |
| `get_agent_run_diagnostics` | Return a compact latest-run diagnostic slice |
| `get_workspace_status` | Return workspace, subentry, and runtime status |
| `list_model_profiles` | List configured profiles without provider probing |
| `get_agent_metrics` | Return the in-memory metrics snapshot |
| `get_tool_source_status` | Return cached MCP and Skill source status |

Services belong to the workspace config entry. Diagnostics and status calls can expose operational metadata, so protect access to Home Assistant service calls.
