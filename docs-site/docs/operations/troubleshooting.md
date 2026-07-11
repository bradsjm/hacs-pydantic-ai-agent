---
title: Troubleshooting
---

### The provider cannot be reached

Check the mode, base URL, headers, credential, and Home Assistant outbound network access. Use manual model entry if discovery is unavailable, then inspect the provider validation result and logs.

### Tools are missing

Confirm the capability is selected on the Conversation or AI task subentry. For MCP, verify the server URL, refresh its catalog, and check the selected `all`, `specified`, or `disabled` mode.

### Context or structured output differs from expectation

Check the resolved model profile capabilities. Context mode controls only the model request, and structured output resolves among `tool`, `native`, and `prompted` based on support.

For a bounded view of a recent run, use `get_agent_run_diagnostics` and select a section such as `errors`, `timeline`, or `model_profile`.
