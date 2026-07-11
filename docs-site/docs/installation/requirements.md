---
title: Requirements
---

- Home Assistant **2026.6.4 or newer**.
- A provider account and credentials appropriate to the selected mode.
- Network access from Home Assistant to the provider endpoint. Remote MCP servers and web tools also require outbound access when enabled.
- HACS is optional; manual installation is supported.

The integration pins its runtime dependencies in [`manifest.json`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/manifest.json). Provider credentials are stored on provider subentries and are not read from environment variables.
