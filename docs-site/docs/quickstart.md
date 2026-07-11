---
title: Quickstart
---

1. Install the integration and restart Home Assistant.
2. Open **Settings → Devices & services**, add **Pydantic AI Agent**, and create a workspace.
3. Add a **provider** subentry. Select one of the four provider modes, enter its credentials, and create or select a model profile.
4. Add a **Conversation** subentry for Assist, or an **AI task** subentry for data generation.
5. Select a model profile and save. The integration creates a `conversation.*` or `ai_task.*` entity.
6. For Home Assistant control, select an available Home Assistant LLM API in the entity configuration. Add remote MCP, Skills, web tools, or a virtual workspace only when needed.

Start with a model and no optional tools. Add one capability at a time so failures are easy to isolate. The [configuration guides](configuration/providers) describe each option without relying on incidental UI labels.
