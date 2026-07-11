---
title: Entities and exposed attributes
---

Conversation subentries create `conversation.*` entities. AI task subentries create `ai_task.*` entities that support Home Assistant data generation and attachments. The integration also exposes diagnostic sensors and binary sensors for runtime and capability state.

Entity attributes summarize resolved model profiles and enabled capabilities such as structured output, web tools, and virtual workspace. Treat attributes as informational and JSON-serializable; use the services page for bounded diagnostics and metrics.
