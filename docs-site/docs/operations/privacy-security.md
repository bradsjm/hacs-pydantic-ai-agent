---
title: Privacy and security
---

- Provider credentials and secret headers stay on provider/server subentries and are redacted from diagnostics.
- Prompt and completion capture in Logfire is opt-in and disabled by default.
- Remote MCP uses Streamable HTTP only, validates endpoints, and does not run local processes.
- Web fetch is optional and includes SSRF protection; outbound access is still a data-handling decision.
- Virtual workspace data is per-run in memory, has no host mounts, and has no network access.
- Skills are guidance, not executable extensions.

Grant capabilities narrowly. A model with Home Assistant control, MCP, web, and workspace tools has a larger action and data surface than a plain conversation agent.
