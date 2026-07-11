---
title: Diagnostics and repairs
---

The integration provides redacted config-entry diagnostics, system health information, runtime metrics, and repair handling. Diagnostics include provider and subentry summaries while protecting credentials and secret headers. Use `get_agent_run_diagnostics` for a compact slice of a conversation or AI task run.

The implementation is in [`diagnostics.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/diagnostics.py), [`system_health.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/system_health.py), and [`repair_issues.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/repair_issues.py). Do not paste unredacted diagnostics into public issues.
