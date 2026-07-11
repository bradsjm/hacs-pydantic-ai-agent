---
title: Configure Skills
---

A Skill subentry stores reusable guidance as raw content. Conversation and AI task subentries can select Skills; at runtime they are exposed through `list_skills` and `load_skill` rather than automatically inserted into every request.

Skill content cannot override system, Home Assistant, developer, or safety instructions. Skills do not execute scripts, clone repositories, update themselves, or read filesystem folders. References are surfaced in diagnostics. See [`agent/skills.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/agent/skills.py).
