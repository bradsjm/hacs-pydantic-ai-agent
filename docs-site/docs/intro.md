---
title: Pydantic AI Agent
slug: /
description: Pydantic AI-powered conversation agents and AI tasks for Home Assistant.
---

Pydantic AI Agent is a Home Assistant custom integration for Assist conversation agents and AI task data generation. It connects configurable provider models to Home Assistant capabilities, remote MCP tools, Skills, web tools, and an optional per-run virtual workspace.

The current integration version is **0.6.0** and requires Home Assistant **2026.6.4 or newer**. The supported provider modes are `openai_compatible_completions`, `openai_compatible_responses`, `anthropic`, and `google_gemini` (see the [provider constants](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/const.py)).

## Where to begin

- [Install with HACS](installation/hacs) or [install manually](installation/manual).
- Follow the [quickstart](quickstart) to create a workspace, provider, and entity.
- Read [workspaces and subentries](concepts/workspaces-subentries) before adding more capabilities.
- Use [troubleshooting](operations/troubleshooting) when a provider, tool, or entity does not behave as expected.

This site documents the checked-in integration. It does not promise local MCP processes, runtime package installation, or shared memory between workspaces.
