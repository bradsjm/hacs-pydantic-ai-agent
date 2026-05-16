# Pydantic AI Agent

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)

Home Assistant custom integration scaffold for Assist conversation agents backed by
Pydantic AI.

## Status

This repository currently contains the provider/configuration foundation. Assist
conversation entities can be configured and exposed as distinct Home Assistant
conversation agents, but the Pydantic AI chat runtime is not implemented yet. A
configured agent returns a placeholder response until runtime support is added.

## Installation

### HACS Custom Repository

1. Open HACS in Home Assistant.
2. Go to Integrations, then Custom repositories.
3. Add `https://github.com/bradsjm/hacs-pydantic-agent` as an Integration.
4. Install `Pydantic AI Agent`.
5. Restart Home Assistant.

### Manual

1. Copy `custom_components/pydantic_ai_agent` into your Home Assistant
   `custom_components` directory.
2. Restart Home Assistant.

## Configuration

1. Go to Settings > Devices & services.
2. Add `Pydantic AI Agent`.
3. Configure a provider connection with an OpenAI or OpenAI-compatible API key.
4. Add conversation-agent subentries for each Assist agent you want to expose.

Each Assist agent is represented by its own `conversation.*` entity. That entity
ID is the Home Assistant conversation agent ID.

### Logfire Tracing

Provider setup includes optional Logfire tracing fields. Leave the Logfire token
blank to disable Logfire for that provider connection. When a token is provided,
the integration adds Home Assistant metadata such as entry, subentry, entity,
model, and conversation IDs to Pydantic AI traces.

The `Include prompt and response content in Logfire` option is disabled by
default. Enable it only if you want Logfire to capture prompt, completion, and
tool payload content. Logfire is configured process-wide in Home Assistant: the
first loaded provider entry with a token wins, later entries with a different
token are left loaded but get a repair warning and do not emit Logfire traces.

## Development

Use Python 3.14.2 or newer.

```bash
scripts/setup
scripts/check
```

## Support

Issues: <https://github.com/bradsjm/hacs-pydantic-agent/issues>

Code owner: `@bradsjm`
