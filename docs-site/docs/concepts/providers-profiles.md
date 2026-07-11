---
title: Providers and model profiles
---

The four provider modes are:

- `openai_compatible_completions`
- `openai_compatible_responses`
- `anthropic`
- `google_gemini`

OpenAI-compatible modes use the integration's in-repository `OpenAICompatibleChatModel`, `OpenAICompatibleResponsesModel`, and `OpenAICompatibleProvider` rather than the OpenAI SDK. Each provider owns a stable map of model profiles, including capability and pricing metadata used by entity configuration and structured output selection. See [`models/provider.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/models/provider.py) and [`models/model_profiles.py`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/models/model_profiles.py).

Model discovery is provider-specific. If discovery fails, manual model entry remains available; listing profiles does not probe providers (see [services](../reference/services)).
