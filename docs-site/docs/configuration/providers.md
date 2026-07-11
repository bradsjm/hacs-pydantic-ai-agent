---
title: Configure providers
---

Create a provider subentry inside a workspace, select its mode, and enter the provider credential. OpenAI-compatible providers default to `https://api.openai.com/v1` when no base URL is entered; custom endpoints can be configured. Provider headers are used for discovery and model requests.

Select or manually enter a model, then review its provider-owned profile. Conversation and AI task subentries refer to that profile rather than storing credentials themselves. Keep credentials on the provider subentry and redact them from support output.
