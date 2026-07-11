---
title: Manual installation
---

1. Download the repository at the desired release.
2. Copy `custom_components/pydantic_ai_agent` into your Home Assistant `custom_components` directory.
3. Restart Home Assistant.
4. Add the integration from **Settings → Devices & services**.

Do not copy the repository's development environment into Home Assistant. Home Assistant installs the pinned integration requirements from [`manifest.json`](https://github.com/bradsjm/hacs-pydantic-ai-agent/blob/main/custom_components/pydantic_ai_agent/manifest.json).
