---
title: Contributing
---

For integration changes, keep Home Assistant lifecycle, async I/O, redaction, and per-entry runtime ownership intact. Add behavioral coverage when changing user-visible behavior, and keep schemas and translations aligned with flows.

For documentation changes, edit Markdown under `docs-site/docs/`, update `sidebars.ts` when adding pages, and run the docs build. Link important behavior to the relevant source file. Avoid documenting experimental or absent components as available.

Open pull requests against `main` with a focused description and the validation commands you ran. Issues and source changes belong in the [GitHub repository](https://github.com/bradsjm/hacs-pydantic-ai-agent).
