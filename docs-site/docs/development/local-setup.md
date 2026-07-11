---
title: Local development setup
---

The project targets Python **3.14.2 or newer** and uses `uv`-style scripts for its development environment.

```bash
scripts/setup
```

Runtime dependencies are pinned in both `pyproject.toml` and the integration manifest. The docs site is independent:

```bash
cd docs-site
pnpm install
pnpm start
```

Keep docs changes source-grounded: check the integration code and `services.yaml` before adding a claim.
