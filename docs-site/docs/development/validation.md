---
title: Validation
---

Run the repository checks from its root:

```bash
scripts/check
```

Build the static documentation site from `docs-site/`:

```bash
pnpm install --frozen-lockfile
pnpm build
```

The build uses `CI=1`, treats broken links as errors, and writes only generated output to `docs-site/build/`. Do not commit `build/`, `.docusaurus/`, or `node_modules/`.
