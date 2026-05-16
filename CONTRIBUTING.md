# Contributing

## Development Setup

Use Python 3.14.2 or newer, Node.js/npm, and install the locked development environment:

```bash
scripts/setup
```

## Validation

Run these checks before opening a pull request:

```bash
scripts/check
```

## Release Checklist

1. Update `custom_components/.../manifest.json` version.
2. Update `pyproject.toml` version.
3. Update `CHANGELOG.md`.
4. Run `scripts/check`.
5. Create a GitHub release whose tag matches the manifest version, for example `v0.1.0`.
