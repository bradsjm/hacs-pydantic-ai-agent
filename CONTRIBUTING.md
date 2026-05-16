# Contributing

## Development Setup

Use Python 3.14.2 or newer and install the locked development environment:

```bash
uv sync --locked --group dev
```

## Validation

Run these checks before opening a pull request:

```bash
uv run ruff check custom_components/pydantic_ai_agent tests/components/pydantic_ai_agent
uv run pytest --timeout=10 tests/components/pydantic_ai_agent
```

## Release Checklist

1. Update `custom_components/.../manifest.json` version.
2. Update `pyproject.toml` version.
3. Update `CHANGELOG.md`.
4. Run the validation commands.
5. Create a GitHub release whose tag matches the manifest version, for example `v0.1.0`.
