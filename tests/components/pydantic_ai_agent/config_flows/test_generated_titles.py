from collections.abc import Iterator

from custom_components.pydantic_ai_agent.config_flows import generated_titles


def test_generated_default_title_capitalizes_words_and_suffix(
    monkeypatch,
) -> None:
    monkeypatch.setattr(generated_titles, "generate_name", lambda style: "ada_lovelace")

    assert generated_titles.generated_default_title("Agent") == "Ada Lovelace Agent"


def test_generated_default_title_falls_back_to_suffix_when_name_blank(
    monkeypatch,
) -> None:
    monkeypatch.setattr(generated_titles, "generate_name", lambda style: "")

    assert generated_titles.generated_default_title("Workspace") == "Workspace"


def test_generated_default_title_avoids_case_insensitive_collisions(
    monkeypatch,
) -> None:
    names: Iterator[str] = iter(["ada", "grace-hopper"])
    monkeypatch.setattr(generated_titles, "generate_name", lambda style: next(names))

    title = generated_titles.generated_default_title("Agent", ["ADA Agent"])

    assert title == "Grace Hopper Agent"


def test_generated_default_title_uses_available_name_after_collisions(
    monkeypatch,
) -> None:
    names: Iterator[str] = iter(["agent-1", "agent-2", "available-agent"])
    monkeypatch.setattr(generated_titles, "generate_name", lambda style: next(names))

    existing = ["Agent 1 Service", "Agent 2 Service"]

    title = generated_titles.generated_default_title("Service", existing)

    assert title == "Available Agent Service"
