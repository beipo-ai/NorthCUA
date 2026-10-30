import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PUBLIC_RUNTIME_TEMPLATE_KEYS = re.compile(
    r"\{\{\s*(?:date|username|working_directory|operating_system|platform|shell_tool_backend|shell_tool_guidance|env_content)\b"
)


def test_shipped_prompts_do_not_consume_public_only_runtime_context() -> None:
    prompt_files = tuple(
        path
        for path in (_ROOT / "northcua").rglob("*")
        if path.is_file() and path.suffix in {".j2", ".jinja", ".jinja2", ".yaml", ".yml"}
    )

    matches = [
        str(path.relative_to(_ROOT))
        for path in prompt_files
        if _PUBLIC_RUNTIME_TEMPLATE_KEYS.search(path.read_text(encoding="utf-8"))
    ]

    assert matches == []


def test_shipped_agent_yaml_does_not_enable_public_permissions_or_subagents() -> None:
    yaml_files = tuple(path for path in (_ROOT / "northcua").rglob("*.yaml") if path.is_file())
    forbidden = re.compile(r"(?m)^\s*(?:permissions|plugins|sub_agents)\s*:")

    matches = [
        str(path.relative_to(_ROOT)) for path in yaml_files if forbidden.search(path.read_text(encoding="utf-8"))
    ]

    assert matches == []
