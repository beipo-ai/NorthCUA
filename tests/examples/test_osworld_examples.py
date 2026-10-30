from __future__ import annotations

from pathlib import Path

from nexau.archs.main_sub.utils import load_yaml_with_vars

from northcua.agents.desktop.prompt import PromptEnvelopeBuilder
from northcua.core.config import NorthCUAAppConfig
from northcua.core.models import ObservationContext, WindowState
from northcua.runtime.mcp.registry import NorthCUAMCPRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
OSWORLD_EXAMPLES = REPO_ROOT / "examples" / "osworld"


def test_osworld_app_example_is_not_a_top_level_runtime_config() -> None:
    assert not (REPO_ROOT / "config" / "app.osworld.yaml").exists()
    assert not (REPO_ROOT / "config" / "agent.osworld.yaml").exists()

    app_config = NorthCUAAppConfig.from_yaml(OSWORLD_EXAMPLES / "app.osworld.example.yaml")

    assert app_config.agent_config_path == "agent.osworld.example.yaml"
    assert app_config.desktop.backend == "osworld"
    assert app_config.desktop.osworld.vm_ip == "127.0.0.1"
    assert app_config.execution.max_agent_iterations_per_round == 50
    assert app_config.prompt.screenshot_window_images == 3
    assert app_config.prompt.history_window_rounds == 10000


def test_osworld_app_example_drives_osworld_prompt_guidance() -> None:
    app_config = NorthCUAAppConfig.from_yaml(OSWORLD_EXAMPLES / "app.osworld.example.yaml")
    builder = PromptEnvelopeBuilder(
        template_root=REPO_ROOT / "northcua" / "agents" / "desktop" / "prompt" / "templates",
        policy=app_config.prompt,
    )

    envelope = builder.build(
        task="Open Chrome",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        observation_context=ObservationContext(
            platform_name="osworld",
            observation_metadata={"desktop_environment": "osworld"},
        ),
        tools=[],
        tool_choice="auto",
    )

    assert "Desktop environment: osworld" in envelope.system_prompt
    assert "OSWorld benchmark environment:" in envelope.system_prompt
    assert "Do not ask the user for clarification." in envelope.system_prompt
    assert "emit `[INFEASIBLE]` and stop" in envelope.system_prompt
    assert "If terminal use is required" in envelope.system_prompt
    assert "be careful with large direct pastes" in envelope.system_prompt
    assert "Unsafe Paste prompts" in envelope.system_prompt


def test_osworld_agent_example_is_parameterized(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = load_yaml_with_vars(OSWORLD_EXAMPLES / "agent.osworld.example.yaml")
    assert config["llm_config"]["model"] == "test-model"
    assert config["llm_config"]["api_key"] == "test-key"
    assert all(server["type"] == "in-memory" for server in config["mcp_servers"])


def test_osworld_action_executor_uses_shared_input_tool_descriptors(monkeypatch) -> None:
    """OSWorld action tools should expose the same type/key guidance as desktop."""

    import northcua.runtime.mcp.desktop_native_server  # noqa: F401

    monkeypatch.setenv("NORTHCUA_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = load_yaml_with_vars(OSWORLD_EXAMPLES / "agent.osworld.example.yaml")
    visible_action_servers = [server["name"] for server in config["mcp_servers"] if server.get("model_visible") is True]
    schemas = {
        descriptor.tool_id: descriptor.parameters_schema
        for descriptor in NorthCUAMCPRegistry.wire_tool_descriptors("DesktopNativeActionExecutor")
    }

    assert visible_action_servers == ["DesktopNativeActionExecutor"]
    type_text = str(schemas["type"]["properties"]["text"]["description"])
    keypress = schemas["keypress"]
    assert "Unsafe Paste confirmation prompts" in type_text
    assert "opening a file, writing the content there" in type_text
    assert keypress["oneOf"] == [
        {
            "properties": {
                "key": {"type": "string"},
                "keys": {"type": "null"},
            },
            "required": ["key"],
        },
        {
            "properties": {
                "key": {"type": "null"},
                "keys": {"items": {"type": "string"}, "type": "array"},
            },
            "required": ["keys"],
        },
    ]
    assert "Use this only when `keys` is omitted" in str(keypress["properties"]["key"]["description"])
    assert '["ctrl+l", "ctrl+l", "tab"]' in str(keypress["properties"]["keys"]["description"])
