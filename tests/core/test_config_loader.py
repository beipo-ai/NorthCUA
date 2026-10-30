from __future__ import annotations

from pathlib import Path

from northcua.agentinfra.providers.llm_fallback import llm_fallback_candidates
from northcua.core.config import NorthCUAAppConfig
from northcua.core.config_loader import NorthCUAConfigLoader


def test_config_loader_reads_dual_config_files(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    agent_yaml = config_dir / "agent.yaml"
    agent_yaml.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
system_prompt: You are NorthCUA.
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
mcp_servers:
  - name: DesktopNativeDataExecutor
    type: stdio
    model_visible: false
    command: python
  - name: DesktopNativeA11yExecutor
    type: stdio
    model_visible: false
    command: python
  - name: DesktopNativeActionExecutor
    type: stdio
    model_visible: true
    command: python
""".strip(),
        encoding="utf-8",
    )
    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text(
        """
agent_config_path: agent.yaml
websocket:
  host: 0.0.0.0
  port: 9999
desktop:
  backend: osworld
  osworld:
    vm_ip: 10.0.0.10
    server_port: 6100
  coordinate_policy:
    input_space: normalized_1000
""".strip(),
        encoding="utf-8",
    )

    bootstrap = NorthCUAConfigLoader.from_app_config(app_yaml)
    assert bootstrap.app.websocket.port == 9999
    assert bootstrap.app.desktop.backend == "osworld"
    assert bootstrap.app.desktop.osworld.vm_ip == "10.0.0.10"
    assert bootstrap.app.desktop.osworld.server_port == 6100
    assert bootstrap.app.desktop.coordinate_policy.input_space.value == "normalized_1000"
    assert bootstrap.agent.name == "NorthCUAAgent"
    assert [server["name"] for server in bootstrap.app.agent_mcp_servers] == [
        "DesktopNativeDataExecutor",
        "DesktopNativeA11yExecutor",
        "DesktopNativeActionExecutor",
    ]
    assert bootstrap.agent.mcp_servers == []
    assert bootstrap.agent.system_prompt is None


def test_config_loader_strips_desktop_owned_system_prompt_fields(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    agent_yaml = config_dir / "agent.yaml"
    agent_yaml.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
system_prompt:
  - file://northcua/agents/desktop/prompt/templates/computer_using_agent_system.j2
system_prompt_type: jinja
system_prompt_suffix: Extra suffix that should not survive.
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
mcp_servers:
  - name: DesktopNativeActionExecutor
    type: stdio
    model_visible: true
    command: python
""".strip(),
        encoding="utf-8",
    )
    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text(
        """
agent_config_path: agent.yaml
websocket:
  host: 0.0.0.0
  port: 9999
""".strip(),
        encoding="utf-8",
    )

    bootstrap = NorthCUAConfigLoader.from_app_config(app_yaml)

    assert bootstrap.agent.system_prompt is None
    assert bootstrap.agent.system_prompt_suffix is None


def test_config_loader_lifts_llm_max_context_tokens_to_agent_budget(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    agent_yaml = config_dir / "agent.yaml"
    agent_yaml.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
  max_context_tokens: 200000
mcp_servers:
  - name: DesktopNativeActionExecutor
    type: stdio
    model_visible: true
    command: python
""".strip(),
        encoding="utf-8",
    )
    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text("agent_config_path: agent.yaml", encoding="utf-8")

    bootstrap = NorthCUAConfigLoader.from_app_config(app_yaml)

    assert bootstrap.agent.max_context_tokens == 200000
    assert "max_context_tokens" not in bootstrap.agent.llm_config.extra_params


def test_config_loader_attaches_llm_fallbacks(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    agent_yaml = config_dir / "agent.yaml"
    agent_yaml.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
llm_config:
  model: claude-opus-4-6
  base_url: https://gateway.example.invalid/v1
  api_key: key
  api_type: anthropic_chat_completion
  provider_profile: claude-compatible-gateway
  image_transport: anthropic_base64
  transcode_images_to_jpeg: true
llm_fallbacks:
  - name: sonnet
    model: claude-sonnet-4-6
  - name: gpt
    inherit: false
    llm_config:
      model: gpt-5.4
      base_url: https://openai.example.invalid/v1
      api_key: openai-key
      api_type: openai_responses
mcp_servers:
  - name: DesktopNativeActionExecutor
    type: stdio
    model_visible: true
    command: python
""".strip(),
        encoding="utf-8",
    )
    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text("agent_config_path: agent.yaml", encoding="utf-8")

    bootstrap = NorthCUAConfigLoader.from_app_config(app_yaml)
    candidates = llm_fallback_candidates(bootstrap.agent)

    assert [candidate.name for candidate in candidates] == ["claude-opus-4-6", "sonnet", "gpt"]
    assert candidates[1].llm_config.model == "claude-sonnet-4-6"
    assert candidates[1].llm_config.extra_params["provider_profile"] == "claude-compatible-gateway"
    assert candidates[2].llm_config.base_url == "https://openai.example.invalid/v1"
    assert candidates[2].llm_config.api_type == "openai_responses"
