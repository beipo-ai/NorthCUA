from __future__ import annotations

from pathlib import Path

import pytest
from nexau.archs.main_sub.config import AgentConfig, ConfigError
from nexau.archs.tool import Tool

from northcua.agentinfra.nexau_compat.adapters.config import NexAUConfigAdapter, StrictLegacyAgentConfigBuilder
from northcua.agentinfra.nexau_compat.adapters.tool import LegacyCompatibleTool


def _llm_config() -> dict[str, object]:
    return {
        "model": "test-model",
        "api_type": "openai_responses",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
    }


def test_builder_is_always_strict() -> None:
    builder = NexAUConfigAdapter.builder({}, Path.cwd())

    assert isinstance(builder, StrictLegacyAgentConfigBuilder)
    assert builder.strict is True


def test_strict_builder_rejects_invalid_tracer_instead_of_skipping() -> None:
    builder = NexAUConfigAdapter.builder({"tracers": [None]}, Path.cwd())

    with pytest.raises(ConfigError, match="Tracer entries cannot be null"):
        builder.build_tracers()


def test_strict_builder_rejects_missing_hook_instead_of_skipping() -> None:
    builder = NexAUConfigAdapter.builder(
        {"after_model_hooks": ["northcua.missing.module:hook"]},
        Path.cwd(),
    )

    with pytest.raises(ConfigError, match=r"Error loading hook 0"):
        builder.build_hooks()


def test_strict_builder_rejects_missing_system_prompt_instead_of_skipping(tmp_path) -> None:
    builder = NexAUConfigAdapter.builder(
        {
            "system_prompt": "missing.j2",
            "system_prompt_type": "jinja",
        },
        tmp_path,
    )
    builder.build_core_properties()

    with pytest.raises(ConfigError, match=r"System prompt file not found: .*missing\.j2"):
        builder.build_system_prompt_path()


def test_builder_routes_yaml_tools_through_legacy_adapter(tmp_path) -> None:
    definition = tmp_path / "echo.yaml"
    definition.write_text(
        """
name: echo_source
description: Echo a value
input_schema:
  type: object
  properties:
    value:
      type: string
  required: [value]
  additionalProperties: false
""".strip(),
        encoding="utf-8",
    )
    builder = NexAUConfigAdapter.builder(
        {
            "tools": [
                {
                    "name": "echo",
                    "yaml_path": definition.name,
                    "binding": lambda **params: params,
                }
            ]
        },
        tmp_path,
    )

    builder.build_tools()
    tool = builder.agent_params["tools"][0]

    assert isinstance(tool, Tool)
    assert isinstance(tool, LegacyCompatibleTool)
    tool.validate_params({"value": "ok", "extra": True})


def test_builder_does_not_silently_drop_invalid_tool(tmp_path) -> None:
    builder = NexAUConfigAdapter.builder(
        {"tools": [{"name": "missing", "yaml_path": "missing.yaml"}]},
        tmp_path,
    )

    with pytest.raises(ConfigError, match=r"Error loading tool 'missing'"):
        builder.build_tools()


def test_adapter_build_returns_public_agent_config_type(tmp_path) -> None:
    config = {
        "name": "compat-agent",
        "description": "Compatibility fixture",
        "llm_config": _llm_config(),
    }

    result = NexAUConfigAdapter.build(config, tmp_path)

    assert isinstance(result, AgentConfig)
    assert result.name == "compat-agent"
    assert result.tools == []
    assert getattr(result, "skipped_components", []) == []


@pytest.mark.parametrize("field", ["plugins", "skipped_components", "source_id"])
def test_public_only_agent_config_fields_remain_rejected(field: str) -> None:
    with pytest.raises(ConfigError, match="Unsupported legacy agent config fields"):
        NexAUConfigAdapter.normalize({field: "public-only"})


@pytest.mark.parametrize("field", ["permissions", "source_id"])
def test_public_only_tool_config_fields_remain_rejected(field: str) -> None:
    with pytest.raises(ConfigError, match="Unsupported legacy tool config fields"):
        NexAUConfigAdapter.normalize({"tools": [{"name": "tool", "yaml_path": "tool.yaml", field: {}}]})


def test_normalize_rejects_unknown_fields() -> None:
    config = {
        "name": "compat-agent",
        "llm_config": _llm_config(),
        "unknown_component": {},
    }

    with pytest.raises(ConfigError, match="unknown_component"):
        NexAUConfigAdapter.normalize(config)
