from __future__ import annotations

from typing import Any

import pytest
from nexau.archs.main_sub.utils.common import ConfigError as ImportConfigError
from nexau.archs.tool import Tool

from northcua.agentinfra.nexau_compat.adapters import tool as tool_adapter_module
from northcua.agentinfra.nexau_compat.adapters.tool import LegacyCompatibleTool, NexAUToolAdapter


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }


def test_adapter_preserves_nexau_type_identity_and_legacy_representation() -> None:
    tool = NexAUToolAdapter.create(
        name="echo",
        description="Echo a value",
        input_schema=_schema(),
        implementation=lambda **params: params,
    )

    assert isinstance(tool, Tool)
    assert isinstance(tool, LegacyCompatibleTool)
    assert tool.formatter is None
    assert repr(tool).endswith("formatter=xml)")


def test_adapter_enforces_xml_on_tools_constructed_inside_nexau() -> None:
    tool = Tool(
        name="builtin",
        description="NexAU-owned tool",
        input_schema={"type": "object"},
        implementation=lambda: {"status": "ok", "result": "done"},
    )

    returned = NexAUToolAdapter.enforce_legacy_formatter(tool)
    rendered = returned.format_output_for_llm(
        tool_input={},
        tool_output={"status": "ok", "result": "done"},
        tool_call_id="call-builtin",
        is_error=False,
    )

    assert returned is tool
    assert isinstance(rendered, str)
    assert rendered.startswith("<tool_result>")


def test_adapter_enforces_private_validation_on_tools_constructed_inside_nexau() -> None:
    tool = Tool(
        name="mcp_like",
        description="NexAU-owned tool with framework-injected arguments",
        input_schema=_schema(),
        implementation=lambda **params: params,
    )

    returned = NexAUToolAdapter.enforce_legacy_formatter(tool)

    returned.validate_params({"value": "valid", "hook_injected": "kept for execution"})
    assert returned.execute(value="valid", hook_injected="kept for execution") == {
        "value": "valid",
        "hook_injected": "kept for execution",
    }
    with pytest.raises(ValueError, match=r"params=\{'value': 7\}"):
        returned.validate_params({"value": 7, "hook_injected": "ignored by validation"})


def test_nexau_owned_behavior_installation_is_idempotent() -> None:
    tool = Tool(
        name="builtin",
        description="NexAU-owned tool",
        input_schema=_schema(),
        implementation=lambda **params: params,
    )

    first = NexAUToolAdapter.enforce_legacy_formatter(tool)
    first_validate = first.validate_params
    first_formatter = first.formatter
    second = NexAUToolAdapter.enforce_legacy_formatter(tool)

    assert second is first
    assert second.validate_params == first_validate
    assert second.formatter is first_formatter


def test_adapter_replaces_public_markdown_fallback_for_nexau_owned_custom_formatter() -> None:
    def broken_formatter(_context: object) -> object:
        raise RuntimeError("custom formatter failed")

    tool = Tool(
        name="builtin-custom",
        description="NexAU-owned tool with a custom formatter",
        input_schema={"type": "object"},
        implementation=lambda: {"status": "ok", "result": "done"},
        formatter=broken_formatter,
    )

    returned = NexAUToolAdapter.enforce_legacy_formatter(tool)
    rendered = returned.format_output_for_llm(
        tool_input={},
        tool_output={"status": "ok", "result": "done"},
        tool_call_id="call-builtin-custom",
        is_error=False,
    )

    assert returned is tool
    assert isinstance(rendered, str)
    assert rendered.startswith("<tool_result>")
    assert "```" not in rendered


def test_default_and_failure_fallback_both_use_legacy_xml() -> None:
    def broken_formatter(_context: object) -> object:
        raise RuntimeError("custom formatter failed")

    default_tool = NexAUToolAdapter.create(
        name="default",
        description="Default formatter",
        input_schema={"type": "object"},
        implementation=lambda: None,
    )
    custom_tool = NexAUToolAdapter.create(
        name="custom",
        description="Custom formatter",
        input_schema={"type": "object"},
        implementation=lambda: None,
        formatter=broken_formatter,
    )
    output = {"status": "ok", "result": "line one\nline two"}

    expected = default_tool.format_output_for_llm(
        tool_input={},
        tool_output=output,
        tool_call_id="call-1",
        is_error=False,
    )
    fallback = custom_tool.format_output_for_llm(
        tool_input={},
        tool_output=output,
        tool_call_id="call-1",
        is_error=False,
    )

    assert isinstance(expected, str)
    assert expected.startswith("<tool_result>")
    assert '<body field="result"><![CDATA[' in expected
    assert fallback == expected


def test_public_markdown_alias_is_rejected_like_private_pin() -> None:
    tool = NexAUToolAdapter.create(
        name="markdown",
        description="Private pin treated markdown as an import path",
        input_schema={"type": "object"},
        implementation=lambda: None,
        formatter="markdown",
    )

    with pytest.raises(ImportConfigError, match=r"Error importing from 'markdown'"):
        tool.format_output_for_llm(
            tool_input={},
            tool_output={"status": "ok", "result": "line one\nline two"},
            tool_call_id="call-1",
            is_error=False,
        )


def test_formatter_failure_returns_raw_output_when_xml_also_fails(monkeypatch) -> None:
    def broken_formatter(_context: object) -> object:
        raise RuntimeError("custom formatter failed")

    def resolver(formatter):
        if formatter == "xml":
            raise RuntimeError("xml formatter failed")
        return formatter

    monkeypatch.setattr(tool_adapter_module, "resolve_tool_formatter", resolver)
    tool = NexAUToolAdapter.create(
        name="raw-fallback",
        description="Raw fallback",
        input_schema={"type": "object"},
        implementation=lambda: None,
        formatter=broken_formatter,
    )
    raw_output = {"opaque": object()}

    result = tool.format_output_for_llm(
        tool_input={},
        tool_output=raw_output,
        tool_call_id=None,
        is_error=True,
    )

    assert result is raw_output


def test_validation_ignores_undeclared_parameters_but_checks_declared_values() -> None:
    tool = NexAUToolAdapter.create(
        name="echo",
        description="Echo a value",
        input_schema=_schema(),
        implementation=lambda **params: params,
    )

    tool.validate_params({"value": "valid", "hook_injected": object()})
    result = tool.execute(value="valid", hook_injected="still passed to implementation")

    assert result == {"value": "valid", "hook_injected": "still passed to implementation"}
    with pytest.raises(ValueError, match=r"params=\{'value': 7\}"):
        tool.validate_params({"value": 7, "ignored": "extra"})
    with pytest.raises(ValueError, match=r"params=\{\}"):
        tool.validate_params({"ignored": "extra"})


def test_yaml_factory_retains_legacy_defaults(tmp_path) -> None:
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

    tool = NexAUToolAdapter.from_yaml(
        str(definition),
        lambda **params: params,
        name="echo",
    )

    assert isinstance(tool, Tool)
    assert isinstance(tool, LegacyCompatibleTool)
    assert tool.name == "echo"
    assert tool.source_name == "echo_source"
    assert tool.formatter is None
    tool.validate_params({"value": "valid", "extra": True})
