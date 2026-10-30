from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from northcua.agentkit.builtin_common import ToolContractProjector
from northcua.agents.web.mcp.registration import ensure_web_agent_mcp_servers_registered
from northcua.core.errors import MCPRegistryError
from northcua.runtime.mcp import MCPFactoryContext, NorthCUAMCPRegistry, ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.contracts import (
    ToolSchemaDrift,
    compare_contracts_to_fastmcp,
    contract_schema_by_tool_id,
    provider_schema_by_tool_id,
    register_contract_handlers,
    register_mcp_toolset,
    validate_contract_parameters,
)
from northcua.sandboxes.tool_contracts import SANDBOX_PROVIDER_TOOL_CONTRACTS
from northcua_runtime_shared.desktop_surface.a11y_contracts import DESKTOP_NATIVE_A11Y_SERVER_NAME
from northcua_runtime_shared.desktop_surface.action_contracts import DESKTOP_NATIVE_ACTION_SERVER_NAME
from northcua_runtime_shared.desktop_surface.data_contracts import DESKTOP_NATIVE_DATA_SERVER_NAME
from northcua_runtime_shared.tool_contracts import (
    DirectHandlerBinding,
    McpBinding,
    ProviderSchemaMode,
    ToolContract,
    ToolDispatcher,
    ToolDispatchPolicy,
    ToolProfile,
    ToolSurfaceResolver,
    ToolVisibility,
    define_tool,
    owned_tool,
    toolset,
)
from northcua_runtime_shared.validation import validate_command_against_tool

_PROVIDER_UNSUPPORTED_SCHEMA_KEYS = {
    "$defs",
    "allOf",
    "anyOf",
    "dependentRequired",
    "not",
    "oneOf",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PRODUCTION_SCAN_ROOTS = ("northcua", "northcua_runtime_shared", "northcua_device")
_RAW_NEXAU_TOOL_CONSTRUCTOR_ALLOWLIST: dict[str, int] = {}
_RAW_TOOL_DESCRIPTOR_CONSTRUCTOR_ALLOWLIST = {"northcua_runtime_shared/tool_contracts/projection.py": 1}
_REMOVED_TOOL_API_NAMES = (
    "ToolSpec",
    "ProbeSpec",
    "EvaluatorToolSpec",
    "EvaluatorArtifactToolSpec",
    "ARTIFACT_TOOL_SPECS",
)
_PROMPT_SCHEMA_FORBIDDEN_PATTERNS = (
    re.compile(r"\b(input|parameters?|json)[_-]?schema\b", re.IGNORECASE),
    re.compile(r"\b(input|parameters?|json)\s+schema\b", re.IGNORECASE),
    re.compile(r"['\"](?:properties|required|enum|default)['\"]\s*:"),
)


def _contract_schema_by_tool_id(server_name: str) -> dict[str, dict[str, object]]:
    return {
        contract.tool_id: contract.parameters_schema for contract in NorthCUAMCPRegistry.tool_contracts(server_name)
    }


def _descriptor_schema_by_tool_id(server_name: str) -> dict[str, dict[str, object]]:
    return {
        descriptor.tool_id: descriptor.parameters_schema
        for descriptor in NorthCUAMCPRegistry.wire_tool_descriptors(server_name)
    }


def _is_production_registered_server(server_name: str) -> bool:
    registered = NorthCUAMCPRegistry._registry.get(server_name)
    if registered is None:
        return False
    return str(getattr(registered.factory, "__module__", "") or "").startswith("northcua.")


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _production_call_counts(call_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _dotted_name(node.func).split(".")[-1] == call_name:
                    count += 1
            if count:
                counts[str(path.relative_to(_REPO_ROOT))] = count
    return counts


def _production_mcp_tool_call_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tool":
                    count += 1
            if count:
                counts[str(path.relative_to(_REPO_ROOT))] = count
    return counts


def _removed_tool_api_name_references() -> list[str]:
    references: list[str] = []
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in _REMOVED_TOOL_API_NAMES:
                    references.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{node.id}")
                elif isinstance(node, ast.Attribute) and node.attr in _REMOVED_TOOL_API_NAMES:
                    references.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{node.attr}")
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in _REMOVED_TOOL_API_NAMES:
                            references.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{alias.name}")
    return references


def _production_nexau_tool_constructor_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    tool_modules = {"nexau.archs.tool", "nexau.archs.tool.tool"}
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            tool_aliases: set[str] = set()
            module_aliases: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in tool_modules:
                    for alias in node.names:
                        if alias.name == "Tool":
                            tool_aliases.add(alias.asname or alias.name)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in tool_modules:
                            module_aliases.add(alias.asname or alias.name)
            count = 0
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in tool_aliases:
                    count += 1
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr == "Tool":
                    prefix = _dotted_name(node.func.value)
                    if prefix in module_aliases or f"{prefix}.Tool" in {f"{module}.Tool" for module in tool_modules}:
                        count += 1
            if count:
                counts[str(path.relative_to(_REPO_ROOT))] = count
    return counts


def _production_tool_descriptor_constructor_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    descriptor_modules = {
        "northcua_protocol.capabilities",
        "northcua_protocol.tools",
    }
    parent_modules = {"northcua_protocol"}
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            descriptor_aliases: set[str] = set()
            module_aliases: set[str] = set()
            parent_aliases: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in descriptor_modules | parent_modules:
                    for alias in node.names:
                        if alias.name == "ToolDescriptor":
                            descriptor_aliases.add(alias.asname or alias.name)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in descriptor_modules:
                            module_aliases.add(alias.asname or alias.name)
                        if alias.name in parent_modules:
                            parent_aliases.add(alias.asname or alias.name)
            count = 0
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in descriptor_aliases:
                    count += 1
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr == "ToolDescriptor":
                    prefix = _dotted_name(node.func.value)
                    if prefix in module_aliases or any(
                        prefix in {f"{parent}.capabilities", f"{parent}.tools"} for parent in parent_aliases
                    ):
                        count += 1
            if count:
                counts[str(path.relative_to(_REPO_ROOT))] = count
    return counts


def _constructor_names_from_imports(tree: ast.AST, class_name: str, modules: set[str]) -> set[str]:
    aliases = {class_name}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in modules:
            for alias in node.names:
                if alias.name == class_name:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _prompt_template_paths() -> list[Path]:
    candidates: list[Path] = []
    for root in (
        _REPO_ROOT / "northcua" / "agentkernel" / "prompt",
        _REPO_ROOT / "northcua" / "agentkit" / "evaluator" / "prompts",
        _REPO_ROOT / "northcua" / "agents",
    ):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".j2", ".json", ".yaml", ".yml", ".py", ".md"}:
                continue
            lowered_parts = {part.lower() for part in path.parts}
            lowered_name = path.name.lower()
            if "prompt" in lowered_parts or "templates" in lowered_parts or "prompt" in lowered_name:
                candidates.append(path)
    return sorted(set(candidates))


class _AuthoringEchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


def test_owned_tool_authoring_registers_fastmcp_from_contract_and_dispatches() -> None:
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    calls: list[str] = []

    @owned_tool(
        tool_id="echo",
        description="Echo one string.",
        binding=McpBinding(server_name="AuthoringTestExecutor"),
        side_effect=False,
    )
    def echo(params: _AuthoringEchoParams) -> str:
        calls.append(params.text)
        return params.text.upper()

    owned = toolset("authoring-test", [echo])
    server = FastMCP("authoring-test")
    register_mcp_toolset(server, owned)

    live_tools = asyncio.run(server.list_tools())
    assert [str(tool.name) for tool in live_tools] == ["echo"]
    assert live_tools[0].description == "Echo one string."
    assert dict(live_tools[0].parameters) == owned.contracts[0].parameters_schema

    asyncio.run(server.call_tool("echo", {"text": "hi"}))
    assert calls == ["hi"]

    with pytest.raises(ToolError, match="Extra inputs are not permitted"):
        asyncio.run(server.call_tool("echo", {"text": "hi", "unexpected": True}))
    assert calls == ["hi"]


def test_contract_handler_registration_rejects_missing_and_extra_handlers() -> None:
    from fastmcp import FastMCP

    contract = define_tool(
        tool_id="strict_echo",
        description="Strict echo.",
        params_model=_AuthoringEchoParams,
        binding=McpBinding(server_name="StrictHandlerTestExecutor"),
    )

    with pytest.raises(ValueError, match="Missing handlers"):
        register_contract_handlers(FastMCP("missing-handler-test"), [contract], {})
    with pytest.raises(ValueError, match="without matching contracts"):
        register_contract_handlers(
            FastMCP("extra-handler-test"),
            [contract],
            {
                "strict_echo": lambda params: params,
                "extra": lambda params: params,
            },
        )


def test_contract_to_nexau_tool_validates_payload_before_handler() -> None:
    from northcua.agentkernel.tools import contract_to_nexau_tool

    calls: list[str] = []

    def _handler(params: _AuthoringEchoParams) -> str:
        calls.append(params.text)
        return params.text.upper()

    contract = define_tool(
        tool_id="direct_echo",
        description="Direct echo.",
        params_model=_AuthoringEchoParams,
        binding=DirectHandlerBinding(handler_path="tests:direct_echo"),
    )
    tool = contract_to_nexau_tool(contract, implementation=_handler)

    assert asyncio.run(tool.implementation(text="hi")) == "HI"
    assert calls == ["hi"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        asyncio.run(tool.implementation(text="hi", unexpected=True))
    assert calls == ["hi"]


def test_contract_to_nexau_tool_json_schema_validates_before_handler() -> None:
    from northcua.agentkernel.tools import contract_to_nexau_tool

    calls: list[dict[str, object]] = []

    async def _handler(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"ok": True}

    contract = define_tool(
        tool_id="schema_only_direct_echo",
        description="Schema-only direct echo.",
        json_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        binding=DirectHandlerBinding(handler_path="tests:schema_only_direct_echo"),
    )
    tool = contract_to_nexau_tool(contract, implementation=_handler)

    with pytest.raises(ValueError, match="unexpected parameters|required"):
        asyncio.run(tool.implementation(unexpected=True))
    assert calls == []


def test_empty_parameters_contract_rejects_extra_fields() -> None:
    contract = define_tool(
        tool_id="empty_payload",
        description="No-argument tool.",
        binding=McpBinding(server_name="EmptyPayloadExecutor"),
        side_effect=False,
    )

    assert ToolDispatcher().prepare(contract, {}).validated_parameters == {}
    with pytest.raises(ValueError, match="unexpected parameters"):
        ToolDispatcher().prepare(contract, {"unexpected": True})


def test_in_memory_direct_tool_validates_contract_before_build(monkeypatch: pytest.MonkeyPatch) -> None:
    from northcua.agentkernel.tools import contract_to_nexau_tool
    from northcua.runtime.mcp.in_memory_tools import _build_tool_implementation

    contract = define_tool(
        tool_id="in_memory_echo",
        description="In-memory echo.",
        params_model=_AuthoringEchoParams,
        binding=McpBinding(server_name="InMemoryValidationExecutor"),
    )
    build_called = False

    def _fail_build(*_args: object, **_kwargs: object) -> object:
        nonlocal build_called
        build_called = True
        raise AssertionError("server build should not happen for invalid payload")

    monkeypatch.setattr(NorthCUAMCPRegistry, "build", _fail_build)
    implementation = contract_to_nexau_tool(
        contract,
        implementation=_build_tool_implementation("InMemoryValidationExecutor", "in_memory_echo"),
    ).implementation

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        asyncio.run(implementation(text="hi", unexpected=True))
    assert build_called is False


def test_owned_mcp_tools_do_not_use_unmanaged_fastmcp_tool_calls() -> None:
    assert _production_mcp_tool_call_counts() == {}


def test_removed_tool_api_names_are_not_available_in_production() -> None:
    assert _removed_tool_api_name_references() == []


def test_raw_nexau_tool_constructors_are_forbidden_outside_compatibility_adapter() -> None:
    assert _production_nexau_tool_constructor_counts() == _RAW_NEXAU_TOOL_CONSTRUCTOR_ALLOWLIST


def test_raw_tool_descriptor_constructors_are_confined_to_contract_projection() -> None:
    assert _production_tool_descriptor_constructor_counts() == _RAW_TOOL_DESCRIPTOR_CONSTRUCTOR_ALLOWLIST


def test_desktop_native_servers_project_descriptors_from_tool_contracts() -> None:
    ensure_builtin_mcp_servers_registered()

    for server_name in (
        DESKTOP_NATIVE_DATA_SERVER_NAME,
        DESKTOP_NATIVE_A11Y_SERVER_NAME,
        DESKTOP_NATIVE_ACTION_SERVER_NAME,
    ):
        contract_schemas = _contract_schema_by_tool_id(server_name)
        descriptor_schemas = _descriptor_schema_by_tool_id(server_name)

        assert contract_schemas
        assert descriptor_schemas == contract_schemas


def test_registry_api_no_longer_accepts_descriptor_metadata() -> None:
    from fastmcp import FastMCP

    contract = ToolContract(tool_id="contract_only")
    descriptor = contract.to_descriptor().model_copy(update={"parameters_schema": {"type": "object"}})

    with pytest.raises(TypeError, match="tool_descriptors"):
        NorthCUAMCPRegistry.register(
            "RawDescriptorRegistrationTestExecutor",
            lambda _context: FastMCP("raw-descriptor-registration-test"),
            tool_descriptors=[descriptor],
            tool_contracts=[contract],
        )


def test_registry_rejects_descriptor_only_registration_by_default() -> None:
    from fastmcp import FastMCP

    with pytest.raises(MCPRegistryError, match="owned servers must register tool_contracts"):
        NorthCUAMCPRegistry.register(
            "DescriptorOnlyDefaultRejectTestExecutor",
            lambda _context: FastMCP("descriptor-only-default-reject-test"),
        )


def test_agentkit_contract_projector_cannot_force_runtime_only_contract_visible() -> None:
    contract = ToolContract(tool_id="internal_probe", visibility=ToolVisibility.RUNTIME_ONLY)

    with pytest.raises(ValueError, match="Cannot force runtime/operator-only"):
        ToolContractProjector().tool_specs([contract], model_visible=True)


def test_command_line_executor_is_contract_backed_model_visible_surface() -> None:
    ensure_builtin_mcp_servers_registered()

    contracts = NorthCUAMCPRegistry.tool_contracts("CommandLineExecutor")

    assert [contract.tool_id for contract in contracts] == ["exec"]
    assert contracts[0].model_visible is True
    assert contracts[0].provider_parameters_schema["properties"]["bash_command"]["type"] == "string"


def test_browser_snapshot_contract_does_not_expose_arbitrary_filename_write() -> None:
    ensure_builtin_mcp_servers_registered()
    contracts = {
        contract.tool_id: contract for contract in NorthCUAMCPRegistry.tool_contracts("PlaywrightBrowserExecutor")
    }

    schema = contracts["browser_snapshot"].parameters_schema

    assert "filename" not in schema["properties"]


def test_playwright_browser_contracts_reject_runtime_invalid_payloads() -> None:
    ensure_builtin_mcp_servers_registered()
    contracts = {
        contract.tool_id: contract for contract in NorthCUAMCPRegistry.tool_contracts("PlaywrightBrowserExecutor")
    }

    for tool_id in (
        "browser_tabs",
        "browser_click",
        "browser_drag",
        "browser_hover",
        "browser_select_option",
        "browser_generate_locator",
        "browser_check",
        "browser_uncheck",
    ):
        contract = contracts[tool_id]
        assert contract.provider_schema_mode == ProviderSchemaMode.UNION
        assert contract.provider_strict is False
        assert "oneOf" in contract.provider_parameters_schema or "anyOf" in contract.provider_parameters_schema

    with pytest.raises(ValueError, match="allowed enum"):
        validate_contract_parameters(contracts["browser_tabs"], {"action": "definitely-not-valid"})
    with pytest.raises(ValueError, match="exactly one schema"):
        validate_contract_parameters(contracts["browser_tabs"], {"action": "select"})
    assert validate_contract_parameters(contracts["browser_tabs"], {"action": "select", "index": 0}) == {
        "action": "select",
        "index": 0,
    }

    for tool_id in (
        "browser_click",
        "browser_hover",
        "browser_select_option",
        "browser_generate_locator",
        "browser_check",
        "browser_uncheck",
    ):
        payload = {"values": ["option"]} if tool_id == "browser_select_option" else {}
        with pytest.raises(ValueError, match="does not match any allowed schema"):
            validate_contract_parameters(contracts[tool_id], payload)
        with pytest.raises(ValueError, match="does not match any allowed schema"):
            validate_contract_parameters(contracts[tool_id], {**payload, "ref": None})
        assert validate_contract_parameters(contracts[tool_id], {**payload, "ref": "e1"}) == {**payload, "ref": "e1"}

    with pytest.raises(ValueError, match="does not match any allowed schema"):
        validate_contract_parameters(
            contracts["browser_drag"],
            {"startElement": "source", "endElement": "target", "startRef": "e1"},
        )
    assert validate_contract_parameters(
        contracts["browser_drag"],
        {"startElement": "source", "endElement": "target", "startRef": "e1", "endSelector": "#target"},
    ) == {"startElement": "source", "endElement": "target", "startRef": "e1", "endSelector": "#target"}


def test_all_builtin_model_visible_servers_are_contract_backed() -> None:
    ensure_builtin_mcp_servers_registered()
    builtin_model_visible_servers = {
        "CommandLineExecutor",
        "DesktopNativeActionExecutor",
        "PlaywrightBrowserExecutor",
    }

    legacy_visible_servers = {
        server_name
        for server_name in builtin_model_visible_servers
        if not NorthCUAMCPRegistry.tool_contracts(server_name)
    }

    assert legacy_visible_servers == set()


def test_production_tool_declaration_anti_patterns_are_blocked() -> None:
    missing_contract_registrations: list[str] = []
    evaluator_inline_schemas: list[str] = []
    agentkit_schema_overrides: list[str] = []
    agentkit_bare_specs: list[str] = []
    descriptor_only_registry_calls: list[str] = []
    descriptor_to_contract_bridges: list[str] = []
    descriptor_agentkit_projectors: list[str] = []

    for root in _PRODUCTION_SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(_REPO_ROOT)
            tool_spec_names = _constructor_names_from_imports(
                tree,
                "ToolUseRef",
                {"northcua.agentkit", "northcua.agentkit.spec"},
            )
            probe_spec_names = _constructor_names_from_imports(
                tree,
                "ProbeUseRef",
                {"northcua.agentkit", "northcua.agentkit.spec"},
            )
            evaluator_tool_spec_names = _constructor_names_from_imports(
                tree,
                "EvaluatorToolRef",
                {
                    "northcua.agentkit.evaluator",
                    "northcua.agentkit.evaluator.spec",
                    "northcua.agentkit.evaluator.spec.policies",
                },
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = _dotted_name(node.func)
                called = call_name.split(".")[-1]
                keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
                if call_name == "NorthCUAMCPRegistry.register_decorator" and "tool_contracts" not in keywords:
                    missing_contract_registrations.append(f"{rel}:{node.lineno}")
                if called in evaluator_tool_spec_names and "contract" not in keywords:
                    evaluator_inline_schemas.append(f"{rel}:{node.lineno}")
                if called in evaluator_tool_spec_names and "input_schema" in keywords and "contract" not in keywords:
                    evaluator_inline_schemas.append(f"{rel}:{node.lineno}")
                if called in tool_spec_names | probe_spec_names and "parameters_schema_override" in keywords:
                    if str(rel) != "northcua/agentkit/builtin_common.py":
                        agentkit_schema_overrides.append(f"{rel}:{node.lineno}")
                if called in tool_spec_names | probe_spec_names and "contract" not in keywords:
                    if str(rel) != "northcua/agentkit/builtin_common.py":
                        agentkit_bare_specs.append(f"{rel}:{node.lineno}")
                if "tool_descriptors" in keywords or "allow_descriptor_only" in keywords:
                    if str(rel) != "northcua/runtime/mcp/registry.py":
                        descriptor_only_registry_calls.append(f"{rel}:{node.lineno}")
                if call_name.endswith("from_descriptor") or call_name == "contracts_from_registered_descriptors":
                    descriptor_to_contract_bridges.append(f"{rel}:{node.lineno}")
                if called in {"ProtocolDescriptorProjector", "tool_specs_from_protocol", "probe_specs_from_protocol"}:
                    descriptor_agentkit_projectors.append(f"{rel}:{node.lineno}")

    assert missing_contract_registrations == []
    assert evaluator_inline_schemas == []
    assert agentkit_schema_overrides == []
    assert agentkit_bare_specs == []
    assert descriptor_only_registry_calls == []
    assert descriptor_to_contract_bridges == []
    assert descriptor_agentkit_projectors == []


def test_prompts_do_not_redeclare_tool_json_schemas() -> None:
    violations: list[str] = []
    for path in _prompt_template_paths():
        text = path.read_text(encoding="utf-8")
        for pattern in _PROMPT_SCHEMA_FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(_REPO_ROOT)} contains {pattern.pattern!r}")

    assert violations == []


def test_descriptor_validator_enforces_dependent_required_after_provider_strip() -> None:
    ensure_builtin_mcp_servers_registered()

    descriptor = define_tool(
        tool_id="dependent_probe",
        description="Probe a selected process window.",
        json_schema={
            "type": "object",
            "properties": {"pid": {"type": "integer"}, "window_id": {"type": "integer"}},
            "dependentRequired": {"window_id": ["pid"]},
        },
        binding=McpBinding(server_name="TestProbe"),
    ).to_descriptor()

    with pytest.raises(ValueError, match="missing dependent parameters"):
        validate_command_against_tool(
            tool_id="dependent_probe",
            parameters={"window_id": 42},
            descriptor=descriptor,
        )


def test_runtime_validator_rejects_sandbox_exclusive_minimum_and_nested_ref_errors() -> None:
    contracts = {contract.tool_id: contract for contract in SANDBOX_PROVIDER_TOOL_CONTRACTS}

    with pytest.raises(ValueError, match="> 0"):
        validate_command_against_tool(
            tool_id="sandbox_extend_timeout",
            parameters={"handle_id": "sandbox-1", "timeout_seconds": 0},
            descriptor=contracts["sandbox_extend_timeout"].to_descriptor(),
        )
    with pytest.raises(ValueError, match="missing required"):
        validate_command_against_tool(
            tool_id="sandbox_run_command",
            parameters={"handle_id": "sandbox-1", "command": {"timeout_seconds": 1}},
            descriptor=contracts["sandbox_run_command"].to_descriptor(),
        )


def test_runtime_only_contracts_stay_out_of_provider_projection() -> None:
    ensure_builtin_mcp_servers_registered()

    for server_name in (DESKTOP_NATIVE_DATA_SERVER_NAME, DESKTOP_NATIVE_A11Y_SERVER_NAME):
        contracts = NorthCUAMCPRegistry.tool_contracts(server_name)

        assert contracts
        assert all(not contract.model_visible for contract in contracts)


def test_registered_contract_inventory_has_no_provider_visibility_leaks() -> None:
    ensure_builtin_mcp_servers_registered()
    ensure_web_agent_mcp_servers_registered()

    for server_name in NorthCUAMCPRegistry.names():
        contracts = NorthCUAMCPRegistry.tool_contracts(server_name)
        if not contracts:
            continue
        surface = ToolSurfaceResolver(tuple(contracts)).resolve(ToolProfile(name=f"mcp.{server_name}"))
        provider_contract_ids = {tool.contract_id for tool in surface.provider_tools}
        model_visible_contract_ids = {contract.contract_id for contract in contracts if contract.model_visible}

        assert provider_contract_ids == model_visible_contract_ids
        for contract in contracts:
            if contract.model_visible:
                assert contract.description
                assert contract.provider_parameters_schema.get("type") == "object"


def test_contract_validator_rejects_provider_stripped_keypress_oneof_violation() -> None:
    ensure_builtin_mcp_servers_registered()

    contracts = {
        contract.tool_id: contract for contract in NorthCUAMCPRegistry.tool_contracts(DESKTOP_NATIVE_ACTION_SERVER_NAME)
    }
    provider_schemas = provider_schema_by_tool_id(contracts.values())
    keypress_contract = contracts["keypress"]

    assert "oneOf" not in provider_schemas["keypress"]
    assert provider_schemas["keypress"]["properties"]["key"]["type"] == ["string", "null"]
    assert provider_schemas["keypress"]["properties"]["keys"]["type"] == ["array", "null"]
    assert validate_contract_parameters(keypress_contract, {"key": "enter"}).model_dump(mode="json") == {
        "key": "enter",
        "keys": None,
    }
    with pytest.raises(ValidationError, match="exactly one of `key` or `keys`"):
        validate_contract_parameters(keypress_contract, {"key": "enter", "keys": ["ctrl+l"]})


def test_desktop_action_fastmcp_schema_matches_contract_projection(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    server = NorthCUAMCPRegistry.build(
        DESKTOP_NATIVE_ACTION_SERVER_NAME,
        context=MCPFactoryContext(process_name="tool-contract-sot-test"),
    )
    contracts = NorthCUAMCPRegistry.tool_contracts(DESKTOP_NATIVE_ACTION_SERVER_NAME)

    drifts = asyncio.run(
        compare_contracts_to_fastmcp(
            contract_schemas=contract_schema_by_tool_id(contracts),
            server=server,
        )
    )

    assert drifts == []


def test_contract_backed_fastmcp_servers_match_live_surfaces(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    ensure_web_agent_mcp_servers_registered()

    contract_backed_servers = [
        server_name for server_name in NorthCUAMCPRegistry.names() if NorthCUAMCPRegistry.tool_contracts(server_name)
    ]

    assert {
        DESKTOP_NATIVE_A11Y_SERVER_NAME,
        DESKTOP_NATIVE_ACTION_SERVER_NAME,
        DESKTOP_NATIVE_DATA_SERVER_NAME,
        "CommandLineExecutor",
        "PlaywrightBrowserExecutor",
    } <= set(contract_backed_servers)

    for server_name in contract_backed_servers:
        server = NorthCUAMCPRegistry.build(
            server_name,
            context=MCPFactoryContext(process_name="tool-contract-sot-test"),
        )
        contracts = NorthCUAMCPRegistry.tool_contracts(server_name)

        drifts = asyncio.run(
            compare_contracts_to_fastmcp(
                contract_schemas=contract_schema_by_tool_id(contracts),
                server=server,
            )
        )

        assert drifts == []


def test_contract_backed_fastmcp_live_metadata_exactly_matches_contracts(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    ensure_web_agent_mcp_servers_registered()

    async def _check_server(server_name: str) -> None:
        server = NorthCUAMCPRegistry.build(
            server_name,
            context=MCPFactoryContext(process_name="tool-contract-exact-live-test"),
        )
        contracts = {contract.provider_name: contract for contract in NorthCUAMCPRegistry.tool_contracts(server_name)}
        live_tools = {str(tool.name): tool for tool in await server.list_tools()}

        assert set(live_tools) == set(contracts)
        for tool_name, tool in live_tools.items():
            contract = contracts[tool_name]
            assert str(tool.description or "") == contract.description
            if contract.json_schema is not None:
                assert dict(getattr(tool, "parameters", {}) or {}) == contract.parameters_schema

    for server_name in NorthCUAMCPRegistry.names():
        if not _is_production_registered_server(server_name):
            continue
        if NorthCUAMCPRegistry.tool_contracts(server_name):
            asyncio.run(_check_server(server_name))


def test_contract_registry_bindings_match_descriptors_and_live_tool_names(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    ensure_web_agent_mcp_servers_registered()

    from fastmcp import FastMCP

    public_name_contract = define_tool(
        tool_id="canonical_live_name",
        public_name="public_live_name",
        description="Live public-name parity.",
        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        binding=McpBinding(server_name="PublicNameLiveParityExecutor"),
    )

    def _create_public_name_server(context: MCPFactoryContext) -> FastMCP:
        del context
        mcp = FastMCP("PublicNameLiveParityExecutor")
        register_contract_handlers(mcp, [public_name_contract], {"canonical_live_name": lambda: {"ok": True}})
        return mcp

    NorthCUAMCPRegistry.register(
        "PublicNameLiveParityExecutor",
        _create_public_name_server,
        tool_contracts=[public_name_contract],
    )

    try:
        for server_name in NorthCUAMCPRegistry.names():
            if server_name != "PublicNameLiveParityExecutor" and not _is_production_registered_server(server_name):
                continue
            contracts = NorthCUAMCPRegistry.tool_contracts(server_name)
            if not contracts:
                continue
            descriptors = NorthCUAMCPRegistry.wire_tool_descriptors(server_name)
            server = NorthCUAMCPRegistry.build(
                server_name,
                context=MCPFactoryContext(process_name="tool-contract-route-parity-test"),
            )
            live_tool_names = {str(tool.name) for tool in asyncio.run(server.list_tools())}

            assert {contract.tool_id for contract in contracts} == {descriptor.tool_id for descriptor in descriptors}
            assert {contract.provider_name for contract in contracts} == live_tool_names
            assert all(contract.server_name == server_name for contract in contracts)
    finally:
        NorthCUAMCPRegistry._registry.pop("PublicNameLiveParityExecutor", None)


def test_fastmcp_drift_audit_reports_extra_live_tools() -> None:
    from fastmcp import FastMCP

    server = FastMCP("tool-contract-extra-test")

    @server.tool
    def extra_tool() -> str:
        return "unexpected"

    drifts = asyncio.run(compare_contracts_to_fastmcp(contract_schemas={}, server=server))

    assert drifts == [
        ToolSchemaDrift(
            tool_name="extra_tool",
            contract_hash="",
            runtime_hash=drifts[0].runtime_hash,
            reason="extra_runtime_tool",
        )
    ]


def test_fastmcp_drift_audit_reports_missing_live_tools() -> None:
    from fastmcp import FastMCP

    server = FastMCP("tool-contract-missing-test")

    drifts = asyncio.run(compare_contracts_to_fastmcp(contract_schemas={"expected_tool": {}}, server=server))

    assert drifts == [
        ToolSchemaDrift(
            tool_name="expected_tool",
            contract_hash=drifts[0].contract_hash,
            runtime_hash="",
            reason="missing_runtime_tool",
        )
    ]


def test_provider_schema_projection_has_no_dangling_refs_or_unsupported_combinators() -> None:
    ensure_builtin_mcp_servers_registered()
    ensure_web_agent_mcp_servers_registered()

    checked_tool_ids: set[str] = set()
    for server_name in NorthCUAMCPRegistry.names():
        for contract in NorthCUAMCPRegistry.tool_contracts(server_name):
            if not contract.model_visible:
                continue
            checked_tool_ids.add(f"{server_name}.{contract.tool_id}")
            assert contract.description
            _assert_provider_schema_is_self_contained(
                contract.provider_parameters_schema,
                allow_unions=contract.provider_schema_mode == ProviderSchemaMode.UNION,
            )

    assert checked_tool_ids


def test_data_and_a11y_contract_validators_reject_extra_fields() -> None:
    ensure_builtin_mcp_servers_registered()

    data_contract = next(
        contract
        for contract in NorthCUAMCPRegistry.tool_contracts(DESKTOP_NATIVE_DATA_SERVER_NAME)
        if contract.tool_id == "capture_window_screenshot"
    )
    a11y_contract = next(
        contract
        for contract in NorthCUAMCPRegistry.tool_contracts(DESKTOP_NATIVE_A11Y_SERVER_NAME)
        if contract.tool_id == "get_accessibility_capabilities"
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_contract_parameters(data_contract, {"unexpected": True})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_contract_parameters(a11y_contract, {"unexpected": True})


def test_contract_validator_rejects_nested_extra_fields() -> None:
    ensure_builtin_mcp_servers_registered()

    drag_contract = next(
        contract
        for contract in NorthCUAMCPRegistry.tool_contracts(DESKTOP_NATIVE_ACTION_SERVER_NAME)
        if contract.tool_id == "drag"
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_contract_parameters(
            drag_contract,
            {
                "path": [
                    {"x": 1, "y": 2, "z": 3},
                    {"x": 4, "y": 5},
                ],
            },
        )


def _assert_provider_schema_is_self_contained(value: object, *, allow_unions: bool = False) -> None:
    if isinstance(value, list):
        for item in value:
            _assert_provider_schema_is_self_contained(item, allow_unions=allow_unions)
        return
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        key_str = str(key)
        allowed = {"oneOf", "anyOf"} if allow_unions else set()
        assert key_str not in _PROVIDER_UNSUPPORTED_SCHEMA_KEYS - allowed
        if key_str == "$ref":
            raise AssertionError(f"Provider schema contains unresolved $ref: {child!r}")
        _assert_provider_schema_is_self_contained(child, allow_unions=allow_unions)
