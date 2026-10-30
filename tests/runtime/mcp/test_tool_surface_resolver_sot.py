from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import BaseModel

from northcua.agentkernel.tools.surface import ToolSurfaceService
from northcua.runtime.mcp.contracts import external_mcp_snapshot_contracts
from northcua_protocol.capabilities import ToolRiskLevel
from northcua_runtime_shared.tool_contracts import (
    DirectHandlerBinding,
    ExternalMcpSnapshotBinding,
    McpBinding,
    ToolAudience,
    ToolContract,
    ToolDispatcher,
    ToolDispatchPolicy,
    ToolPlacement,
    ToolProfile,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    ToolSurfaceResolver,
    ToolVisibility,
    fingerprint_json,
    provider_schema_projection,
)


class _EchoParams(BaseModel):
    text: str


def _contract(
    tool_id: str,
    *,
    public_name: str = "",
    visibility: ToolVisibility = ToolVisibility.MODEL_VISIBLE,
    json_schema: dict[str, object] | None = None,
    binding: object | None = None,
    imported: bool = False,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        public_name=public_name,
        description=f"{tool_id} description",
        namespace="test",
        audience=ToolAudience.AGENT,
        placement=ToolPlacement.MCP_SERVER,
        visibility=visibility,
        params_model=None if json_schema is not None else _EchoParams,
        json_schema=json_schema,
        side_effect=True,
        binding=binding if binding is not None else McpBinding(server_name="TestExecutor"),
        imported=imported,
    )


def test_resolver_projects_validation_provider_routes_and_visibility_from_contracts() -> None:
    model_tool = _contract("model_tool", binding=McpBinding(server_name="ModelExecutor"))
    runtime_tool = _contract("runtime_tool", visibility=ToolVisibility.RUNTIME_ONLY)
    surface = ToolSurfaceResolver((model_tool, runtime_tool)).resolve(
        ToolProfile(name="agent.test", namespaces=frozenset({"test"})),
        mode=ToolSurfaceMode.REMOTE_EXTERNALIZED,
    )

    assert [descriptor.tool_id for descriptor in surface.validation_descriptors] == ["model_tool", "runtime_tool"]
    assert [tool.name for tool in surface.provider_tools] == ["model_tool"]
    assert surface.route_by_tool_id["model_tool"] == "ModelExecutor"
    assert surface.visibility["runtime_tool"] == ToolVisibility.RUNTIME_ONLY
    assert surface.fingerprints["contract_summary_hash"]
    assert surface.fingerprints["final_prompt_surface_hash"]


def test_resolver_routes_public_name_and_canonical_id_to_same_server() -> None:
    contract = _contract("canonical_tool", public_name="public_tool", binding=McpBinding(server_name="PublicExecutor"))

    surface = ToolSurfaceResolver((contract,)).resolve(ToolProfile(name="agent.test"))

    assert [tool.name for tool in surface.provider_tools] == ["public_tool"]
    assert surface.route_by_tool_id["canonical_tool"] == "PublicExecutor"
    assert surface.route_by_tool_id["public_tool"] == "PublicExecutor"


def test_resolver_rejects_profile_public_name_drift() -> None:
    first = _contract("first", public_name="same")
    second = _contract("second", public_name="same")

    with pytest.raises(ValueError, match="Conflicting tool contract projection"):
        ToolSurfaceResolver((first, second)).resolve(ToolProfile(name="agent.test"))


def test_resolver_rejects_alias_drift() -> None:
    first = replace(_contract("first"), aliases=("shared",))
    second = replace(_contract("second"), aliases=("shared",))

    with pytest.raises(ValueError, match="Tool alias 'shared' maps to both"):
        ToolSurfaceResolver((first, second)).resolve(ToolProfile(name="agent.test"))


def test_resolver_rejects_canonical_tool_id_drift() -> None:
    first = _contract("same", public_name="first", binding=McpBinding(server_name="A"))
    second = _contract("same", public_name="second", binding=McpBinding(server_name="B"))

    with pytest.raises(ValueError, match="Conflicting tool contract identity"):
        ToolSurfaceResolver((first, second)).resolve(ToolProfile(name="agent.test"))


def test_resolver_rejects_alias_colliding_with_canonical_tool_id() -> None:
    first = replace(_contract("first"), aliases=("second",))
    second = _contract("second", public_name="second_public")

    with pytest.raises(ValueError, match="conflicts with canonical tool id"):
        ToolSurfaceResolver((first, second)).resolve(ToolProfile(name="agent.test"))


def test_external_mcp_snapshot_is_not_selected_without_explicit_imported_opt_in() -> None:
    external = _contract(
        "external_search",
        binding=ExternalMcpSnapshotBinding(
            server_name="ExternalSearch",
            source="mcp://external/search",
            schema_fingerprint="abc",
        ),
        imported=True,
    )

    default_surface = ToolSurfaceResolver((external,)).resolve(ToolProfile(name="agent.test"))
    imported_surface = ToolSurfaceResolver((external,)).resolve(
        ToolProfile(name="agent.test", include_imported=True),
    )
    config_imported_surface = ToolSurfaceResolver((external,)).resolve(config=ToolSurfaceConfig(include_imported=True))

    assert default_surface.contracts == ()
    assert [tool.name for tool in imported_surface.provider_tools] == ["external_search"]
    assert [tool.name for tool in config_imported_surface.provider_tools] == ["external_search"]


def test_external_mcp_discovery_snapshot_contracts_require_imported_opt_in() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    contracts = external_mcp_snapshot_contracts(
        [
            {
                "server_name": "SearchMCP",
                "name": "search_web",
                "description": "Search the web.",
                "input_schema": schema,
                "schema_fingerprint": fingerprint_json(schema),
                "source": "mcp://search",
            }
        ]
    )
    contract = contracts[0]

    assert contract.imported is True
    assert contract.binding is not None
    assert contract.binding.server_name == "SearchMCP"
    assert contract.binding.schema_fingerprint == fingerprint_json(contract.parameters_schema)
    default_surface = ToolSurfaceResolver(contracts).resolve(ToolProfile(name="agent.test"))
    imported_surface = ToolSurfaceResolver(contracts).resolve(ToolProfile(name="agent.test", include_imported=True))

    assert default_surface.provider_tools == ()
    assert [tool.name for tool in imported_surface.provider_tools] == ["search_web"]


def test_external_mcp_snapshot_fingerprint_tamper_is_detectable() -> None:
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    contract = external_mcp_snapshot_contracts(
        [
            {
                "server_name": "SearchMCP",
                "name": "search_web",
                "input_schema": schema,
                "schema_fingerprint": fingerprint_json(schema),
                "source": "mcp://search",
            }
        ]
    )[0]

    assert contract.binding is not None
    assert contract.binding.schema_fingerprint != "abc"


def test_external_mcp_snapshot_rejects_declared_schema_fingerprint_mismatch() -> None:
    with pytest.raises(ValueError, match="schema fingerprint mismatch"):
        external_mcp_snapshot_contracts(
            [
                {
                    "server_name": "SearchMCP",
                    "name": "search_web",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "schema_fingerprint": "tampered",
                    "source": "mcp://search",
                }
            ]
        )


def test_external_mcp_snapshot_sanitizes_source_before_summary_and_binding() -> None:
    schema = {"type": "object"}
    contract = external_mcp_snapshot_contracts(
        [
            {
                "server_name": "SearchMCP",
                "name": "search_web",
                "input_schema": schema,
                "schema_fingerprint": fingerprint_json(schema),
                "source": "https://user:secret@example.test/mcp?token=abc#fragment",
            }
        ]
    )[0]

    assert contract.source == "https://example.test/mcp"
    assert contract.binding is not None
    assert contract.binding.source == "https://example.test/mcp"
    assert "secret" not in str(contract.to_summary())
    assert "token" not in str(contract.to_summary())


def test_resolver_dedupes_equivalent_contracts_before_provider_projection() -> None:
    contract = _contract("duplicate")
    surface = ToolSurfaceResolver((contract, contract)).resolve(ToolProfile(name="agent.test"))

    assert [contract.tool_id for contract in surface.contracts] == ["duplicate"]
    assert [tool.name for tool in surface.provider_tools] == ["duplicate"]


def test_surface_fingerprints_change_for_schema_route_and_timeout_mutations() -> None:
    base = _contract("mutable", binding=McpBinding(server_name="ServerA"))
    baseline = ToolSurfaceResolver((base,)).resolve(ToolProfile(name="agent.test")).fingerprints

    schema_mutation = replace(
        base,
        json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "mode": {"type": "string"}},
            "required": ["text", "mode"],
        },
        params_model=None,
    )
    route_mutation = replace(base, binding=McpBinding(server_name="ServerB"))
    timeout_mutation = replace(base, timeout_seconds=base.timeout_seconds + 1)

    schema_hashes = ToolSurfaceResolver((schema_mutation,)).resolve(ToolProfile(name="agent.test")).fingerprints
    route_hashes = ToolSurfaceResolver((route_mutation,)).resolve(ToolProfile(name="agent.test")).fingerprints
    timeout_hashes = ToolSurfaceResolver((timeout_mutation,)).resolve(ToolProfile(name="agent.test")).fingerprints

    assert schema_hashes["contract_summary_hash"] != baseline["contract_summary_hash"]
    assert schema_hashes["protocol_schema_hash"] != baseline["protocol_schema_hash"]
    assert schema_hashes["provider_schema_hash"] != baseline["provider_schema_hash"]
    assert route_hashes["route_hash"] != baseline["route_hash"]
    assert timeout_hashes["contract_summary_hash"] != baseline["contract_summary_hash"]


def test_provider_projection_records_lossiness_manifest() -> None:
    projection = provider_schema_projection(
        {
            "type": "object",
            "properties": {
                "target": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                }
            },
            "dependentRequired": {"target": ["mode"]},
        }
    )

    assert projection.lossy is True
    assert "$.dependentRequired" in projection.lossiness_manifest
    assert "$.properties.target.anyOf" in projection.lossiness_manifest
    assert "dependentRequired" not in projection.schema


def test_provider_projection_can_preserve_structural_unions_explicitly() -> None:
    schema = {
        "type": "object",
        "oneOf": [
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            {"type": "object", "properties": {"duration_ms": {"type": "integer"}}, "required": ["duration_ms"]},
        ],
    }

    projection = provider_schema_projection(schema, preserve_unions=True)

    assert projection.schema["oneOf"] == schema["oneOf"]
    assert "$.oneOf" not in projection.lossiness_manifest


def test_dispatcher_validates_full_schema_after_provider_projection_strips_constraints() -> None:
    contract = _contract(
        "dependent_tool",
        json_schema={
            "type": "object",
            "properties": {
                "window_id": {"type": "integer"},
                "pid": {"type": "integer"},
            },
            "dependentRequired": {"window_id": ["pid"]},
            "additionalProperties": False,
        },
        binding=DirectHandlerBinding(handler_path="tests:dependent_tool"),
    )

    assert "dependentRequired" not in contract.provider_parameters_schema
    with pytest.raises(ValueError, match="missing dependent parameters"):
        ToolDispatcher().prepare(contract, {"window_id": 1})
    envelope = ToolDispatcher().prepare(contract, {"window_id": 1, "pid": 2})
    assert envelope.route["handler_path"] == "tests:dependent_tool"


def test_dispatcher_rejects_undeclared_binding_override() -> None:
    contract = _contract("routed_tool", binding=McpBinding(server_name="DeclaredExecutor"))

    with pytest.raises(LookupError, match="binding is not declared"):
        ToolDispatcher().prepare(
            contract,
            {"text": "hi"},
            binding=McpBinding(server_name="UndeclaredExecutor"),
        )

    envelope = ToolDispatcher().prepare(
        contract,
        {"text": "hi"},
        binding=McpBinding(server_name="DeclaredExecutor"),
    )

    assert envelope.route["server_name"] == "DeclaredExecutor"


def test_dispatcher_enforces_visibility_readonly_and_payload_gates() -> None:
    operator_tool = _contract("operator_tool", visibility=ToolVisibility.OPERATOR_ONLY)
    readonly_tool = _contract("readonly_tool", visibility=ToolVisibility.MODEL_VISIBLE)
    readonly_tool = replace(readonly_tool, side_effect=False, readonly=True, max_payload_bytes=12)

    dispatcher = ToolDispatcher()
    with pytest.raises(PermissionError, match="operator-only"):
        dispatcher.prepare(operator_tool, {"text": "hi"})
    dispatcher.prepare(operator_tool, {"text": "hi"}, policy=ToolDispatchPolicy(allow_operator=True))

    with pytest.raises(ValueError, match="payload exceeds"):
        dispatcher.prepare(readonly_tool, {"text": "payload too large"})
    with pytest.raises(ValueError, match="payload exceeds"):
        dispatcher.prepare(
            readonly_tool,
            {"text": "payload too large"},
            policy=ToolDispatchPolicy(max_payload_bytes=10_000),
        )
    with pytest.raises(PermissionError, match="mutating"):
        dispatcher.prepare(_contract("mutating"), {"text": "hi"}, policy=ToolDispatchPolicy(readonly_required=True))


def test_dispatcher_enforces_max_allowed_risk_gate() -> None:
    high_risk_tool = replace(_contract("danger"), risk_level=ToolRiskLevel.HIGH)

    with pytest.raises(PermissionError, match="risk level 'high' exceeds allowed maximum 'medium'"):
        ToolDispatcher().prepare(
            high_risk_tool,
            {"text": "hi"},
            policy=ToolDispatchPolicy(max_allowed_risk=ToolRiskLevel.MEDIUM),
        )

    ToolDispatcher().prepare(
        high_risk_tool,
        {"text": "hi"},
        policy=ToolDispatchPolicy(max_allowed_risk=ToolRiskLevel.HIGH),
    )


def test_tool_surface_service_can_build_provider_payload_from_resolved_contract_surface() -> None:
    surface = ToolSurfaceResolver((_contract("model_tool"),)).resolve(ToolProfile(name="agent.test"))

    for api_type in ("openai_responses", "openai_chat_completion", "anthropic_chat_completion"):
        provider_surface = ToolSurfaceService().build_from_resolved_surface(surface, api_type=api_type)

        assert provider_surface.tools
        assert provider_surface.tool_choice is not None


def test_tool_surface_service_provider_payload_shapes_are_provider_specific() -> None:
    surface = ToolSurfaceResolver((_contract("model_tool"),)).resolve(ToolProfile(name="agent.test"))

    responses = ToolSurfaceService().build_from_resolved_surface(surface, api_type="openai_responses")
    chat = ToolSurfaceService().build_from_resolved_surface(surface, api_type="openai_chat_completion")
    anthropic = ToolSurfaceService().build_from_resolved_surface(surface, api_type="anthropic_chat_completion")

    assert responses.tool_choice == "auto"
    assert responses.tools == [
        {
            "type": "function",
            "name": "model_tool",
            "description": "model_tool description",
            "parameters": {
                "properties": {"text": {"title": "Text", "type": "string"}},
                "required": ["text"],
                "title": "_EchoParams",
                "type": "object",
            },
        }
    ]
    assert chat.tool_choice == "auto"
    assert chat.tools == [
        {
            "type": "function",
            "function": {
                "name": "model_tool",
                "description": "model_tool description",
                "parameters": {
                    "properties": {"text": {"title": "Text", "type": "string"}},
                    "required": ["text"],
                    "title": "_EchoParams",
                    "type": "object",
                },
            },
        }
    ]
    assert anthropic.tool_choice == {"type": "auto"}
    assert anthropic.tools == [
        {
            "name": "model_tool",
            "description": "model_tool description",
            "input_schema": {
                "properties": {"text": {"title": "Text", "type": "string"}},
                "required": ["text"],
                "title": "_EchoParams",
                "type": "object",
            },
            "eager_input_streaming": True,
        }
    ]


def test_tool_surface_service_translates_required_policy_to_provider_wire_shape() -> None:
    surface = ToolSurfaceResolver((_contract("model_tool"),)).resolve(ToolProfile(name="agent.test"))
    service = ToolSurfaceService()

    chat = service.build_from_resolved_surface(
        surface,
        api_type="openai_chat_completion",
        tool_choice_policy="required",
    )
    anthropic = service.build_from_resolved_surface(
        surface,
        api_type="anthropic_chat_completion",
        tool_choice_policy="required",
    )

    assert chat.tool_choice == "required"
    assert anthropic.tool_choice == {"type": "any"}


def test_tool_surface_service_rejects_unknown_or_unsupported_required_policy() -> None:
    surface = ToolSurfaceResolver((_contract("model_tool"),)).resolve(ToolProfile(name="agent.test"))
    service = ToolSurfaceService()

    with pytest.raises(ValueError, match="unsupported tool choice policy"):
        service.build_from_resolved_surface(surface, api_type="openai_chat_completion", tool_choice_policy="always")
    with pytest.raises(ValueError, match="unsupported for provider API"):
        service.build_from_resolved_surface(surface, api_type="gemini_rest", tool_choice_policy="required")


def test_tool_surface_service_rejects_required_policy_without_visible_tools() -> None:
    surface = ToolSurfaceResolver(()).resolve(ToolProfile(name="agent.empty"))

    with pytest.raises(ValueError, match="needs at least one model-visible tool"):
        ToolSurfaceService().build_from_resolved_surface(
            surface,
            api_type="openai_chat_completion",
            tool_choice_policy="required",
        )
