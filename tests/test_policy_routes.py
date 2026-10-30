from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.archs.llm.llm_config import LLMConfig

from northcua.agentinfra.providers.llm_fallback import LLM_FALLBACKS_STORAGE_KEY
from northcua.agentinfra.runtime.protocol import AgentKind
from northcua.core.config_loader import NorthCUAConfigLoader
from northcua.policy_routes import FilePolicyRouteResolver, PolicyRoute, apply_task_policy_route

_ROUTE_REF = "nexcua-policy-route://session/member-1"


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    credential = tmp_path / "member-1.credential"
    credential.write_text("temporary-member-token\n", encoding="utf-8")
    credential.chmod(0o600)
    manifest = tmp_path / "routes.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "routes": {
                    _ROUTE_REF: {
                        "route_ref": _ROUTE_REF,
                        "base_url": "http://train-head:18001/v1",
                        "model": "Qwen3.5-0.8B",
                        "api_type": "openai_chat_completion",
                        "credential_file": credential.name,
                        "sampling_params": {
                            "temperature": 0.7,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_new_tokens": 64,
                            "sampling_seed": 9,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    return manifest, credential


def _app() -> SimpleNamespace:
    return SimpleNamespace(
        agent_config=SimpleNamespace(
            llm_config=LLMConfig(
                model="bootstrap-model",
                base_url="https://bootstrap.example.invalid/v1",
                api_key="bootstrap-key",
                max_context_tokens=200_000,
                reasoning={"effort": "xhigh"},
                include=["reasoning.encrypted_content"],
                supports_images=True,
            ),
            global_storage={LLM_FALLBACKS_STORAGE_KEY: ("must-not-bypass-training-route",), "keep": True},
        )
    )


def test_file_route_stays_controller_local_and_disables_fallbacks(tmp_path: Path) -> None:
    manifest, _credential = _runtime_files(tmp_path)
    app = _app()

    assert apply_task_policy_route(
        app,
        {"policy_route_ref": _ROUTE_REF},
        resolver=FilePolicyRouteResolver(manifest),
    )

    routed = app.agent_config.llm_config
    assert routed.base_url == "http://train-head:18001/v1"
    assert routed.model == "Qwen3.5-0.8B"
    assert routed.api_key == "temporary-member-token"
    assert routed.temperature == 0.7
    assert routed.top_p == 1.0
    assert routed.max_tokens == 64
    assert routed.extra_params == {
        "max_context_tokens": 200_000,
        "supports_images": True,
        "extra_body": {"top_k": -1, "sampling_seed": 9},
    }
    assert LLM_FALLBACKS_STORAGE_KEY not in app.agent_config.global_storage
    assert app.agent_config.global_storage["keep"] is True


def test_route_files_must_be_owner_only(tmp_path: Path) -> None:
    manifest, credential = _runtime_files(tmp_path)
    manifest.chmod(0o644)
    with pytest.raises(PermissionError, match="manifest"):
        FilePolicyRouteResolver(manifest).resolve(_ROUTE_REF)

    manifest.chmod(0o600)
    credential.chmod(0o640)
    with pytest.raises(PermissionError, match="credential"):
        FilePolicyRouteResolver(manifest).resolve(_ROUTE_REF)


def test_missing_or_drifting_route_fails_closed(tmp_path: Path) -> None:
    manifest, _credential = _runtime_files(tmp_path)
    with pytest.raises(LookupError, match="unknown policy route"):
        FilePolicyRouteResolver(manifest).resolve("nexcua-policy-route://missing")

    class _DriftingResolver:
        def resolve(self, route_ref: str) -> PolicyRoute:
            del route_ref
            return PolicyRoute(
                route_ref="nexcua-policy-route://different",
                base_url="http://train-head:18001/v1",
                model="Qwen3.5-0.8B",
                api_key="token",
            )

    with pytest.raises(ValueError, match="different route"):
        apply_task_policy_route(_app(), {"policy_route_ref": _ROUTE_REF}, resolver=_DriftingResolver())


def test_no_route_metadata_leaves_bootstrap_model_unchanged() -> None:
    app = _app()
    original = app.agent_config.llm_config
    assert not apply_task_policy_route(app, {})
    assert app.agent_config.llm_config is original


def test_policy_routed_controller_config_loads_without_global_provider_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller bootstrap must not require the policy server credential."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "tests/fixtures/configs/app.policy-routed.yaml"

    desktop = NorthCUAConfigLoader.from_app_config(app_path, agent_kind=AgentKind.DESKTOP)
    web = NorthCUAConfigLoader.from_app_config(app_path, agent_kind=AgentKind.WEB)

    assert desktop.agent.llm_config.api_key == "policy-route-required"
    assert desktop.agent.llm_config.base_url == "http://127.0.0.1:9/v1"
    assert desktop.app.desktop.coordinate_policy.input_space.value == "normalized_1000"
    assert desktop.app.desktop.coordinate_policy.clamp_to_bounds is True
    assert desktop.agent.llm_config.extra_params["reasoning_replay_policy"] == "chat_reasoning_content"
    assert desktop.agent.llm_config.max_tokens == 4096
    assert [server["name"] for server in desktop.app.agent_mcp_servers if server.get("model_visible")] == [
        "DesktopNativeActionExecutor"
    ]
    assert web.agent.llm_config.api_key == "policy-route-required"
    assert web.agent.llm_config.base_url == "http://127.0.0.1:9/v1"
    assert web.agent.llm_config.max_tokens == 4096
    assert web.agent.max_context_tokens == 262144
    assert web.agent.llm_config.extra_params["reasoning_replay_policy"] == "chat_reasoning_content"


def test_policy_route_keeps_chat_reasoning_history_replay(tmp_path: Path) -> None:
    manifest, _credential = _runtime_files(tmp_path)
    app = _app()
    app.agent_config.llm_config.extra_params = {
        **app.agent_config.llm_config.extra_params,
        "reasoning_replay_policy": "chat_reasoning_content",
    }

    assert apply_task_policy_route(
        app,
        {"policy_route_ref": _ROUTE_REF},
        resolver=FilePolicyRouteResolver(manifest),
    )

    assert app.agent_config.llm_config.extra_params["reasoning_replay_policy"] == "chat_reasoning_content"


def test_route_rejects_unknown_or_nonfinite_sampling_values(tmp_path: Path) -> None:
    manifest, _credential = _runtime_files(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["routes"][_ROUTE_REF]["sampling_params"] = {"unknown": 1}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o600)

    with pytest.raises(ValueError, match="unsupported task policy sampling"):
        FilePolicyRouteResolver(manifest).resolve(_ROUTE_REF)
