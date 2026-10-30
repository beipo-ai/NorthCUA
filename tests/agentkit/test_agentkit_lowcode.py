from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from pydantic import BaseModel

from northcua.agentkit import (
    AgentArtifactPaths,
    AgentCliCommandSpec,
    AgentCliSpec,
    AgentPresentationSpec,
    AgentRemoteSpec,
    AgentRenderContext,
    AgentRuntimeSpec,
    AgentSessionSpec,
    AgentSpec,
    ArtifactLayoutSpec,
    ObservationProvider,
    ProbeBundle,
    PromptProjector,
    SessionAdapter,
    StepwiseToolAgentTemplate,
    StepwiseToolStepContext,
    StepwiseToolStepResult,
    TemplateTerminalRenderer,
    ToolBundle,
    ToolUseRef,
    cua_probe,
    cua_tool,
    resolve_panel_projector,
    resolve_terminal_renderer,
    run_with_terminal_renderer,
)
from northcua.agentkit.agent_config import clone_agent_config_with_hooks
from northcua.agentkit.cli import GeneratedAgentCliAdapter
from northcua.agentkit.controller_dispatch import command_result_record
from northcua.agentkit.controller_runner import (
    AgentKitRemoteRunnerConfig,
    register_controller_runner,
)
from northcua.agentkit.device_adapter import AgentKitMCPRuntimeAdapter
from northcua.agentkit.external_tools import tool_execution_records_from_command_results
from northcua.agentkit.registry import get_agentkit_registry, load_all_agent_specs, reset_agentkit_registry_for_tests
from northcua.agentkit.session_context import AGENTKIT_SETTINGS_PARAM
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.session import RegisteredWebAgentSession, WebAgentSessionRegistry
from northcua.agents.web.session.agentkit_adapter import WebAgentDeviceSessionAdapter
from northcua.controller.execution.executor_protocol import ControllerExecutionContext
from northcua.controller.execution.runner_plugins import ControllerRunnerRegistrationContext
from northcua.controller.execution.runner_registry import RunnerRegistry
from northcua.core.config import NorthCUAAppConfig
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.protocol import GenericTaskRequestPayload, parse_task_request_payload
from northcua.submitter import DesktopSubmitSpec
from northcua_controller_plugins.agentkit_web_remote_dispatch import RemoteWebActionBackend
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.capabilities import Capability, PlatformKind
from northcua_protocol.payloads import AgentKind as ProtocolAgentKind
from northcua_protocol.payloads import CommandRequest, CommandResult, SessionLeaseGranted, TaskRequest
from northcua_protocol.state import CommandState


class TapParams(BaseModel):
    x: int
    y: int


class AndroidSettings(BaseModel):
    adb_binary: str = "adb"
    max_steps: int = 20


class AndroidTools(ToolBundle):
    @cua_tool("android_tap", params=TapParams, capability=Capability.DESKTOP_INPUT)
    async def tap(self, params: TapParams) -> dict[str, object]:
        return {"ok": True, "x": params.x, "y": params.y}


class AndroidProbes(ProbeBundle):
    @cua_probe("android_screenshot", capability=Capability.DESKTOP_SCREENSHOT)
    async def screenshot(self) -> dict[str, object]:
        return {"path": "screenshot.png"}


class AndroidObservationProvider(ObservationProvider):
    async def collect(self, *, session: object, step_index: int) -> dict[str, object]:
        del session
        return {"screen": f"settings-{step_index}"}


class AndroidPromptProjector(PromptProjector):
    def project(self, *, query: str, observation: object, history: tuple[object, ...] = ()) -> list[dict[str, object]]:
        return [
            {"role": "system", "content": "You are controlling an Android test device."},
            {"role": "user", "content": {"query": query, "observation": observation, "history": len(history)}},
        ]


class AndroidSessionAdapter(SessionAdapter):
    calls: list[str] = []

    def ensure_device_session(self, *, command: object, artifact_root: Path) -> None:
        del artifact_root
        AndroidSessionAdapter.calls.append(getattr(command, "session_id", ""))


class AndroidPanelProjector:
    def project(self, step: object) -> dict[str, object]:
        return {"step": 7, "status": "ok", "summary": f"projected {step}"}


def test_lowcode_decorators_attach_tool_contract_source() -> None:
    tool_spec = AndroidTools.declared_tools()[0]
    probe_spec = AndroidProbes.declared_probes()[0]

    assert tool_spec.contract is not None
    assert tool_spec.params_model is TapParams
    assert tool_spec.parameters_schema == tool_spec.contract.parameters_schema
    assert probe_spec.contract is not None
    assert probe_spec.parameters_schema == probe_spec.contract.parameters_schema


def test_generated_cli_adapter_registers_spec_namespace() -> None:
    adapter = GeneratedAgentCliAdapter(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(
                agent_name="android",
                help="Android CUA commands",
                local_commands=(
                    AgentCliCommandSpec(
                        name="run",
                        help="Run Android CUA",
                        handler="tests.agentkit.test_agentkit_lowcode:_dummy_handler",
                    ),
                ),
            ),
        )
    )
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="agent_kind", required=True)

    adapter.register_local_commands(subparsers, default_config_help="config/app.yaml")
    args = parser.parse_args(["android", "run"])

    assert args.agent_kind == "android"
    assert args.agent_action == "run"


def test_generated_cli_adapter_dispatch_projects_named_arguments() -> None:
    _DISPATCH_CALLS.clear()
    adapter = GeneratedAgentCliAdapter(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(
                agent_name="android",
                help="Android CUA commands",
                local_commands=(
                    AgentCliCommandSpec(
                        name="run",
                        help="Run Android CUA",
                        handler=_recording_handler,
                    ),
                ),
            ),
        )
    )

    async def _run() -> None:
        code = await adapter.dispatch(
            argparse.Namespace(command_group="local", agent_action="run", query="hello", session_id="sess-android")
        )
        assert code == 7

    asyncio.run(_run())

    assert _DISPATCH_CALLS == [{"query": "hello", "session_id": "sess-android"}]


async def _dummy_handler(args: argparse.Namespace) -> int:
    del args
    return 0


_DISPATCH_CALLS: list[dict[str, object]] = []


async def _recording_handler(query: str, session_id: str) -> int:
    _DISPATCH_CALLS.append({"query": query, "session_id": session_id})
    return 7


def _dummy_runner_factory(*, spec: AgentSpec, config: AgentKitRemoteRunnerConfig, sdk: object) -> Any:
    del spec, config, sdk

    async def _run(context: object) -> object:
        del context
        raise AssertionError("dummy runner should not be invoked by registration test")

    return _run
