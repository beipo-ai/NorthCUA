from __future__ import annotations

from northcua.agentinfra.runtime.application import AgentApplicationBase
from northcua.agentinfra.runtime.artifacts import AgentArtifactStoreBase, ArtifactStoreFactory, BrowserArtifactStoreBase
from northcua.agentinfra.runtime.contracts import BrowserStateProvider
from northcua.agentinfra.runtime.protocol import (
    AgentKind,
    BrowserStepEvent,
    DesktopRunRequest,
    DesktopStepEvent,
    WebRunRequest,
    WebStepEvent,
)
from northcua.agentkit.cli import GeneratedAgentCliAdapter
from northcua.agentkit.registry import load_all_agent_specs
from northcua.agents.desktop.application import DesktopApplication
from northcua.agents.web.application import WebAgentApplication
from northcua.agents.web.artifacts import WebArtifactStore
from northcua.cli import resolve_cli_adapter
from northcua.core.config import ArtifactConfig
from northcua.core.enums import TaskStatus
from northcua.core.models import LLMTraceRecord, StepLogRecord
from northcua.platforms import PlatformKind
from northcua.runtime.artifacts.store import NorthCUAArtifactStore


def test_internal_applications_and_artifact_stores_share_common_bases() -> None:
    assert issubclass(DesktopApplication, AgentApplicationBase)
    assert issubclass(WebAgentApplication, AgentApplicationBase)
    assert issubclass(NorthCUAArtifactStore, AgentArtifactStoreBase)
    assert issubclass(WebArtifactStore, BrowserArtifactStoreBase)
    assert issubclass(BrowserArtifactStoreBase, AgentArtifactStoreBase)


def test_cli_adapter_resolution_uses_registered_agent_commands() -> None:
    assert isinstance(resolve_cli_adapter("desktop"), GeneratedAgentCliAdapter)
    assert isinstance(resolve_cli_adapter("web"), GeneratedAgentCliAdapter)
    assert resolve_cli_adapter("not-a-command") is None


def test_agentkit_specs_are_source_of_truth_for_builtins() -> None:
    specs = {spec.kind: spec for spec in load_all_agent_specs().enabled()}

    assert set(specs) >= {"desktop", "web"}
    assert specs["web"].remote.enabled is True
    assert specs["desktop"].cli.agent_name == "desktop"


def test_artifact_store_factory_creates_session_scoped_store(tmp_path) -> None:
    store = ArtifactStoreFactory.create_session_store(
        NorthCUAArtifactStore,
        root_dir=tmp_path,
        session_id="agentinfra-session",
        config=ArtifactConfig(),
    )

    assert store.root.name == "agentinfra-session"
    assert store.primary_trace_path.name == "steps.jsonl"


def test_step_event_adapters_build_typed_events() -> None:
    desktop_event = DesktopStepEvent.from_step_log(
        StepLogRecord(
            session_id="sess",
            session_step=3,
            round_step=1,
            round_num=2,
            status=TaskStatus.CONTINUE,
            request="Open Calculator",
            result="Still working",
            llm_trace=LLMTraceRecord(),
            assistant_text="I will click the calculator icon.",
        )
    )
    browser_event = BrowserStepEvent(
        agent_kind=AgentKind.WEB, step_index=1, status="ok", summary="reserved", url="https://example.com"
    )
    web_step_event = WebStepEvent(
        agent_kind=AgentKind.WEB,
        step_index=2,
        status="ok",
        summary="clicked",
        url="https://example.com",
        action="web_click_ref",
    )

    assert desktop_event.agent_kind == AgentKind.DESKTOP
    assert desktop_event.tool_count == 0
    assert desktop_event.summary == "Still working"
    assert browser_event.agent_kind == AgentKind.WEB
    assert web_step_event.action == "web_click_ref"


def test_desktop_step_event_does_not_fallback_to_internal_request_context() -> None:
    desktop_event = DesktopStepEvent.from_step_log(
        StepLogRecord(
            session_id="sess",
            session_step=3,
            round_step=1,
            round_num=2,
            status=TaskStatus.CONTINUE,
            request=("NorthCUA CUA target context:\n- pid: 123\n- window_id: 456\n\nUser task:\nClick 1 + 2 =."),
            result="",
            llm_trace=LLMTraceRecord(),
            assistant_text="",
        )
    )

    assert desktop_event.summary == ""
    assert "NorthCUA CUA target context" not in desktop_event.summary


def test_reserved_browser_interfaces_are_exposed() -> None:
    assert BrowserStateProvider is not None


def test_typed_run_requests_expose_stable_agent_kinds() -> None:
    desktop_request = DesktopRunRequest(session_id="sess", query="Open Calculator")
    assert desktop_request.agent_kind == AgentKind.DESKTOP
    assert desktop_request.max_steps == 100
    assert WebRunRequest(session_id="sess", query="Find pricing", url="https://example.com").agent_kind == AgentKind.WEB
