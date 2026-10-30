from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.device_adapter import AgentKitMCPRuntimeAdapter
from northcua.agentkit.session_context import AGENTKIT_SETTINGS_PARAM
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.domain.actions import WebAgentActionType
from northcua.agents.web.domain.observation_recovery import (
    WebObservationRecoveryInvariantError,
    WebObservationRecoveryMode,
    WebObservationRecoveryReason,
)
from northcua.agents.web.domain.observations import (
    WebElementRef,
    WebObservationContext,
    WebScreenshotObservation,
    WebSnapshotObservation,
)
from northcua.agents.web.mcp.action_service import WebAgentMCPActionService
from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope
from northcua.agents.web.prompt.history import WebHistoryToolResultProjector
from northcua.agents.web.prompt.ref_window import (
    REFS_INVALIDATED_NOTICE,
    WebRefSnapshotWindowBudgeter,
)
from northcua.agents.web.runner.action_resolution import ResolvedWebAgentAction
from northcua.agents.web.runner.observation_recovery import (
    WebObservationRecoveryPolicy,
    WebObservationRecoveryService,
)
from northcua.agents.web.session import RegisteredWebAgentSession, WebAgentSessionRegistry
from northcua.core.config import WebAgentConfig
from northcua.core.models import AssistantToolCallRecord, ToolExecutionRecord
from northcua_controller_plugins.agentkit_web_remote_dispatch import RemoteWebActionBackend
from northcua_controller_plugins.agentkit_web_runner import _web_ref_alias_side_channel
from northcua_protocol.payloads import CommandRequest, CommandResult
from northcua_protocol.state import CommandState


def _action(
    *,
    action: WebAgentActionType = WebAgentActionType.CLICK_REF,
    output: dict[str, object] | None = None,
    is_error: bool = False,
    call_id: str = "call-real",
) -> ResolvedWebAgentAction:
    return ResolvedWebAgentAction(
        action=action,
        tool_name=f"web_{action.value}",
        tool_input={},
        tool_output=output or {"status": "ok"},
        is_error=is_error,
        tool_call_id=call_id,
    )


@pytest.mark.parametrize(
    ("code", "action", "expected_mode"),
    [
        ("invalid_params", WebAgentActionType.WAIT, WebObservationRecoveryMode.REUSE_CURRENT),
        ("invalid_wait_target_combination", WebAgentActionType.WAIT, WebObservationRecoveryMode.REUSE_CURRENT),
        ("invalid_url_scheme", WebAgentActionType.OPEN, WebObservationRecoveryMode.REUSE_CURRENT),
        ("wrong_action_for_element", WebAgentActionType.SELECT_OPTION_REF, WebObservationRecoveryMode.REUSE_CURRENT),
        ("element_disabled", WebAgentActionType.CLICK_REF, WebObservationRecoveryMode.REUSE_CURRENT),
        ("element_not_receiving_events", WebAgentActionType.CLICK_REF, WebObservationRecoveryMode.REUSE_CURRENT),
        ("stale_ref", WebAgentActionType.CLICK_REF, WebObservationRecoveryMode.REFRESH_AND_ATTACH),
        ("missing_ref", WebAgentActionType.CLICK_REF, WebObservationRecoveryMode.REFRESH_AND_ATTACH),
        ("browser_timeout", WebAgentActionType.CLICK_REF, WebObservationRecoveryMode.REFRESH_AND_ATTACH),
    ],
)
def test_observation_recovery_policy_classifies_retryable_errors(
    code: str,
    action: WebAgentActionType,
    expected_mode: WebObservationRecoveryMode,
) -> None:
    feedback_source = (
        "preflight"
        if code
        in {
            "wrong_action_for_element",
            "element_disabled",
            "element_not_receiving_events",
        }
        else ""
    )
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                action=action,
                output={
                    "status": "error",
                    "code": code,
                    "reason_code": code,
                    "retryable": True,
                    "feedback_source": feedback_source,
                },
                is_error=True,
            )
        ]
    )

    assert plan.mode is expected_mode


def test_observation_recovery_policy_prefers_existing_post_action_observation() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                output={
                    "status": "ok",
                    "observation": {"observation_id": "post-action"},
                },
                is_error=True,
            )
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.USE_POST_ACTION
    assert plan.reason is WebObservationRecoveryReason.POST_ACTION_AVAILABLE


def test_observation_recovery_policy_refreshes_mutation_without_observation() -> None:
    plan = WebObservationRecoveryPolicy().plan(actions=[_action(output={"status": "ok", "action": "click_ref"})])

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.reason is WebObservationRecoveryReason.MUTATION_WITHOUT_OBSERVATION


def test_observation_recovery_policy_reuses_state_after_read_only_result() -> None:
    plan = WebObservationRecoveryPolicy().plan(actions=[_action(action=WebAgentActionType.GET_URL)])

    assert plan.mode is WebObservationRecoveryMode.REUSE_CURRENT


def test_observation_recovery_policy_refreshes_failed_post_action_observation() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                output={
                    "status": "ok",
                    "observation_error": {
                        "code": "post_action_observation_failed",
                        "retryable": True,
                    },
                }
            )
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.reason is WebObservationRecoveryReason.POST_ACTION_OBSERVATION_FAILED


def test_observation_error_makes_attached_observation_unusable() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                output={
                    "status": "ok",
                    "observation": {"observation_id": "partial"},
                    "observation_error": {
                        "code": "post_action_observation_failed",
                        "retryable": True,
                    },
                }
            )
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.reason is WebObservationRecoveryReason.POST_ACTION_OBSERVATION_FAILED


class _FakeActionService:
    def __init__(self, observation: WebObservationContext) -> None:
        self.observation = observation
        self.collect_calls = 0
        self.alias_updates = 0

    async def collect_current_observation(self, *_args: object, **_kwargs: object) -> WebObservationContext:
        assert _kwargs.get("publish_ref_aliases") is False
        self.collect_calls += 1
        return self.observation

    def update_ref_aliases_from_observation(self, *_args: object, **_kwargs: object) -> None:
        self.alias_updates += 1

    def project_observation_payload(
        self,
        observation: WebObservationContext,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {
                "observation_id": observation.observation_id,
                "settled": True,
                "settle_reason": "captured",
                "page": {"url": observation.snapshot.url, "title": observation.snapshot.title},
                "snapshot": {"text": '1 button "Continue"', "refs": [], "ref_count": 0},
                "screenshot": {"kind": "viewport", "image_attached": True},
            },
            {"screenshot_url": "data:image/png;base64,QUJD"},
        )

    def attach_observation_payload(
        self,
        payload: dict[str, object],
        *,
        observation: dict[str, object] | None,
        image_urls: dict[str, object],
    ) -> dict[str, object]:
        if observation is not None:
            payload["observation"] = observation
        payload.update(image_urls)
        return payload


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "session"
        self.settings = SimpleNamespace(post_action_observation_mode="som")
        self.cleared_observation_ids: list[str] = []

    def clear_ref_aliases(self, *, observation_id: str = "") -> None:
        self.cleared_observation_ids.append(observation_id)


def test_recovery_service_attaches_to_existing_real_tool_execution() -> None:
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(url="https://example.com/next", title="Next"),
        observation_id="recovery-observation",
    )
    action_service = _FakeActionService(observation)
    service = WebObservationRecoveryService(action_service=action_service)  # type: ignore[arg-type]
    original_output = {
        "status": "error",
        "code": "stale_ref",
        "retryable": True,
        "error": "stale ref",
    }
    trace_state = TraceCaptureState(
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call-real",
                tool_name="web_click_ref",
                tool_output=original_output,
                is_error=True,
            )
        ]
    )
    session = _FakeSession()

    result = asyncio.run(
        service.recover(
            session=session,  # type: ignore[arg-type]
            trace_state=trace_state,
            actions=[
                _action(
                    output=original_output,
                    is_error=True,
                )
            ],
        )
    )

    assert result.recovered is True
    assert action_service.collect_calls == 1
    assert action_service.alias_updates == 1
    assert session.cleared_observation_ids == []
    assert len(trace_state.tool_executions) == 1
    execution = trace_state.tool_executions[0]
    assert execution.tool_call_id == "call-real"
    assert execution.tool_name == "web_click_ref"
    assert execution.tool_output["observation"]["observation_id"] == "recovery-observation"
    assert execution.tool_output["observation_recovery"] == {
        "status": "recovered",
        "source": "runner_refresh",
        "reason": "stale_or_missing_ref",
        "trigger_code": "stale_ref",
        "observation_id": "recovery-observation",
    }
    assert "observation_id" not in execution.llm_tool_output["observation"]
    assert "observation_recovery" not in execution.llm_tool_output
    assert "runner_refresh" not in json.dumps(execution.llm_tool_output)
    assert "latest refs" in execution.llm_tool_output["hint"]


def test_recovered_history_uses_only_the_original_declared_tool_call() -> None:
    builder = WebAgentPromptBuilder()
    trace_state = TraceCaptureState()
    trace_state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-real", name="web_click_ref", arguments='{"ref":"1"}')
    )
    trace_state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-real",
            tool_name="web_click_ref",
            tool_output={
                "status": "error",
                "code": "stale_ref",
                "retryable": True,
                "observation": {
                    "observation_id": "recovery-observation",
                    "settled": True,
                    "settle_reason": "captured",
                    "page": {"url": "https://example.com", "title": "Example"},
                    "snapshot": {"text": '1 button "Continue"', "refs": [], "ref_count": 0},
                    "screenshot": {"kind": "none", "image_attached": False},
                },
                "observation_recovery": {"status": "recovered"},
            },
            is_error=True,
        )
    )
    initial = [Message(role=Role.USER, content=[TextBlock(text="Task:\ncontinue")])]
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=initial),
        prior_messages=initial,
        trace_state=trace_state,
    )

    tool_uses = [block for message in history for block in message.content if isinstance(block, ToolUseBlock)]
    tool_results = [block for message in history for block in message.content if isinstance(block, ToolResultBlock)]

    assert [(block.id, block.name) for block in tool_uses] == [("call-real", "web_click_ref")]
    assert [block.tool_use_id for block in tool_results] == ["call-real"]


def test_remote_recovery_uses_runtime_snapshot_and_attaches_to_real_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched_tools: list[str] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            tool_id = str(kwargs.get("tool_id") or "")
            dispatched_tools.append(tool_id)
            assert tool_id == "web.web_agent_snapshot"
            return CommandResult(
                command_id=str(kwargs.get("command_id") or "snapshot"),
                session_id="remote-session",
                lease_id="lease",
                device_id="device",
                status=CommandState.SUCCEEDED,
                result={
                    "url": "http://127.0.0.1:8765/next",
                    "title": "Next",
                    "snapshot": 'button "Continue" [ref=@e1]',
                    "refs": {
                        "e1": {
                            "role": "button",
                            "name": "Continue",
                            "text": "",
                            "tag": "button",
                        }
                    },
                },
            )

    async def _run() -> tuple[TraceCaptureState, str, dict[str, object]]:
        registry = WebAgentSessionRegistry()
        registry.clear()
        session = RegisteredWebAgentSession(
            session_id="remote-session",
            client=RemoteWebActionBackend(_FakeDispatcher()),  # type: ignore[arg-type]
            artifact_store=object(),  # type: ignore[arg-type]
            settings=WebAgentConfig(
                post_action_observation_mode="none",
                allow_private_network_urls=True,
            ),
        )
        registry.register(session)
        action_service = WebAgentMCPActionService(registry=registry)
        service = WebObservationRecoveryService(action_service=action_service)
        output = {
            "status": "error",
            "code": "stale_ref",
            "retryable": True,
            "error": "stale ref",
        }
        trace_state = TraceCaptureState(
            tool_executions=[
                ToolExecutionRecord(
                    tool_call_id="call-remote",
                    tool_name="web_click_ref",
                    tool_output=output,
                    is_error=True,
                )
            ]
        )
        result = await service.recover(
            session=session,
            trace_state=trace_state,
            actions=[
                _action(
                    output=output,
                    is_error=True,
                    call_id="call-remote",
                )
            ],
        )
        assert result.recovered is True
        resolved_ref = session.resolve_model_ref("1")
        side_channel = _web_ref_alias_side_channel("remote-session")
        registry.clear()
        return trace_state, resolved_ref, side_channel

    trace_state, resolved_ref, side_channel = asyncio.run(_run())

    assert dispatched_tools == ["web.web_agent_snapshot"]
    assert resolved_ref == "@e1"
    assert side_channel["ref_alias_observation_id"].startswith("recovery-remote-session-")
    assert side_channel["ref_alias_epoch"] == 1
    assert side_channel["ref_aliases"] == [
        {
            "ref": "1",
            "source_ref": "@e1",
        }
    ]
    assert len(trace_state.tool_executions) == 1
    execution = trace_state.tool_executions[0]
    assert execution.llm_tool_output["observation"]["page"]["url"] == "http://127.0.0.1:8765/next"
    assert "observation_recovery" not in execution.llm_tool_output
    assert execution.tool_output["observation_recovery"]["source"] == "runner_refresh"

    clicked_refs: list[str] = []

    class _FakeDeviceBackend:
        async def click_ref(self, ref: str) -> WebActionCommandResult:
            clicked_refs.append(ref)
            return WebActionCommandResult(success=True, data={"ref": ref})

    monkeypatch.setattr(
        "northcua.agents.web.session.agentkit_adapter.build_playwright_web_backend",
        lambda *args, **kwargs: _FakeDeviceBackend(),
    )

    async def _execute_device_click() -> CommandResult:
        device_adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"web"}, artifact_root=tmp_path)
        return await device_adapter.execute(
            CommandRequest(
                command_id="cmd-click-recovered-ref",
                idempotency_key="idem-click-recovered-ref",
                session_id="remote-session",
                lease_id="lease",
                device_id="device",
                tool_id="web.web_click_ref",
                parameters={
                    "ref": "1",
                    AGENTKIT_SETTINGS_PARAM: {
                        **side_channel,
                        "post_action_observation_mode": "none",
                        "persist_post_action_observation": False,
                    },
                },
            )
        )

    device_result = asyncio.run(_execute_device_click())
    WebAgentSessionRegistry().clear()

    assert device_result.status is CommandState.SUCCEEDED
    assert clicked_refs == ["@e1"]


def test_observation_recovery_policy_refreshes_after_later_mutation() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                action=WebAgentActionType.SNAPSHOT,
                output={"status": "ok", "observation": {"observation_id": "intermediate"}},
                call_id="call-snapshot",
            ),
            _action(
                action=WebAgentActionType.CLICK_REF,
                output={"status": "ok"},
                call_id="call-click",
            ),
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.target_tool_call_id == "call-click"
    assert plan.reason is WebObservationRecoveryReason.MUTATION_WITHOUT_OBSERVATION


def test_recovery_attaches_to_trigger_before_trailing_read_only_action() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                action=WebAgentActionType.CLICK_REF,
                output={
                    "status": "error",
                    "reason_code": "browser_timeout",
                    "retryable": True,
                },
                is_error=True,
                call_id="call-timeout",
            ),
            _action(
                action=WebAgentActionType.NOTE,
                output={"status": "ok"},
                call_id="call-note",
            ),
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.target_tool_call_id == "call-timeout"


def test_recovery_attaches_to_latest_mutation_before_trailing_read_only_action() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                action=WebAgentActionType.CLICK_REF,
                output={"status": "ok"},
                call_id="call-click",
            ),
            _action(
                action=WebAgentActionType.NOTE,
                output={"status": "ok"},
                call_id="call-note",
            ),
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.target_tool_call_id == "call-click"


@pytest.mark.parametrize(
    ("actions", "expected_call_id", "expected_reason"),
    [
        (
            [
                _action(
                    action=WebAgentActionType.CLICK_REF,
                    output={"status": "error", "reason_code": "browser_timeout", "retryable": True},
                    is_error=True,
                    call_id="call-error",
                ),
                _action(
                    action=WebAgentActionType.FILL_REF,
                    output={"status": "ok"},
                    call_id="call-mutation",
                ),
            ],
            "call-mutation",
            WebObservationRecoveryReason.MUTATION_WITHOUT_OBSERVATION,
        ),
        (
            [
                _action(
                    action=WebAgentActionType.CLICK_REF,
                    output={
                        "status": "ok",
                        "observation_error": {"code": "post_action_observation_failed"},
                    },
                    call_id="call-observation-failure",
                ),
                _action(
                    action=WebAgentActionType.FILL_REF,
                    output={"status": "ok"},
                    call_id="call-mutation",
                ),
            ],
            "call-mutation",
            WebObservationRecoveryReason.MUTATION_WITHOUT_OBSERVATION,
        ),
        (
            [
                _action(
                    action=WebAgentActionType.FILL_REF,
                    output={"status": "ok"},
                    call_id="call-mutation",
                ),
                _action(
                    action=WebAgentActionType.CLICK_REF,
                    output={"status": "error", "reason_code": "browser_timeout", "retryable": True},
                    is_error=True,
                    call_id="call-error",
                ),
            ],
            "call-error",
            WebObservationRecoveryReason.ACTION_OUTCOME_UNCERTAIN,
        ),
    ],
)
def test_recovery_uses_chronologically_latest_refresh_trigger(
    actions: list[ResolvedWebAgentAction],
    expected_call_id: str,
    expected_reason: WebObservationRecoveryReason,
) -> None:
    plan = WebObservationRecoveryPolicy().plan(actions=actions)

    assert plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert plan.target_tool_call_id == expected_call_id
    assert plan.reason is expected_reason


def test_observation_recovery_policy_accepts_observation_after_last_mutation() -> None:
    plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                action=WebAgentActionType.CLICK_REF,
                output={"status": "ok"},
                call_id="call-click",
            ),
            _action(
                action=WebAgentActionType.SNAPSHOT,
                output={"status": "ok", "observation": {"observation_id": "final"}},
                call_id="call-snapshot",
            ),
        ]
    )

    assert plan.mode is WebObservationRecoveryMode.USE_POST_ACTION


def test_playwright_actionability_error_refreshes_but_preflight_reuses() -> None:
    playwright_plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                output={
                    "status": "error",
                    "reason_code": "element_disabled",
                    "retryable": True,
                    "feedback_source": "playwright",
                },
                is_error=True,
            )
        ]
    )
    preflight_plan = WebObservationRecoveryPolicy().plan(
        actions=[
            _action(
                output={
                    "status": "error",
                    "reason_code": "element_disabled",
                    "retryable": True,
                    "feedback_source": "preflight",
                },
                is_error=True,
            )
        ]
    )

    assert playwright_plan.mode is WebObservationRecoveryMode.REFRESH_AND_ATTACH
    assert preflight_plan.mode is WebObservationRecoveryMode.REUSE_CURRENT


def test_recovery_failure_clears_aliases_and_updates_real_tool_result() -> None:
    class _FailingActionService(_FakeActionService):
        async def collect_current_observation(self, *_args: object, **_kwargs: object) -> WebObservationContext:
            raise RuntimeError("snapshot unavailable")

    observation = WebObservationContext(snapshot=WebSnapshotObservation(), observation_id="unused")
    action_service = _FailingActionService(observation)
    service = WebObservationRecoveryService(action_service=action_service)  # type: ignore[arg-type]
    output = {"status": "error", "code": "stale_ref", "retryable": True}
    trace_state = TraceCaptureState(
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call-real",
                tool_name="web_click_ref",
                tool_output=output,
                is_error=True,
            )
        ]
    )
    session = _FakeSession()

    result = asyncio.run(
        service.recover(
            session=session,  # type: ignore[arg-type]
            trace_state=trace_state,
            actions=[_action(output=output, is_error=True)],
        )
    )

    assert result.recovered is False
    assert "snapshot unavailable" in result.error
    assert len(session.cleared_observation_ids) == 1
    recovery = trace_state.tool_executions[0].tool_output["observation_recovery"]
    assert recovery["status"] == "failed"
    assert recovery["retryable"] is True
    assert "web_snapshot" in recovery["hint"]
    llm_output = trace_state.tool_executions[0].llm_tool_output
    assert "observation_recovery" not in llm_output
    assert llm_output["observation_error"] == {
        "code": "current_observation_unavailable",
        "retryable": True,
        "refs_invalidated": True,
        "hint": "Current browser state could not be collected. Call web_snapshot before using any ref.",
    }


def test_alias_publication_failure_discards_unusable_recovery_observation() -> None:
    class _AliasFailingActionService(_FakeActionService):
        def update_ref_aliases_from_observation(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("alias publication failed")

    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(url="https://example.com"),
        observation_id="recovery-observation",
    )
    action_service = _AliasFailingActionService(observation)
    service = WebObservationRecoveryService(action_service=action_service)  # type: ignore[arg-type]
    output = {
        "status": "error",
        "code": "stale_ref",
        "retryable": True,
        "result": {
            "observation": {
                "observation_id": "stale-observation",
                "page": {"url": "https://example.com", "title": "Example"},
                "snapshot": {
                    "text": '7 button "Old target"',
                    "refs": [{"ref": "7", "role": "button", "label": "Old target"}],
                },
            },
            "raw_image_url": "data:image/png;base64,U1RBTEU=",
            "som_image_url": "data:image/png;base64,U1RBTEVTT00=",
            "action_overlay_image_url": "data:image/png;base64,T1ZFUkxBWQ==",
            "action_result": "preserved",
        },
        "observation_error": {
            "code": "post_action_observation_failed",
            "recovered": False,
        },
    }
    trace_state = TraceCaptureState(
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call-real",
                tool_name="web_click_ref",
                tool_output=output,
                is_error=True,
            )
        ]
    )
    session = _FakeSession()

    result = asyncio.run(
        service.recover(
            session=session,  # type: ignore[arg-type]
            trace_state=trace_state,
            actions=[_action(output=output, is_error=True)],
        )
    )

    assert result.recovered is False
    assert "alias publication failed" in result.error
    assert len(session.cleared_observation_ids) == 1
    execution = trace_state.tool_executions[0]
    assert "observation" not in execution.tool_output["result"]
    assert execution.tool_output["result"]["action_overlay_image_url"].endswith("T1ZFUkxBWQ==")
    llm_output = execution.llm_tool_output
    assert "observation_recovery" not in llm_output
    assert "observation" not in llm_output
    assert "screenshot_url" not in llm_output
    assert "observation" not in llm_output["result"]
    assert "raw_image_url" not in llm_output["result"]
    assert "som_image_url" not in llm_output["result"]
    assert llm_output["result"]["action_overlay_image_url"].endswith("T1ZFUkxBWQ==")
    assert llm_output["result"]["action_result"] == "preserved"
    assert llm_output["observation_error"]["code"] == "current_observation_unavailable"
    assert llm_output["observation_error"]["refs_invalidated"] is True
    rendered = WebHistoryToolResultProjector().render_web_tool_output(llm_output)
    assert "Old target" not in rendered
    assert "Post-action observation recovered after" not in rendered
    assert REFS_INVALIDATED_NOTICE in rendered


def test_failed_recovery_expires_current_observation_refs_and_image() -> None:
    current = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='1 button "Continue"',
            elements=[WebElementRef(ref_id="1", role="button", name="Continue")],
        ),
        screenshot=WebScreenshotObservation(data_url="data:image/png;base64,QUJD"),
        observation_id="old-observation",
    )

    invalidated = WebObservationRecoveryService.without_actionable_refs(current)

    assert invalidated.snapshot.url == current.snapshot.url
    assert invalidated.snapshot.title == current.snapshot.title
    assert invalidated.snapshot.snapshot == ""
    assert invalidated.snapshot.elements == []
    assert invalidated.snapshot.raw == {}
    assert invalidated.screenshot is None
    assert invalidated.settled is False
    assert invalidated.settle_reason == "observation_recovery_failed"


def test_failed_recovery_is_ref_window_barrier_for_older_observations() -> None:
    old_observation = {
        "observation": {
            "observation_id": "old-observation",
            "snapshot": {
                "text": '1 button "Old target"',
                "refs": [{"ref": "1", "role": "button", "label": "Old target"}],
                "ref_count": 1,
            },
            "screenshot": {"kind": "set_of_mark", "annotation": {"marks": [{"ref": "1"}]}},
        }
    }
    failure = {
        "status": "error",
        "code": "stale_ref",
        "observation_error": {
            "code": "current_observation_unavailable",
            "refs_invalidated": True,
            "hint": "Call web_snapshot before using any ref.",
        },
    }
    failure_text = WebHistoryToolResultProjector().render_web_tool_output(failure)
    messages = [
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call-old",
                    content=json.dumps(old_observation),
                    is_error=False,
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call-failure",
                    content=failure_text,
                    is_error=True,
                )
            ],
        ),
    ]

    budgeted = WebRefSnapshotWindowBudgeter(max_history_snapshots=1).apply(messages)
    old_result = budgeted[0].content[0]
    assert isinstance(old_result, ToolResultBlock)
    old_text = str(old_result.content)

    assert '"refs": []' in old_text
    assert "Old target" not in old_text
    assert REFS_INVALIDATED_NOTICE in failure_text
    assert WebRefSnapshotWindowBudgeter(max_history_snapshots=1).latest_retained_tool_result_ids(budgeted) == set()


def test_recovery_fails_closed_when_declared_call_id_is_missing() -> None:
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(url="https://example.com"),
        observation_id="recovery-observation",
    )
    action_service = _FakeActionService(observation)
    service = WebObservationRecoveryService(action_service=action_service)  # type: ignore[arg-type]
    trace_state = TraceCaptureState(
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="different-call",
                tool_name="web_note",
                tool_output={"status": "ok"},
            )
        ]
    )
    session = _FakeSession()

    with pytest.raises(WebObservationRecoveryInvariantError, match="was not found"):
        asyncio.run(
            service.recover(
                session=session,  # type: ignore[arg-type]
                trace_state=trace_state,
                actions=[
                    _action(
                        output={"status": "error", "code": "stale_ref", "retryable": True},
                        is_error=True,
                    )
                ],
            )
        )

    assert trace_state.tool_executions[0].tool_output == {"status": "ok"}
    assert action_service.collect_calls == 0


def test_recovery_fails_closed_when_declared_call_id_is_empty() -> None:
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(url="https://example.com"),
        observation_id="recovery-observation",
    )
    action_service = _FakeActionService(observation)
    service = WebObservationRecoveryService(action_service=action_service)  # type: ignore[arg-type]
    trace_state = TraceCaptureState(
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="last-call",
                tool_name="web_note",
                tool_output={"status": "ok"},
            )
        ]
    )

    with pytest.raises(WebObservationRecoveryInvariantError, match="was not found"):
        asyncio.run(
            service.recover(
                session=_FakeSession(),  # type: ignore[arg-type]
                trace_state=trace_state,
                actions=[
                    _action(
                        output={"status": "error", "code": "stale_ref", "retryable": True},
                        is_error=True,
                        call_id="",
                    )
                ],
            )
        )

    assert trace_state.tool_executions[0].tool_output == {"status": "ok"}
    assert action_service.collect_calls == 0
