"""Local benchmark callbacks must execute on the harness and rejoin history."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from northcua.agents.desktop.round_execution.tool_dispatch import (
    LocalExternalToolDispatcher,
    RemoteExternalCommandExecutor,
)
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.protocol import CommandResultPayload
from northcua.integrations.osworld2.tools import ask_user_tool
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
@pytest.mark.parametrize("remote", [False, True])
async def test_callback_and_device_actions_keep_order_and_custody(tamper, remote):
    observed = []
    tool = ask_user_tool(lambda question: observed.append(question) or "original answer")
    if tamper:
        tool.description = "stale projection"

    class Device:
        async def execute_command_request(self, payload):
            observed.extend(row.tool_name for row in payload.commands)
            return CommandResultPayload(
                session_id=payload.session_id,
                status=ResultStatus.SUCCESS,
                results=[
                    CommandExecutionResultRecord(call_id=row.call_id, status=ResultStatus.SUCCESS, result="device")
                    for row in payload.commands
                ],
            )

    async def dispatch_remote(calls):
        observed.extend(call["name"] for call in calls)
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(call_id=call["id"], status=ResultStatus.SUCCESS, result="device")
                for call in calls
            ]
        )

    executor = RemoteExternalCommandExecutor(dispatch_remote) if remote else Device()
    rows, traces = await LocalExternalToolDispatcher(device_executor=executor).execute(
        session_id="fixture",
        round_num=1,
        round_step=1,
        pending_tool_calls=[
            {"id": "before", "name": "click", "server_name": "DesktopNativeActionExecutor", "input": {"x": 1, "y": 1}},
            {"id": "question", "name": "ask_user", "input": {"question": "Which file?"}},
            {
                "id": "after",
                "name": "keypress",
                "server_name": "DesktopNativeActionExecutor",
                "input": {"key": "enter"},
            },
        ],
        agent=NS(config=NS(tools=[tool])),
        agent_state=None,
    )
    assert observed == ["click", "ask_user" if tamper else "Which file?", "keypress"]
    assert [row.call_id for row in rows] == ["before", "question", "after"]
    assert rows[1].result == ("device" if tamper else "original answer")
    assert traces[1].llm_tool_output == rows[1].result


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_callback_failure_and_cancellation_do_not_go_to_device(cancel):
    def answer(question):
        raise asyncio.CancelledError() if cancel else OSError("simulator offline")

    class Device:
        async def execute_command_request(self, payload):
            pytest.fail("A harness callback was sent to the device")

    pending = [{"id": "question", "name": "ask_user", "input": {"question": "Which file?"}}]
    call = LocalExternalToolDispatcher(device_executor=Device()).execute(
        session_id="fixture",
        round_num=1,
        round_step=1,
        pending_tool_calls=pending,
        agent=NS(config=NS(tools=[ask_user_tool(answer)])),
        agent_state=None,
    )
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        rows, traces = await call
        assert rows[0].status == ResultStatus.FAILURE
        assert "simulator offline" in rows[0].error
        assert traces[0].is_error


@pytest.mark.asyncio
async def test_real_desktop_query_receives_callback_answer_in_next_model_request(
    tmp_path, monkeypatch, desktop_native_test_session
):
    import yaml

    from northcua.agents.desktop.application import DesktopApplication
    from northcua.core.config_loader import NorthCUAConfigLoader
    from northcua.integrations.osworld2 import run_task
    from northcua.runtime.mcp.desktop_native import state
    from northcua.runtime.mcp.desktop_native.backend import PyAutoGuiDesktopBackend
    from tests.integration.test_web_modes import ModelFixture
    from tests.runtime.mcp.test_desktop_native_server import _FakeDesktopBackend

    backend = _FakeDesktopBackend()
    monkeypatch.setattr(state.DesktopNativeServerState, "get_backend", lambda self, backend_factory=None: backend)

    def forbid_physical_desktop(*args, **kwargs):
        pytest.fail("The isolated harness test attempted to access the physical desktop")

    monkeypatch.setattr(PyAutoGuiDesktopBackend, "capture_desktop_screenshot", forbid_physical_desktop)
    assert state.DesktopNativeServerState().get_backend() is backend
    questions = []
    with ModelFixture(desktop_text="fixture typed output") as fixture:
        original_tool_call = fixture.tool_call

        def tool_call(call_id, name, arguments):
            # This task tests callback/history wiring, with no physical input.
            if call_id in {"type", "enter"}:
                name, arguments = "note", {"text": "Record fixture progress", "key": "progress"}
            message = original_tool_call(call_id, name, arguments)
            if call_id == "memory":
                message["tool_calls"].extend(
                    original_tool_call("question", "ask_user", {"question": "Which file?"})["tool_calls"]
                )
            return message

        fixture.tool_call = tool_call
        config_path = fixture.write_config(tmp_path / "config")
        data = yaml.safe_load(config_path.read_text())
        data["evaluation"]["enabled"] = False
        data["observation"] = {"a11y": {"enabled": False}}
        data["prompt"] = {"screenshot_window_images": 1, "system_prompt_profile": "osworld2_formal"}
        data["execution"] = {"no_tool_call_attempts": 3}
        data["desktop"] = {"coordinate_policy": {"input_space": "normalized_1000"}}
        config_path.write_text(yaml.safe_dump(data))
        agent_path = config_path.parent / "agent.yaml"
        data = yaml.safe_load(agent_path.read_text())
        data["mcp_servers"] = [
            {"name": "DesktopNativeDataExecutor", "type": "in-memory", "model_visible": False},
            {"name": "DesktopNativeActionExecutor", "type": "in-memory", "model_visible": True},
        ]
        agent_path.write_text(yaml.safe_dump(data))
        bootstrap = NorthCUAConfigLoader.from_app_config(config_path)
        bootstrap.agent.tools.append(ask_user_tool(lambda q: questions.append(q) or "receipt from original simulator"))
        app = DesktopApplication(bootstrap.app, bootstrap.agent, session_id="prepared-question")

        async def evaluate():
            assert questions == ["Which file?"]
            assert backend.screenshot_calls > 0
            return 1.0

        try:
            report = await run_task(
                app=app,
                session_id="prepared-question",
                task_id="fixture-task",
                query="Complete the fixture.",
                evaluate=evaluate,
                max_steps=8,
            )
        finally:
            await app.stop()
        assert report["score_valid"] and report["score"] == 1, report
        tool_messages = [m for m in fixture.requests[1]["messages"] if m.get("role") == "tool"]
        answers = [m for m in tool_messages if m.get("tool_call_id") == "question"]
        assert len(answers) == 1 and "receipt from original simulator" in str(answers[0]["content"])
        assert fixture.actor_calls == 6  # three tool steps, then three identical terminal requests
