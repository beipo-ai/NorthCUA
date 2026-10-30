from __future__ import annotations

import asyncio
from types import SimpleNamespace

from northcua.agents.desktop.server_remote import DesktopRemoteServerBridge
from northcua.core.config import NorthCUAAppConfig
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.protocol import NorthCUAWebSocketEnvelope


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)


class _FakeFacade:
    def __init__(self, app_config: NorthCUAAppConfig) -> None:
        self.app_config = app_config
        self.rounds = SimpleNamespace(
            state=SimpleNamespace(
                session_id="sess-1",
                round_num=1,
                round_step=2,
            )
        )

    async def consume_command_results(self, call_ids: list[str]) -> list[CommandExecutionResultRecord]:
        return [
            CommandExecutionResultRecord(
                call_id=call_ids[0],
                status=ResultStatus.SUCCESS,
                result={"text": "ok"},
            )
        ]


def test_remote_tool_dispatch_omits_default_desktop_coordinate_metadata() -> None:
    websocket = _FakeWebSocket()
    bridge = DesktopRemoteServerBridge(command_result_timeout_seconds=1.0, command_poll_interval_seconds=0.01)

    asyncio.run(
        bridge.dispatch_external_tool_calls(
            websocket=websocket,  # type: ignore[arg-type]
            facade=_FakeFacade(NorthCUAAppConfig(agent_config_path="agent.yaml")),  # type: ignore[arg-type]
            pending_tool_calls=[{"id": "call-1", "name": "click", "input": {"x": 10, "y": 20}}],
        )
    )

    envelope = NorthCUAWebSocketEnvelope.model_validate_json(websocket.sent[0])

    assert envelope.payload.metadata == {}


def test_remote_tool_dispatch_sends_non_default_desktop_coordinate_metadata() -> None:
    websocket = _FakeWebSocket()
    bridge = DesktopRemoteServerBridge(command_result_timeout_seconds=1.0, command_poll_interval_seconds=0.01)
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        desktop={"coordinate_policy": {"input_space": "normalized_1000", "clamp_to_bounds": True}},
    )

    asyncio.run(
        bridge.dispatch_external_tool_calls(
            websocket=websocket,  # type: ignore[arg-type]
            facade=_FakeFacade(app_config),  # type: ignore[arg-type]
            pending_tool_calls=[{"id": "call-1", "name": "click", "input": {"x": 500, "y": 500}}],
        )
    )

    envelope = NorthCUAWebSocketEnvelope.model_validate_json(websocket.sent[0])

    assert envelope.payload.metadata == {
        "northcua.desktop.coordinate_space": "normalized_1000",
        "northcua.desktop.coordinate_clamp_to_bounds": "true",
    }
