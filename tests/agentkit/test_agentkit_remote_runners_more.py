from __future__ import annotations

import asyncio

from northcua.agentkit.external_tools import tool_execution_records_from_command_results
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua_controller_plugins.agentkit_web_remote_dispatch import RemoteWebActionBackend
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.payloads import CommandResult
from northcua_protocol.state import CommandState


def test_remote_web_playwright_backend_preserves_unwrapped_screenshot_artifact() -> None:
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="cmd-shot-screenshot_url",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-png",
        mime_type="image/png",
    )

    class _FakeDispatcher:
        async def dispatch(
            self,
            *,
            tool_id: str,
            parameters: dict[str, object],
            timeout_seconds: float,
        ) -> CommandResult:
            assert tool_id == "web.web_agent_backend_screenshot"
            assert parameters["annotate"] is False
            assert "full_page" not in parameters
            assert timeout_seconds == 30.0
            return CommandResult(
                command_id="cmd-shot",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={
                    "status": "ok",
                    "text": "Captured screenshot.",
                    "result": {"width": 10, "height": 20, "kind": "viewport", "image_attached": True},
                    "screenshot_url": "",
                    "screenshot_url_artifact_id": artifact.artifact_id,
                },
                artifacts=[artifact],
            )

    backend = RemoteWebActionBackend(_FakeDispatcher())  # type: ignore[arg-type]

    screenshot = asyncio.run(backend.screenshot())

    assert screenshot.width == 10
    assert screenshot.height == 20
    assert screenshot.data_url.startswith("data:image/png;base64,")


def test_external_tool_trace_keeps_provider_tool_call_id() -> None:
    records = tool_execution_records_from_command_results(
        pending_tool_calls=[
            {
                "id": "call-provider-1",
                "name": "web_get_url",
                "input": {},
            }
        ],
        command_results=[
            CommandExecutionResultRecord(
                call_id="web-agent:action:1:1:call-provider-1",
                status=ResultStatus.SUCCESS,
                result={"status": "ok"},
            )
        ],
    )

    assert records[0].tool_call_id == "call-provider-1"
    assert records[0].llm_tool_output == {"status": "ok"}
