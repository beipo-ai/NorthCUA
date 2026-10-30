"""Tests for the standalone device worker data plane."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from northcua_device.config import DeviceClientConfig
from northcua_device.executor import CommandExecutor
from northcua_device.runtime.browser import BrowserRuntimeAdapter
from northcua_device.runtime.desktop import DesktopRuntimeAdapter
from northcua_device.runtime.shell import EchoRuntimeAdapter, ShellRuntimeAdapter
from northcua_device.session_actor import SessionActor
from northcua_device.worker import DeviceWorker
from northcua_protocol.capabilities import Capability, PlatformKind, ToolDescriptor, ToolRiskLevel
from northcua_protocol.payloads import CommandRequest, SessionLeaseGranted
from northcua_protocol.state import CommandState, DeviceState
from northcua_runtime_shared.desktop_surface import desktop_tool_contracts
from northcua_runtime_shared.desktop_surface.profiles import DesktopToolProfile


def _executor(allowed_tools: set[str] | None = None) -> CommandExecutor:
    return CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.MACOS,
        adapters=[EchoRuntimeAdapter()],
        allowed_tools=allowed_tools or {"device.echo"},
    )


def test_desktop_adapter_uses_registered_tool_ids_and_extracts_artifacts() -> None:
    class _FakeServer:
        async def list_tools(self) -> list[object]:
            return [
                type(
                    "_Tool",
                    (),
                    {
                        "name": "capture_desktop_screenshot",
                        "parameters": {"type": "object", "properties": {"all_screens": {"type": "boolean"}}},
                    },
                )()
            ]

        async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
            assert tool_name == "capture_desktop_screenshot"
            assert parameters == {"all_screens": False}
            return {"screenshot_url": "data:image/png;base64,ZmFrZS1wbmc="}

    async def _run() -> None:
        contract = next(
            contract for contract in desktop_tool_contracts() if contract.tool_id == "capture_desktop_screenshot"
        )
        adapter = DesktopRuntimeAdapter(
            servers_by_name={"DesktopNativeDataExecutor": _FakeServer()},
            tool_contracts=[contract],
            normalize_result=lambda result: result if isinstance(result, dict) else {"result": str(result)},
        )
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-shot",
                idempotency_key="idem",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="capture_desktop_screenshot",
                parameters={"all_screens": False},
            )
        )

        assert result.status == CommandState.SUCCEEDED
        assert result.result == {
            "screenshot_url": "",
            "screenshot_url_artifact_id": "cmd-shot-screenshot_url",
            "screenshot_url_artifact_sha256": result.artifacts[0].sha256,
            "screenshot_url_artifact_mime_type": "image/png",
        }
        assert result.artifacts[0].mime_type == "image/png"
        assert result.artifacts[0].inline_data is not None

    asyncio.run(_run())


def test_desktop_adapter_inlines_worker_local_jpeg_with_matching_mime_type(tmp_path: Path) -> None:
    screenshot = tmp_path / "shot.jpg"
    screenshot.write_bytes(b"fake-jpeg")

    result = DesktopRuntimeAdapter._with_inline_screenshot_urls({"screenshot_path": str(screenshot)})

    assert str(result["screenshot_url"]).startswith("data:image/jpeg;base64,")
    assert result["screenshot_format"] == "local_path"


def test_browser_adapter_advertises_agentkit_tools_and_rejects_unknown_tool() -> None:
    async def _run() -> None:
        adapter = BrowserRuntimeAdapter(agent_kinds={"web"})
        health = await adapter.healthcheck()
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-browser",
                idempotency_key="idem",
                session_id="sess",
                lease_id="lease",
                device_id="browser-1",
                tool_id="browser.health",
                lease_generation=3,
            )
        )

        assert adapter.capabilities == {Capability.BROWSER_PAGE}
        assert health.healthy is True
        assert {tool.tool_id for tool in adapter.tools} >= {
            "web_snapshot",
            "web.web_scroll",
        }
        assert result.status == CommandState.FAILED
        assert result.lease_generation == 3
        assert "unknown AgentKit tool_id" in (result.error or "")

    asyncio.run(_run())


def test_desktop_adapter_tool_ids_match_registered_desktop_mcp_surface() -> None:
    adapter = DesktopRuntimeAdapter()

    from northcua_runtime_shared.desktop_surface.profiles import default_desktop_tool_profile

    shared = {tool.tool_id: tool.parameters_schema for tool in default_desktop_tool_profile().tools}

    assert {tool.tool_id for tool in adapter.tools} == set(shared)
    assert "cua_replay_trajectory" not in shared
    assert "cua_move_cursor" not in shared
    assert "cua_set_config" not in shared
    assert "cua_page" not in shared
    scroll_schema = shared["scroll"]
    assert "positive scrolls up" in scroll_schema["properties"]["delta_y"]["description"]


def test_shell_readonly_tool_allows_simple_allowlisted_probe() -> None:
    async def _run() -> None:
        adapter = ShellRuntimeAdapter()
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-readonly",
                idempotency_key="idem",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="shell.exec_readonly",
                parameters={"cmd": "pwd"},
            )
        )

        assert result.status == CommandState.SUCCEEDED
        assert isinstance(result.result, dict)
        assert result.result["returncode"] == 0

    asyncio.run(_run())


@pytest.mark.parametrize(
    "cmd",
    [
        "touch /tmp/x",
        "rm -rf /tmp/x",
        "echo ok > /tmp/x",
        "curl http://example.test",
        "env sh -c 'touch /tmp/northcua_ro_bypass'",
        "date -s '2020-01-01'",
        'python3 -c \'import pathlib; p=pathlib.Path("/tmp/northcua_pwn"); getattr(p,"write_text")("x")\'',
    ],
)
def test_shell_readonly_tool_rejects_mutating_or_network_commands(cmd: str) -> None:
    async def _run() -> None:
        adapter = ShellRuntimeAdapter()
        with pytest.raises(ValueError):
            await adapter.execute(
                CommandRequest(
                    command_id="cmd-readonly",
                    idempotency_key="idem",
                    session_id="sess",
                    lease_id="lease",
                    device_id="dev-1",
                    tool_id="shell.exec_readonly",
                    parameters={"cmd": cmd},
                )
            )

    asyncio.run(_run())


def test_device_health_report_surfaces_unhealthy_runtime() -> None:
    class _FailingAdapter(EchoRuntimeAdapter):
        async def healthcheck(self) -> object:
            from northcua_runtime_shared import RuntimeHealth

            return RuntimeHealth(
                healthy=False,
                message="desktop bridge missing",
                capability_health={Capability.DESKTOP_INPUT: False},
            )

    async def _run() -> None:
        worker = DeviceWorker(
            DeviceClientConfig(server_url="ws://controller", device_id="dev-health"),
            adapters=[_FailingAdapter()],
        )

        report = await worker.health_report()

        assert report.healthy is False
        assert report.capability_health == {Capability.DESKTOP_INPUT.value: False}
        assert report.messages == ["desktop bridge missing"]

    asyncio.run(_run())
