"""Controller/device integration tests that exercise real websocket workers."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import suppress

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.execution.command_bus import command_idempotency_key
from northcua_device.config import DeviceClientConfig
from northcua_device.runtime.desktop import DesktopRuntimeAdapter
from northcua_device.worker import DeviceWorker
from northcua_protocol.capabilities import Capability, PlatformKind, ToolDescriptor
from northcua_protocol.payloads import CommandRequest, TaskRequest
from northcua_protocol.state import CommandState
from northcua_runtime_shared import (
    clear_desktop_mcp_bridge,
    desktop_tool_contracts,
    desktop_tool_server_name,
    desktop_wire_tool_descriptors,
    register_desktop_mcp_bridge,
)


class _FakeDesktopBridge:
    """Minimal bridge that makes fake MCP results self-contained in this test."""

    def ensure_registered(self) -> None:
        """No-op because servers are injected directly into the adapter."""

    def build_server(self, *, server_name: str, process_name: str) -> object:
        """This test injects servers directly and should not ask the bridge."""

        raise AssertionError("fake bridge should not build servers in this test")

    def is_registered(self, server_name: str) -> bool:
        """Return true because descriptor registration is tested separately."""

        return True

    def normalize_tool_result(self, tool_result: object) -> dict[str, object] | str:
        """Return fake server dictionaries as normalized MCP payloads."""

        if isinstance(tool_result, dict):
            return dict(tool_result)
        return str(tool_result)


class _FakeMcpServer:
    """FastMCP-like server that records calls and returns a tiny screenshot."""

    def __init__(self, *, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
        """Pretend to execute one registered MCP tool."""

        self.calls.append((tool_name, parameters))
        encoded = base64.b64encode(f"fake-image:{self.label}".encode("utf-8")).decode("ascii")
        return {
            "tool_name": tool_name,
            "sandbox": self.label,
            "image": f"data:image/png;base64,{encoded}",
        }


def _screenshot_descriptor() -> ToolDescriptor:
    for descriptor in desktop_wire_tool_descriptors():
        if descriptor.tool_id == "capture_desktop_screenshot":
            return descriptor
    raise AssertionError("desktop MCP surface is missing capture_desktop_screenshot")


def _screenshot_contract():
    for contract in desktop_tool_contracts():
        if contract.tool_id == "capture_desktop_screenshot":
            return contract
    raise AssertionError("desktop contract surface is missing capture_desktop_screenshot")


async def _wait_for_session(server: ControllerServer, task_id: str) -> tuple[str, str, int]:
    for _ in range(100):
        row = server.store.session_row_for_task(task_id)
        if row is not None and row["lease_id"]:
            lease_row = server.store.lease_row(str(row["lease_id"]))
            if lease_row is not None:
                return str(row["session_id"]), str(row["lease_id"]), int(lease_row["generation"])
        await asyncio.sleep(0.01)
    raise AssertionError(f"session was not leased for task {task_id}")


async def _wait_for_worker_lease(worker: DeviceWorker, session_id: str) -> None:
    for _ in range(100):
        if worker.actor.session(session_id) is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"worker did not accept session lease {session_id}")


def test_two_device_workers_self_submit_and_execute_registered_mcp_commands(tmp_path) -> None:
    """Two sandbox workers stay isolated while executing real MCP tool ids."""

    async def _run() -> None:
        register_desktop_mcp_bridge(_FakeDesktopBridge())
        descriptor = _screenshot_descriptor()
        contract = _screenshot_contract()
        server_name = desktop_tool_server_name(descriptor.tool_id)
        assert server_name
        controller = ControllerServer(
            ControllerConfig(
                host="127.0.0.1",
                port=0,
                db_path=str(tmp_path / "controller.sqlite"),
                blob_dir=str(tmp_path / "blobs"),
                security=ControllerSecurityConfig(
                    insecure_dev_no_auth=True,
                    max_inline_artifact_bytes=1,
                ),
            )
        )
        await controller.start()
        assert controller._server is not None  # noqa: SLF001 - test needs the ephemeral port.
        port = controller._server.sockets[0].getsockname()[1]  # noqa: SLF001
        controller_url = f"ws://127.0.0.1:{port}"

        fake_a = _FakeMcpServer(label="sandbox-a")
        fake_b = _FakeMcpServer(label="sandbox-b")
        worker_a = DeviceWorker(
            DeviceClientConfig(
                server_url=controller_url,
                device_id="sandbox-a",
                platform=PlatformKind.MACOS,
                gui_resource_id="sandbox-a",
                max_sessions=1,
                max_concurrent_commands=1,
                allowed_tools={descriptor.tool_id},
                insecure_dev_no_auth=True,
                heartbeat_interval_seconds=0.2,
                log_dir=str(tmp_path / "device-a"),
            ),
            adapters=[
                DesktopRuntimeAdapter(
                    servers_by_name={server_name: fake_a},
                    tool_contracts=[contract],
                )
            ],
            initial_task=TaskRequest(
                task_id="task-a",
                trace_id="trace-a",
                agent_kind="desktop",
                request="sandbox A task",
                required_capabilities={Capability.DESKTOP_SCREENSHOT},
            ),
        )
        worker_b = DeviceWorker(
            DeviceClientConfig(
                server_url=controller_url,
                device_id="sandbox-b",
                platform=PlatformKind.MACOS,
                gui_resource_id="sandbox-b",
                max_sessions=1,
                max_concurrent_commands=1,
                allowed_tools={descriptor.tool_id},
                insecure_dev_no_auth=True,
                heartbeat_interval_seconds=0.2,
                log_dir=str(tmp_path / "device-b"),
            ),
            adapters=[
                DesktopRuntimeAdapter(
                    servers_by_name={server_name: fake_b},
                    tool_contracts=[contract],
                )
            ],
            initial_task=TaskRequest(
                task_id="task-b",
                trace_id="trace-b",
                agent_kind="desktop",
                request="sandbox B task",
                required_capabilities={Capability.DESKTOP_SCREENSHOT},
            ),
        )

        worker_tasks = [asyncio.create_task(worker_a.run_once()), asyncio.create_task(worker_b.run_once())]
        try:
            session_a, lease_a, generation_a = await _wait_for_session(controller, "task-a")
            session_b, lease_b, generation_b = await _wait_for_session(controller, "task-b")
            await _wait_for_worker_lease(worker_a, session_a)
            await _wait_for_worker_lease(worker_b, session_b)

            command_a = CommandRequest(
                command_id="cmd-a",
                idempotency_key=command_idempotency_key(
                    session_id=session_a,
                    lease_id=lease_a,
                    command_id="cmd-a",
                    payload={"tool_id": descriptor.tool_id, "parameters": {"all_screens": False}},
                ),
                session_id=session_a,
                lease_id=lease_a,
                device_id="sandbox-a",
                tool_id=descriptor.tool_id,
                parameters={"all_screens": False},
                lease_generation=generation_a,
            )
            command_b = CommandRequest(
                command_id="cmd-b",
                idempotency_key=command_idempotency_key(
                    session_id=session_b,
                    lease_id=lease_b,
                    command_id="cmd-b",
                    payload={"tool_id": descriptor.tool_id, "parameters": {"all_screens": True}},
                ),
                session_id=session_b,
                lease_id=lease_b,
                device_id="sandbox-b",
                tool_id=descriptor.tool_id,
                parameters={"all_screens": True},
                lease_generation=generation_b,
            )

            result_a, result_b = await asyncio.gather(
                controller.command_bus.send(command_a, trace_id="trace-a"),
                controller.command_bus.send(command_b, trace_id="trace-b"),
            )

            assert result_a.status == CommandState.SUCCEEDED
            assert result_b.status == CommandState.SUCCEEDED
            assert fake_a.calls == [(descriptor.tool_id, {"all_screens": False})]
            assert fake_b.calls == [(descriptor.tool_id, {"all_screens": True})]
            assert result_a.device_id == "sandbox-a"
            assert result_b.device_id == "sandbox-b"
            rows_a = controller.store.artifact_rows(trace_id="trace-a")
            assert len(rows_a) == 1
            assert len(controller.store.artifact_rows(trace_id="trace-b")) == 1
            assert rows_a[0]["storage_kind"] == "device_local"
            artifact_json = json.loads(rows_a[0]["artifact_json"])
            assert artifact_json["uri"].startswith("agent-artifacts://sha256/")
            assert "inline_data" not in artifact_json
            assert not any((tmp_path / "blobs").iterdir())
        finally:
            for task in worker_tasks:
                task.cancel()
            for task in worker_tasks:
                with suppress(asyncio.CancelledError):
                    await task
            await controller.stop()
            clear_desktop_mcp_bridge()

    asyncio.run(_run())
