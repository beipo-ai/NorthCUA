"""Hardening regressions for device command authority and timeout behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable
from types import SimpleNamespace

import pytest

import northcua_device.executor as executor_module
import northcua_runtime_shared.execution as execution_module
from northcua_device.config import DeviceClientConfig
from northcua_device.device_actor import DeviceActor
from northcua_device.executor import CommandExecutor
from northcua_device.runtime.shell import EchoRuntimeAdapter
from northcua_device.worker import DeviceWorker
from northcua_protocol.capabilities import Capability, PlatformKind, ToolDescriptor
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    DeviceDrain,
    SessionLeaseGranted,
    SessionLeaseReleased,
)
from northcua_protocol.state import CommandState, DeviceState
from northcua_runtime_shared import RuntimeHealth
from northcua_runtime_shared.desktop_surface.action_contracts import desktop_action_contracts
from northcua_runtime_shared.desktop_surface.data_contracts import desktop_data_contracts
from northcua_runtime_shared.execution import RuntimeExecutionError, current_execution, execution_scope, fail_execution
from northcua_runtime_shared.tool_contracts import (
    DeviceCommandBinding,
    ToolAudience,
    ToolContract,
    ToolPlacement,
    ToolVisibility,
    define_tool,
)


def _executor() -> CommandExecutor:
    return CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.MACOS,
        adapters=[EchoRuntimeAdapter()],
        allowed_tools={"device.echo"},
    )


class _ContractBackedTestAdapter:
    @property
    def tools(self) -> list[ToolDescriptor]:
        raise NotImplementedError

    @property
    def tool_contracts(self) -> list[ToolContract]:
        return [_contract_from_descriptor(tool) for tool in self.tools]


def _contract_from_descriptor(descriptor: ToolDescriptor) -> ToolContract:
    return define_tool(
        tool_id=descriptor.tool_id,
        public_name=descriptor.tool_id,
        namespace="tests.device",
        audience=ToolAudience.RUNTIME,
        placement=ToolPlacement.DEVICE_COMMAND,
        visibility=ToolVisibility.RUNTIME_ONLY,
        capability=descriptor.capability,
        json_schema=descriptor.parameters_schema,
        timeout_seconds=descriptor.timeout_seconds,
        side_effect=descriptor.side_effect,
        requires_focus=descriptor.requires_focus,
        risk_level=descriptor.risk_level,
        max_payload_bytes=descriptor.max_payload_bytes,
        binding=DeviceCommandBinding(server_name=type(descriptor).__name__),
        source="tests.device.test_device_worker_hardening",
    )


def test_executor_enforces_runtime_timeout() -> None:
    class _SlowAdapter(_ContractBackedTestAdapter):
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="slow.tool",
                    capability=Capability.DESKTOP_INPUT,
                    timeout_seconds=0.01,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_INPUT: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            await asyncio.sleep(1)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    async def _run() -> None:
        events: list[tuple[str, dict[str, object]]] = []
        result = await CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[_SlowAdapter()],
            allowed_tools={"slow.tool"},
            event_writer=lambda event, **payload: events.append((event, payload)),
        ).execute(
            CommandRequest(
                command_id="cmd-timeout",
                idempotency_key="idem-timeout",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="slow.tool",
                timeout_seconds=30.0,
            )
        )

        assert result.status == CommandState.TIMED_OUT
        assert result.error == "COMMAND_TIMED_OUT"
        assert [event for event, _ in events] == [
            "device_command_prepare_started",
            "device_command_timeout_capped",
            "device_command_prepare_finished",
            "device_command_execution_started",
            "device_driver_call_started",
            "device_driver_call_finished",
        ]
        assert events[-1][1]["status"] == CommandState.TIMED_OUT.value
        assert events[-1][1]["error"] == "COMMAND_TIMED_OUT"

    asyncio.run(_run())


def test_executor_rejects_descriptor_only_runtime_adapter() -> None:
    class _DescriptorOnlyAdapter:
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [ToolDescriptor(tool_id="legacy.tool", capability=Capability.DESKTOP_INPUT)]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True)

        async def execute(self, command: CommandRequest) -> CommandResult:
            raise AssertionError("descriptor-only adapter must not execute")

    with pytest.raises(ValueError, match="descriptor-only runtime adapters are disabled"):
        CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[_DescriptorOnlyAdapter()],
            allowed_tools={"legacy.tool"},
        )


def test_executor_emits_failed_driver_call_telemetry() -> None:
    class _FailingAdapter(_ContractBackedTestAdapter):
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="failing.tool",
                    capability=Capability.DESKTOP_INPUT,
                    timeout_seconds=1.0,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_INPUT: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            del command
            raise ValueError("driver exploded")

    async def _run() -> None:
        events: list[tuple[str, dict[str, object]]] = []
        result = await CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[_FailingAdapter()],
            allowed_tools={"failing.tool"},
            event_writer=lambda event, **payload: events.append((event, payload)),
        ).execute(
            CommandRequest(
                command_id="cmd-failed",
                idempotency_key="idem-failed",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="failing.tool",
                timeout_seconds=1.0,
            )
        )

        assert result.status == CommandState.FAILED
        assert result.error == "driver exploded"
        assert events[-1][0] == "device_driver_call_finished"
        assert events[-1][1]["status"] == CommandState.FAILED.value
        assert events[-1][1]["error"] == "driver exploded"

    asyncio.run(_run())


def test_executor_reports_tool_stage_timings_without_returning_them() -> None:
    class _TimedAdapter(_ContractBackedTestAdapter):
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [ToolDescriptor(tool_id="timed.tool", capability=Capability.DESKTOP_INPUT, timeout_seconds=1.0)]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_INPUT: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                result={"ok": True, "northcua_stage_timings": {"backend_action_ms": 7}},
            )

    async def _run() -> None:
        events: list[tuple[str, dict[str, object]]] = []
        result = await CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[_TimedAdapter()],
            allowed_tools={"timed.tool"},
            event_writer=lambda event, **payload: events.append((event, payload)),
        ).execute(
            CommandRequest(
                command_id="cmd-timed",
                idempotency_key="idem-timed",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="timed.tool",
                timeout_seconds=1.0,
            )
        )

        assert result.result == {"ok": True}
        finished = next(payload for event, payload in events if event == "device_driver_call_finished")
        assert finished["tool_stage_timings"] == {"backend_action_ms": 7}

    asyncio.run(_run())


@pytest.mark.parametrize("controller_policy", [True, False], ids=["controller-floor", "descriptor-cap"])
def test_executor_honors_controller_timeout_policy_metadata(
    monkeypatch: pytest.MonkeyPatch, controller_policy: bool
) -> None:
    # Freeze only the two command-budget clocks, not asyncio's scheduler clock.
    # Advancing logical driver time past the descriptor cap must succeed only
    # with the controller floor, regardless of host/thread scheduling latency.
    clock = [100.0]
    expected_budget = 0.04 if controller_policy else 0.01
    observed_waits: list[float | None] = []
    caller_thread = threading.get_ident()
    command_clock = SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter)
    monkeypatch.setattr(executor_module, "time", command_clock)
    monkeypatch.setattr(execution_module, "time", command_clock)

    async def _observe_wait_for(execution: Awaitable[CommandResult], timeout: float | None) -> CommandResult:
        observed_waits.append(timeout)
        assert timeout == pytest.approx(expected_budget)
        # A real watchdog still bounds the test. The 10/40 ms production budget
        # is checked above and by the real executor after logical driver time.
        return await asyncio.wait_for(execution, timeout=5.0)

    monkeypatch.setattr(
        executor_module, "asyncio", SimpleNamespace(**(vars(asyncio) | {"wait_for": _observe_wait_for}))
    )

    class _SlowAdapter(_ContractBackedTestAdapter):
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="slow.tool",
                    capability=Capability.DESKTOP_INPUT,
                    timeout_seconds=0.01,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_INPUT: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            # Keep the actual isolated executor/fresh event loop and copied
            # execution context in this regression; do not stub the adapter hop.
            assert threading.get_ident() != caller_thread
            context = current_execution()
            assert context is not None
            assert command.timeout_seconds == pytest.approx(expected_budget)
            assert context.remaining() == pytest.approx(expected_budget)
            clock[0] += 0.02
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    async def _run() -> None:
        result = await CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[_SlowAdapter()],
            allowed_tools={"slow.tool"},
        ).execute(
            CommandRequest(
                command_id="cmd-controller-floor",
                idempotency_key="idem-controller-floor",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="slow.tool",
                timeout_seconds=0.04,
                metadata=(
                    {"controller_timeout_policy": {"effective_timeout_seconds": 0.04}} if controller_policy else {}
                ),
            )
        )

        assert observed_waits == [pytest.approx(expected_budget)]
        assert result.status == (CommandState.SUCCEEDED if controller_policy else CommandState.TIMED_OUT)
        assert result.error == (None if controller_policy else "COMMAND_TIMED_OUT")

    asyncio.run(_run())


def test_executor_isolates_blocking_gui_adapter_timeout() -> None:
    class _BlockingGuiAdapter(_ContractBackedTestAdapter):
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="blocking.click",
                    capability=Capability.DESKTOP_INPUT,
                    timeout_seconds=0.02,
                    requires_focus=True,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_INPUT: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            time.sleep(0.25)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    async def _run() -> None:
        executor = CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[_BlockingGuiAdapter()],
            allowed_tools={"blocking.click"},
        )
        started = time.perf_counter()
        result = await executor.execute(
            CommandRequest(
                command_id="cmd-blocking-timeout",
                idempotency_key="idem-blocking-timeout",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="blocking.click",
                timeout_seconds=30.0,
            )
        )
        elapsed = time.perf_counter() - started

        assert result.status == CommandState.TIMED_OUT
        assert result.error == "COMMAND_TIMED_OUT"
        assert elapsed < 0.15

    asyncio.run(_run())


def test_executor_does_not_overtake_timed_out_mutation() -> None:
    class _SometimesBlockingGuiAdapter(_ContractBackedTestAdapter):
        def __init__(self) -> None:
            self.calls = 0
            self.release = threading.Event()
            self.finished = threading.Event()

        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.DESKTOP_INPUT}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="blocking.click",
                    capability=Capability.DESKTOP_INPUT,
                    timeout_seconds=0.01,
                    requires_focus=True,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_INPUT: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            self.calls += 1
            if self.calls == 1:
                assert self.release.wait(2)
                self.finished.set()
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    async def _run() -> None:
        adapter = _SometimesBlockingGuiAdapter()
        executor = CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[adapter],
            allowed_tools={"blocking.click"},
        )
        request = CommandRequest(
            command_id="cmd-blocking-timeout",
            idempotency_key="idem-blocking-timeout",
            session_id="sess",
            lease_id="lease",
            device_id="dev-1",
            tool_id="blocking.click",
            timeout_seconds=30.0,
        )
        try:
            first = await executor.execute(request)
            second = await executor.execute(request.model_copy(update={"command_id": "cmd-blocking-timeout-2"}))
            assert first.status == CommandState.TIMED_OUT
            assert second.error == "COMMAND_EXECUTION_STILL_RUNNING"
            assert adapter.calls == 1
        finally:
            adapter.release.set()
        assert await asyncio.to_thread(adapter.finished.wait, 2)
        await asyncio.gather(*(asyncio.wrap_future(f) for f in executor._pending_gui_mutations))
        third = await executor.execute(request.model_copy(update={"command_id": "cmd-after-completion"}))
        assert third.status == CommandState.SUCCEEDED
        assert adapter.calls == 2

    asyncio.run(_run())


def test_device_queue_budget_expires_before_driver_dispatch(tmp_path) -> None:
    sent = []

    class Connection:
        async def send(self, envelope):
            sent.append(envelope)

    async def run():
        worker = DeviceWorker(
            DeviceClientConfig(device_id="dev-1", log_dir=str(tmp_path / "logs")),
            adapters=[EchoRuntimeAdapter()],
            connection=Connection(),
        )
        lease = SessionLeaseGranted(
            lease_id="lease", session_id="session", task_id="task", device_id="dev-1", trace_id="trace"
        )
        worker.actor.accept_lease(lease)
        request = CommandRequest(
            command_id="queued",
            idempotency_key="queued",
            session_id="session",
            lease_id="lease",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=0.01,
        )
        worker._command_received_at[request.command_id] = time.monotonic() - 1
        await worker._handle_envelope(Envelope.wrap(MessageType.COMMAND_REQUEST, request, trace_id="trace"))
        assert sent[-1].payload.status == CommandState.TIMED_OUT
        assert sent[-1].payload.result is None

    asyncio.run(run())


def test_executor_retains_failure_wrapped_by_mcp_and_propagates_deadline() -> None:
    class WrappedAdapter(_ContractBackedTestAdapter):
        tools = [ToolDescriptor(tool_id="wrapped.click", capability=Capability.DESKTOP_INPUT, timeout_seconds=2)]

        async def execute(self, command):
            context = current_execution()
            assert context is not None and 0 < context.remaining() <= 0.5
            try:
                fail_execution("COMMAND_DELIVERY_UNKNOWN")
            except RuntimeExecutionError:
                # Simulate an MCP layer turning exceptions into tool output.
                return CommandResult(
                    command_id=command.command_id,
                    session_id=command.session_id,
                    lease_id=command.lease_id,
                    device_id=command.device_id,
                    status=CommandState.SUCCEEDED,
                    result={"text": "tool failed"},
                )

    async def run():
        executor = CommandExecutor(
            device_id="dev-1", platform=PlatformKind.LINUX, adapters=[WrappedAdapter()], allowed_tools=set()
        )
        request = CommandRequest(
            command_id="wrapped",
            idempotency_key="wrapped",
            session_id="session",
            lease_id="lease",
            device_id="dev-1",
            tool_id="wrapped.click",
            timeout_seconds=2,
        )
        with execution_scope(time.monotonic() + 0.5):
            result = await executor.execute(request)
        assert result.status == CommandState.FAILED
        assert result.error == "COMMAND_DELIVERY_UNKNOWN"
        assert result.result is None

    asyncio.run(run())


def test_cancelled_gui_thread_blocks_mutations_but_preserves_readonly_contracts() -> None:
    release, started = threading.Event(), threading.Event()
    calls = []

    class Adapter(_ContractBackedTestAdapter):
        tools = [
            ToolDescriptor(tool_id="test.click", capability=Capability.DESKTOP_INPUT, timeout_seconds=2),
            *(
                contract.to_descriptor()
                for contract in (*desktop_action_contracts(), *desktop_data_contracts())
                if contract.tool_id
                in {"screenshot", "note", "wait", "get_app_window_info", "get_desktop_app_target_info"}
            ),
        ]

        async def execute(self, command):
            calls.append(command.tool_id)
            if command.tool_id == "test.click":
                started.set()
                assert release.wait(2)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    async def run():
        executor = CommandExecutor(
            device_id="dev-1", platform=PlatformKind.LINUX, adapters=[Adapter()], allowed_tools=set()
        )
        request = CommandRequest(
            command_id="one",
            idempotency_key="one",
            session_id="session",
            lease_id="lease",
            device_id="dev-1",
            tool_id="test.click",
            timeout_seconds=2,
        )
        active = asyncio.create_task(executor.execute(request))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            blocked = await executor.execute(request.model_copy(update={"command_id": "two"}))
            assert blocked.error == "COMMAND_EXECUTION_STILL_RUNNING"
            readonly = {
                "screenshot": {},
                "note": {"text": "remember"},
                "wait": {"ms": 0},
                "get_app_window_info": {"field_list": []},
                "get_desktop_app_target_info": {},
            }
            for tool_id, parameters in readonly.items():
                result = await executor.execute(
                    request.model_copy(
                        update={
                            "command_id": tool_id,
                            "tool_id": tool_id,
                            "parameters": parameters,
                        }
                    )
                )
                assert result.status == CommandState.SUCCEEDED
            assert calls == ["test.click", *readonly]
        finally:
            release.set()
            await asyncio.gather(*(asyncio.wrap_future(f) for f in executor._pending_gui_mutations))

    asyncio.run(run())


def test_executor_keeps_browser_page_focus_tools_on_event_loop() -> None:
    events: list[dict[str, object]] = []

    class _BrowserFocusAdapter(_ContractBackedTestAdapter):
        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.BROWSER_PAGE}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="web.web_fill_ref",
                    capability=Capability.BROWSER_PAGE,
                    timeout_seconds=1.0,
                    requires_focus=True,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.BROWSER_PAGE: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    async def _run() -> None:
        executor = CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.LINUX,
            adapters=[_BrowserFocusAdapter()],
            allowed_tools={"web.web_fill_ref"},
            event_writer=lambda event, **payload: events.append({"event": event, **payload}),
        )
        result = await executor.execute(
            CommandRequest(
                command_id="cmd-browser-focus",
                idempotency_key="idem-browser-focus",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="web.web_fill_ref",
                timeout_seconds=30.0,
            )
        )

        assert result.status == CommandState.SUCCEEDED

    asyncio.run(_run())

    prepared = next(event for event in events if event["event"] == "device_command_prepare_finished")
    assert prepared["capability"] == Capability.BROWSER_PAGE.value
    assert prepared["isolated_execution"] is False


def test_device_actor_rejects_stale_or_wrong_device_lease_messages() -> None:
    actor = DeviceActor(device_id="dev-1", executor=_executor(), max_sessions=1, max_concurrent_commands=1)
    try:
        actor.accept_lease(
            SessionLeaseGranted(
                lease_id="lease-other",
                session_id="sess-other",
                task_id="task-other",
                device_id="dev-other",
                trace_id="trace",
            )
        )
    except RuntimeError as exc:
        assert "different device" in str(exc)
    else:
        raise AssertionError("worker should reject leases for another device")

    lease = SessionLeaseGranted(
        lease_id="lease-1",
        session_id="sess-1",
        task_id="task-1",
        device_id="dev-1",
        trace_id="trace-1",
    )
    actor.accept_lease(lease)
    try:
        actor.release_lease(
            SessionLeaseReleased(
                lease_id="lease-stale",
                session_id="sess-1",
                device_id="dev-1",
                reason="stale release",
            )
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("stale release should not release a newer lease")

    assert actor.active_sessions == 1


def test_worker_ignores_controller_messages_for_another_device(tmp_path) -> None:
    class _FakeConnection:
        def __init__(self) -> None:
            self.sent: list[Envelope] = []

        async def send(self, envelope: Envelope) -> None:
            self.sent.append(envelope)

    async def _run() -> None:
        connection = _FakeConnection()
        worker = DeviceWorker(
            DeviceClientConfig(device_id="dev-1", log_dir=str(tmp_path / "logs")),
            adapters=[EchoRuntimeAdapter()],
            connection=connection,  # type: ignore[arg-type]
        )
        await worker._handle_envelope(
            Envelope.wrap(
                MessageType.SESSION_LEASE_GRANTED,
                SessionLeaseGranted(
                    lease_id="lease-other",
                    session_id="sess-other",
                    task_id="task-other",
                    device_id="dev-other",
                    trace_id="trace",
                ),
                trace_id="trace",
            )
        )
        await worker._handle_envelope(
            Envelope.wrap(
                MessageType.COMMAND_REQUEST,
                CommandRequest(
                    command_id="cmd-other",
                    idempotency_key="idem-other",
                    session_id="sess-other",
                    lease_id="lease-other",
                    device_id="dev-other",
                    tool_id="device.echo",
                ),
                trace_id="trace",
            )
        )
        await worker._handle_envelope(
            Envelope.wrap(
                MessageType.DEVICE_DRAIN,
                DeviceDrain(device_id="dev-other", reason="wrong worker"),
                trace_id="trace",
            )
        )

        assert worker.actor.active_sessions == 0
        assert worker.actor.state == DeviceState.CONNECTING
        assert connection.sent == []

    asyncio.run(_run())
