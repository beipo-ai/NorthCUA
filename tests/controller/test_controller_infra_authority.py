"""Controller authority and lease rollback regression tests."""

from __future__ import annotations

import asyncio
import time

from northcua.controller.control_plane.leases import LeaseManager
from northcua.controller.control_plane.registry import DeviceRegistry
from northcua.controller.control_plane.scheduler import Scheduler
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.payloads import SessionLeaseGranted, SessionLeaseReleased
from tests.controller.test_controller_infra import (
    Capability,
    ControllerConfig,
    ControllerSecurityConfig,
    ControllerServer,
    DeviceHello,
    Envelope,
    MessageType,
    SessionState,
    TaskFinished,
    TaskRequest,
    TaskState,
    _descriptor,
    _FailingLeaseWebSocket,
    _FakeWebSocket,
    parse_envelope,
)


def test_submitter_without_device_lease_cannot_finish_task(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        device = _FakeWebSocket()
        submitter = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-a",
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="task A",
                    target_device_id="dev-a",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in device.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        try:
            await server._dispatch(
                submitter,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_FINISHED,
                    TaskFinished(
                        task_id="task-a",
                        trace_id="trace-task",
                        session_id=lease.session_id,  # type: ignore[attr-defined]
                        device_id="dev-a",
                        lease_id=lease.lease_id,  # type: ignore[attr-defined]
                        state=TaskState.FINISHED,
                    ),
                    trace_id="trace-task",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("submitter should not finish a leased task without device ownership")
        finally:
            server.store.close()

    asyncio.run(_run())


def test_interrupt_device_notifies_submitter_with_task_finished(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-a",
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task",
            ),
        )

        await server.task_service.interrupt_device(device_id="dev-a", trace_id="trace-task", reason="disconnect")

        await asyncio.sleep(0)
        finished = [parse_envelope(raw) for raw in ws.sent if parse_envelope(raw).type == MessageType.TASK_FINISHED]
        assert finished
        assert finished[-1].payload.state == TaskState.INTERRUPTED  # type: ignore[attr-defined]
        server.store.close()

    asyncio.run(_run())


def test_interrupt_device_releases_lease_when_submitter_notification_fails(tmp_path) -> None:
    async def _run() -> None:
        class _FailingTaskFinishedWebSocket(_FakeWebSocket):
            async def send(self, raw: str) -> None:
                if parse_envelope(raw).type == MessageType.TASK_FINISHED:
                    raise RuntimeError("submitter send failed")
                await super().send(raw)

        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FailingTaskFinishedWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-send-fails-before-release",
                    trace_id="trace-send-fails-before-release",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-send-fails-before-release",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )

        await server.task_service.interrupt_device(device_id="dev-a", trace_id="trace-disconnect", reason="disconnect")

        row = server.store.task_row("task-send-fails-before-release")
        session = server.store.session_row(lease.session_id)  # type: ignore[attr-defined]
        lease_row = server.store.lease_row(lease.lease_id)  # type: ignore[attr-defined]
        assert row is not None
        assert session is not None
        assert lease_row is not None
        assert row["status"] == TaskState.INTERRUPTED.value
        assert session["status"] == SessionState.INTERRUPTED.value
        assert lease_row["status"] == SessionState.RELEASED.value
        assert server.registry.get("dev-a").active_sessions == 0  # type: ignore[union-attr]
        assert lease.session_id not in server.task_service._lease_by_session_id  # type: ignore[attr-defined]  # noqa: SLF001
        server.store.close()

    asyncio.run(_run())


def test_disconnect_cleanup_reconnect_not_blocked_by_hanging_submitter_notification(tmp_path) -> None:
    async def _run() -> None:
        class _HangingTaskFinishedWebSocket(_FakeWebSocket):
            def __init__(self) -> None:
                super().__init__()
                self.task_finished_send_started = asyncio.Event()
                self.allow_task_finished_send = asyncio.Event()

            async def send(self, raw: str) -> None:
                if parse_envelope(raw).type == MessageType.TASK_FINISHED:
                    self.task_finished_send_started.set()
                    await self.allow_task_finished_send.wait()
                await super().send(raw)

        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                device_disconnect_grace_seconds=0.0,
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _HangingTaskFinishedWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-hanging-submitter-notify",
                    trace_id="trace-hanging-submitter-notify",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-hanging-submitter-notify",
            ),
        )
        server.connections.unregister(ws)  # type: ignore[arg-type]
        server._schedule_device_disconnect_cleanup(
            device_id="dev-a",
            trace_id="trace-disconnect",
            reason="websocket disconnected",
        )
        await asyncio.wait_for(ws.task_finished_send_started.wait(), timeout=1.0)
        for _ in range(20):
            if "dev-a" not in server._device_disconnect_cleanup_tasks:  # noqa: SLF001
                break
            await asyncio.sleep(0.01)
        assert "dev-a" not in server._device_disconnect_cleanup_tasks  # noqa: SLF001

        reconnect = _FakeWebSocket()
        await asyncio.wait_for(
            server._dispatch(  # type: ignore[arg-type]
                reconnect,
                Envelope.wrap(
                    MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-b"
                ),
            ),
            timeout=1.0,
        )
        active_connection = server.connections.for_device("dev-a")
        assert active_connection is not None
        assert active_connection.websocket is reconnect
        row = server.store.task_row("task-hanging-submitter-notify")
        assert row is not None
        assert row["status"] == TaskState.INTERRUPTED.value
        ws.allow_task_finished_send.set()
        await asyncio.sleep(0)
        server.store.close()

    asyncio.run(_run())


def test_interrupt_device_does_not_overwrite_finished_task(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-finished-before-disconnect",
                    trace_id="trace-task-finished",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task-finished",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_FINISHED,
                TaskFinished(
                    task_id="task-finished-before-disconnect",
                    trace_id="trace-task-finished",
                    session_id=lease.session_id,  # type: ignore[attr-defined]
                    device_id="dev-a",
                    lease_id=lease.lease_id,  # type: ignore[attr-defined]
                    state=TaskState.FINISHED,
                    result={"winner": "device"},
                ),
                trace_id="trace-task-finished",
            ),
        )

        before_disconnect_finished = [
            parse_envelope(raw).payload for raw in ws.sent if parse_envelope(raw).type == MessageType.TASK_FINISHED
        ]
        await server.task_service.interrupt_device(device_id="dev-a", trace_id="trace-disconnect", reason="disconnect")
        after_disconnect_finished = [
            parse_envelope(raw).payload for raw in ws.sent if parse_envelope(raw).type == MessageType.TASK_FINISHED
        ]

        row = server.store.task_row("task-finished-before-disconnect")
        assert row is not None
        assert row["status"] == TaskState.FINISHED.value
        assert "device" in str(row["result_json"])
        assert len(after_disconnect_finished) == len(before_disconnect_finished)
        server.store.close()

    asyncio.run(_run())


def test_interrupt_device_does_not_interrupt_session_after_task_finish_cas(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-finish-cas-before-disconnect",
                    trace_id="trace-finish-cas-before-disconnect",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-finish-cas-before-disconnect",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        won = await server.queue.finish_async(
            TaskFinished(
                task_id="task-finish-cas-before-disconnect",
                trace_id="trace-finish-cas-before-disconnect",
                session_id=lease.session_id,  # type: ignore[attr-defined]
                device_id="dev-a",
                lease_id=lease.lease_id,  # type: ignore[attr-defined]
                state=TaskState.FINISHED,
                result={"winner": "device-cas"},
            )
        )
        assert won

        await server.task_service.interrupt_device(device_id="dev-a", trace_id="trace-disconnect", reason="disconnect")

        row = server.store.task_row("task-finish-cas-before-disconnect")
        session = server.store.session_row(lease.session_id)  # type: ignore[attr-defined]
        assert row is not None
        assert session is not None
        assert row["status"] == TaskState.FINISHED.value
        assert "device-cas" in str(row["result_json"])
        assert session["status"] != SessionState.INTERRUPTED.value
        server.store.close()

    asyncio.run(_run())


def test_accept_cancel_after_db_commit_terminalizes_pending_orphan(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        original_create_task = SQLiteControllerStore.create_task

        def slow_after_commit(self: SQLiteControllerStore, *args, **kwargs) -> None:
            original_create_task(self, *args, **kwargs)
            time.sleep(0.05)

        monkeypatch.setattr(SQLiteControllerStore, "create_task", slow_after_commit)
        ws = _FakeWebSocket()
        task = asyncio.create_task(
            server._dispatch(
                ws,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id="task-cancelled-after-accept-commit",
                        trace_id="trace-cancelled-after-accept-commit",
                        agent_kind="desktop",
                        request="task A",
                        required_capabilities={Capability.DESKTOP_INPUT},
                    ),
                    trace_id="trace-cancelled-after-accept-commit",
                ),
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("task request dispatch should observe cancellation")

        row = server.store.task_row("task-cancelled-after-accept-commit")
        assert row is not None
        assert row["status"] == TaskState.INTERRUPTED.value
        assert "acceptance cancelled" in str(row["result_json"])
        assert "task-cancelled-after-accept-commit" not in server.task_service._submitter_by_task_id
        server.store.close()

    asyncio.run(_run())


def test_lease_delivery_failure_rolls_back_to_pending(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FailingLeaseWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )

        try:
            await server._dispatch(
                ws,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id="task-a",
                        trace_id="trace-task",
                        agent_kind="desktop",
                        request="task A",
                        required_capabilities={Capability.DESKTOP_INPUT},
                    ),
                    trace_id="trace-task",
                ),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("lease delivery should fail")

        session = server.store.session_row_for_task("task-a")
        assert session["status"] == "pending"
        assert session["lease_id"] is None
        assert server.registry.get("dev-a").active_sessions == 0  # type: ignore[union-attr]
        server.store.close()

    asyncio.run(_run())


def test_lease_delivery_cancel_after_commit_does_not_leave_scheduled_orphan(tmp_path) -> None:
    async def _run() -> None:
        class _BlockingLeaseWebSocket(_FakeWebSocket):
            def __init__(self) -> None:
                super().__init__()
                self.lease_send_started = asyncio.Event()
                self.allow_lease_send = asyncio.Event()

            async def send(self, raw: str) -> None:
                if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED:
                    self.lease_send_started.set()
                    await self.allow_lease_send.wait()
                await super().send(raw)

        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _BlockingLeaseWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        dispatch_task = asyncio.create_task(
            server._dispatch(
                ws,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id="task-cancel-during-lease-send",
                        trace_id="trace-cancel-during-lease-send",
                        agent_kind="desktop",
                        request="task A",
                        required_capabilities={Capability.DESKTOP_INPUT},
                    ),
                    trace_id="trace-cancel-during-lease-send",
                ),
            )
        )
        await asyncio.wait_for(ws.lease_send_started.wait(), timeout=1.0)
        dispatch_task.cancel()
        await asyncio.sleep(0)
        ws.allow_lease_send.set()
        try:
            await dispatch_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("dispatch should preserve caller cancellation")

        row = server.store.task_row("task-cancel-during-lease-send")
        session = server.store.session_row_for_task("task-cancel-during-lease-send")
        assert row is not None
        assert session is not None
        assert row["status"] == TaskState.RUNNING.value
        assert session["status"] == SessionState.RUNNING.value
        assert server.registry.get("dev-a").active_sessions == 1  # type: ignore[union-attr]
        server.store.close()

    asyncio.run(_run())


def test_release_cancel_after_db_commit_releases_registry(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        store = SQLiteControllerStore(tmp_path / "controller.sqlite")
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(_descriptor("dev-release-cancel"), trace_id="trace-register")
        scheduler = Scheduler(registry)
        leases = LeaseManager(store=store, scheduler=scheduler, events=events)
        request = TaskRequest(
            task_id="task-release-cancel",
            trace_id="trace-release-cancel",
            agent_kind="desktop",
            request="task A",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        store.create_task(request, session_id="sess-release-cancel", submitter_id="submitter")
        store.create_lease_if_schedulable(
            lease_id="lease-release-cancel",
            session_id="sess-release-cancel",
            task_id=request.task_id,
            device_id="dev-release-cancel",
        )
        registry.get("dev-release-cancel").active_sessions = 1  # type: ignore[union-attr]
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        original_release = registry.release_session_async

        async def slow_release(device_id: str) -> None:
            release_started.set()
            await allow_release.wait()
            await original_release(device_id)

        monkeypatch.setattr(registry, "release_session_async", slow_release)
        release_task = asyncio.create_task(
            leases.release_async(
                SessionLeaseReleased(
                    lease_id="lease-release-cancel",
                    session_id="sess-release-cancel",
                    device_id="dev-release-cancel",
                    state=SessionState.RELEASED,
                    reason="test cancellation",
                ),
                trace_id="trace-release-cancel",
            )
        )
        await asyncio.wait_for(release_started.wait(), timeout=1.0)
        release_task.cancel()
        allow_release.set()
        try:
            await release_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("lease release should preserve caller cancellation")

        assert registry.get("dev-release-cancel").active_sessions == 0  # type: ignore[union-attr]
        assert store.lease_row("lease-release-cancel")["status"] == SessionState.RELEASED.value
        assert store.session_row("sess-release-cancel")["status"] == SessionState.RELEASED.value
        store.close()

    asyncio.run(_run())


def test_rollback_cancel_after_db_commit_releases_registry_and_requeues(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        store = SQLiteControllerStore(tmp_path / "controller.sqlite")
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(_descriptor("dev-rollback-cancel"), trace_id="trace-register")
        scheduler = Scheduler(registry)
        leases = LeaseManager(store=store, scheduler=scheduler, events=events)
        request = TaskRequest(
            task_id="task-rollback-cancel",
            trace_id="trace-rollback-cancel",
            agent_kind="desktop",
            request="task A",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        store.create_task(request, session_id="sess-rollback-cancel", submitter_id="submitter")
        store.create_lease_if_schedulable(
            lease_id="lease-rollback-cancel",
            session_id="sess-rollback-cancel",
            task_id=request.task_id,
            device_id="dev-rollback-cancel",
        )
        registry.get("dev-rollback-cancel").active_sessions = 1  # type: ignore[union-attr]
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        original_release = registry.release_session_async

        async def slow_release(device_id: str) -> None:
            release_started.set()
            await allow_release.wait()
            await original_release(device_id)

        monkeypatch.setattr(registry, "release_session_async", slow_release)
        rollback_task = asyncio.create_task(
            leases.rollback_undelivered_async(
                SessionLeaseGranted(
                    lease_id="lease-rollback-cancel",
                    session_id="sess-rollback-cancel",
                    task_id=request.task_id,
                    device_id="dev-rollback-cancel",
                    trace_id="trace-rollback-cancel",
                ),
                trace_id="trace-rollback-cancel",
                reason="test cancellation",
            )
        )
        await asyncio.wait_for(release_started.wait(), timeout=1.0)
        rollback_task.cancel()
        allow_release.set()
        try:
            await rollback_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("lease rollback should preserve caller cancellation")

        assert registry.get("dev-rollback-cancel").active_sessions == 0  # type: ignore[union-attr]
        assert store.lease_row("lease-rollback-cancel")["status"] == SessionState.RELEASED.value
        assert store.session_row("sess-rollback-cancel")["status"] == SessionState.PENDING.value
        assert store.task_row(request.task_id)["status"] == TaskState.ACCEPTED.value
        store.close()

    asyncio.run(_run())


def test_disconnect_cleanup_cancel_after_db_release_releases_registry(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-disconnect-cleanup-cancel",
                    trace_id="trace-disconnect-cleanup-cancel",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-disconnect-cleanup-cancel",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        original_release = server.registry.release_session_async

        async def slow_release(device_id: str) -> None:
            release_started.set()
            await allow_release.wait()
            await original_release(device_id)

        monkeypatch.setattr(server.registry, "release_session_async", slow_release)
        cleanup_task = asyncio.create_task(
            server.task_service._lease_releases._release_active_device_leases(  # noqa: SLF001
                device_id="dev-a",
                trace_id="trace-disconnect-cleanup-cancel",
                reason="disconnect cleanup",
            )
        )
        await asyncio.wait_for(release_started.wait(), timeout=1.0)
        cleanup_task.cancel()
        allow_release.set()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("disconnect cleanup should preserve caller cancellation")

        assert server.registry.get("dev-a").active_sessions == 0  # type: ignore[union-attr]
        assert lease.session_id not in server.task_service._lease_by_session_id  # type: ignore[attr-defined]  # noqa: SLF001
        assert server.store.lease_row(lease.lease_id)["status"] == SessionState.RELEASED.value  # type: ignore[attr-defined]
        assert server.store.session_row(lease.session_id)["status"] == SessionState.RELEASED.value  # type: ignore[attr-defined]
        server.store.close()

    asyncio.run(_run())
