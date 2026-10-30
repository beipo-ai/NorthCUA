"""Correctness and observability coverage for the command SQLite fast path."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import pytest

import northcua.controller.storage.async_store as async_store
from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.execution.command_artifacts import CommandArtifactMaterializer
from northcua.controller.execution.command_authority import validate_and_claim_active_command
from northcua.controller.execution.command_results import CommandResultRecorder
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.async_store import (
    ControllerStorePriority,
    controller_store_call_profiled,
    controller_store_read_profiled,
)
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.payloads import CommandRequest, CommandResult, TaskRequest
from northcua_protocol.state import CommandState, TaskState


def _active_store(tmp_path) -> SQLiteControllerStore:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    store.create_task(
        TaskRequest(
            task_id="task-fast",
            trace_id="trace-fast",
            agent_kind="desktop",
            request="benchmark",
        ),
        session_id="session-fast",
        submitter_id="submitter-fast",
        status=TaskState.ACCEPTED,
    )
    store.create_lease(
        lease_id="lease-fast",
        session_id="session-fast",
        task_id="task-fast",
        device_id="device-fast",
    )
    return store


def _command(*, command_id: str = "command-fast", generation: int = 1) -> CommandRequest:
    return CommandRequest(
        command_id=command_id,
        idempotency_key=f"idem-{command_id}",
        session_id="session-fast",
        lease_id="lease-fast",
        device_id="device-fast",
        lease_generation=generation,
        tool_id="device.echo",
    )


def test_command_normal_path_uses_five_data_sql_statements(tmp_path) -> None:
    store = _active_store(tmp_path)
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)
    request = _command()
    try:
        outcome = validate_and_claim_active_command(store, request)
        assert outcome.claim == "new"
        claim_data_statements = _data_statements(statements)
        claim_statements = _command_statements(statements)
        assert len(claim_data_statements) == 3
        assert len(claim_statements) == 1
        assert claim_statements[0].lstrip().upper().startswith("INSERT INTO COMMANDS")

        statements.clear()
        assert store.mark_command_dispatched_for_request(request) is None
        dispatch_data_statements = _data_statements(statements)
        dispatch_statements = _command_statements(statements)
        assert len(dispatch_data_statements) == 1
        assert len(dispatch_statements) == 1
        assert dispatch_statements[0].lstrip().upper().startswith("UPDATE COMMANDS")

        statements.clear()
        result = CommandResult(
            command_id=request.command_id,
            session_id=request.session_id,
            lease_id=request.lease_id,
            device_id=request.device_id,
            lease_generation=request.lease_generation,
            status=CommandState.SUCCEEDED,
            result={"ok": True},
        )
        assert store.record_command_result(result) is None
        result_data_statements = _data_statements(statements)
        result_statements = _command_statements(statements)
        assert len(result_data_statements) == 1
        assert len(result_statements) == 1
        assert result_statements[0].lstrip().upper().startswith("UPDATE COMMANDS")
        assert len(claim_data_statements + dispatch_data_statements + result_data_statements) == 5

        row = store.command_row(request.command_id, session_id=request.session_id, lease_id=request.lease_id)
        assert row is not None
        assert row["status"] == CommandState.SUCCEEDED.value
        assert row["result_json"]
    finally:
        store.close()


def test_async_result_recorder_normal_path_does_not_pre_read_command(tmp_path, monkeypatch) -> None:
    store = _active_store(tmp_path)
    request = _command(command_id="command-recorder")
    try:
        assert validate_and_claim_active_command(store, request).claim == "new"
        assert store.mark_command_dispatched_for_request(request) is None
        recorder = CommandResultRecorder(
            store=store,
            events=ControllerEventLog(store, command_stage_event_mode="off"),
            artifacts=CommandArtifactMaterializer(
                store=store,
                blob_store=None,
                max_inline_artifact_bytes=1024,
            ),
        )

        def _unexpected_command_read(*_args, **_kwargs):
            raise AssertionError("normal async result persistence must not pre-read the command row")

        def _unexpected_session_read(*_args, **_kwargs):
            raise AssertionError("artifact-free result persistence must not re-read the session row")

        monkeypatch.setattr(SQLiteControllerStore, "command_row", _unexpected_command_read)
        monkeypatch.setattr(SQLiteControllerStore, "session_row", _unexpected_session_read)

        recorded = asyncio.run(
            recorder.record_async(
                CommandResult(
                    command_id=request.command_id,
                    session_id=request.session_id,
                    lease_id=request.lease_id,
                    device_id=request.device_id,
                    lease_generation=request.lease_generation,
                    status=CommandState.SUCCEEDED,
                    result={"ok": True},
                ),
                trace_id="trace-fast",
            )
        )

        assert recorded.status == CommandState.SUCCEEDED
    finally:
        store.close()


def test_async_result_recorder_fills_same_status_terminal_without_result_json(tmp_path) -> None:
    store = _active_store(tmp_path)
    request = _command(command_id="command-terminal-gap")
    try:
        assert validate_and_claim_active_command(store, request).claim == "new"
        assert store.mark_command_dispatched_for_request(request) is None
        store.connection.execute(
            """
            UPDATE commands
            SET status=?, result_json=NULL
            WHERE command_id=? AND session_id=? AND lease_id=?
            """,
            (
                CommandState.SUCCEEDED.value,
                request.command_id,
                request.session_id,
                request.lease_id,
            ),
        )
        store.connection.commit()
        preloaded = store.command_row(
            request.command_id,
            session_id=request.session_id,
            lease_id=request.lease_id,
        )
        assert preloaded is not None
        assert preloaded["result_json"] is None
        recorder = CommandResultRecorder(
            store=store,
            events=ControllerEventLog(store, command_stage_event_mode="off"),
            artifacts=CommandArtifactMaterializer(
                store=store,
                blob_store=None,
                max_inline_artifact_bytes=1024,
            ),
        )
        incoming = CommandResult(
            command_id=request.command_id,
            session_id=request.session_id,
            lease_id=request.lease_id,
            device_id=request.device_id,
            lease_generation=request.lease_generation,
            status=CommandState.SUCCEEDED,
            result={"filled": True},
        )

        recorded, persisted = asyncio.run(
            recorder.record_async_with_status(
                incoming,
                trace_id="trace-fast",
                preloaded_command_row=dict(preloaded),
            )
        )

        row = store.command_row(
            request.command_id,
            session_id=request.session_id,
            lease_id=request.lease_id,
        )
        assert persisted is True
        assert recorded.result == {"filled": True}
        assert row is not None
        assert row["result_json"]
        assert CommandResult.model_validate_json(str(row["result_json"])).result == {"filled": True}
    finally:
        store.close()


def test_validate_and_claim_rolls_back_stale_authority(tmp_path) -> None:
    store = _active_store(tmp_path)
    request = _command(command_id="command-stale", generation=2)
    try:
        with pytest.raises(PermissionError, match="lease_generation is stale"):
            validate_and_claim_active_command(store, request)
        assert store.command_row(request.command_id, session_id=request.session_id, lease_id=request.lease_id) is None
        assert store.connection.in_transaction is False
    finally:
        store.close()


def test_read_pool_is_query_only_and_reports_metrics(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        try:
            row, metrics = await controller_store_read_profiled(
                store.path,
                lambda reader: reader.session_row("session-fast"),
                operation_name="test_read_session",
            )
            assert row is not None
            assert metrics.operation == "test_read_session"
            assert metrics.mode == "read"
            assert metrics.queue_ms >= 0
            assert metrics.operation_ms >= metrics.exec_ms
            assert metrics.exec_ms >= 0
            assert metrics.total_ms >= metrics.exec_ms
            assert metrics.queue_depth >= 0
            assert metrics.sql_count == 1

            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                await controller_store_read_profiled(
                    store.path,
                    lambda reader: reader.connection.execute(
                        "UPDATE sessions SET status=status WHERE session_id='session-fast'"
                    ),
                    operation_name="test_read_pool_rejects_write",
                )
        finally:
            store.close()

    asyncio.run(_run())


def test_writer_metrics_expose_real_queue_depth(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        try:

            async def one_call(index: int):
                return await controller_store_call_profiled(
                    store.path,
                    lambda _writer: time.sleep(0.015),
                    operation_name=f"queued-{index}",
                )

            outcomes = await asyncio.gather(*(one_call(index) for index in range(4)))
            metrics = [item[1] for item in outcomes]
            assert max(item.queue_depth for item in metrics) >= 2
            assert max(item.queue_ms for item in metrics) >= 20
            assert all(item.mode == "write" for item in metrics)
            assert all(item.operation_ms >= 15 for item in metrics)
            assert all(item.exec_ms == 0 for item in metrics)
            assert all(item.sql_count == 0 for item in metrics)
        finally:
            store.close()

    asyncio.run(_run())


def test_critical_writer_overtakes_bounded_normal_backlog(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        writer_gate = threading.Event()
        writer_started = threading.Event()
        execution_order: list[str] = []
        try:

            def _block_writer(_store) -> None:
                writer_started.set()
                writer_gate.wait(timeout=2)
                execution_order.append("blocker")

            blocker = asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    _block_writer,
                    operation_name="priority-blocker",
                )
            )
            await _wait_for_thread_events([writer_started])

            async def _normal(index: int):
                return await controller_store_call_profiled(
                    store.path,
                    lambda _store: execution_order.append(f"normal-{index}"),
                    operation_name=f"normal-{index}",
                )

            normal = [asyncio.create_task(_normal(index)) for index in range(160)]
            await asyncio.sleep(0.02)
            critical = asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    lambda _store: execution_order.append("critical"),
                    operation_name="task_accept",
                    priority=ControllerStorePriority.CRITICAL,
                )
            )
            await asyncio.sleep(0.02)
            writer_gate.set()

            _result, critical_metrics = await asyncio.wait_for(critical, timeout=1)
            outcomes = await asyncio.gather(blocker, *normal)
            queue_depths = [critical_metrics.queue_depth, *(item[1].queue_depth for item in outcomes)]
            admitted_depths = [
                critical_metrics.admitted_queue_depth,
                *(item[1].admitted_queue_depth for item in outcomes),
            ]

            assert execution_order[:2] == ["blocker", "critical"]
            assert max(queue_depths) >= 128
            assert max(admitted_depths) < 128
        finally:
            writer_gate.set()
            store.close()

    asyncio.run(_run())


def test_command_writer_keeps_reserved_admission_under_normal_backlog(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        previous_executor = async_store._EXECUTOR
        previous_admission = async_store._WRITER_ADMISSION
        executor = async_store._PrioritizedWriterExecutor()
        async_store._EXECUTOR = executor
        async_store._WRITER_ADMISSION = async_store._WriterAdmission(
            maximum=4,
            command_maximum=2,
            command_reserved=1,
            critical_reserved=1,
            normal_maximum=4,
            background_maximum=1,
        )
        gate = threading.Event()
        started = threading.Event()
        execution_order: list[str] = []
        blocker: asyncio.Task | None = None
        normals: list[asyncio.Task] = []
        try:

            def _blocker(_store) -> None:
                started.set()
                gate.wait(timeout=2)
                execution_order.append("blocker")

            blocker = asyncio.create_task(
                controller_store_call_profiled(store.path, _blocker, operation_name="command-reserve-blocker")
            )
            await _wait_for_thread_events([started])
            normals = [
                asyncio.create_task(
                    controller_store_call_profiled(
                        store.path,
                        lambda _store, index=index: execution_order.append(f"normal-{index}"),
                        operation_name=f"normal-{index}",
                    )
                )
                for index in range(20)
            ]
            await asyncio.sleep(0.02)
            command = asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    lambda _store: execution_order.append("command"),
                    operation_name="command_validate_and_claim",
                    priority=ControllerStorePriority.COMMAND,
                )
            )
            await _wait_for_writer_queue(executor, ControllerStorePriority.COMMAND)
            gate.set()

            _result, metrics = await asyncio.wait_for(command, timeout=1)
            await asyncio.gather(blocker, *normals)
            assert execution_order[:2] == ["blocker", "command"]
            assert metrics.admission_ms < 1000
        finally:
            gate.set()
            pending = ([blocker] if blocker is not None else []) + normals
            await asyncio.gather(*pending, return_exceptions=True)
            async_store._EXECUTOR = previous_executor
            async_store._WRITER_ADMISSION = previous_admission
            executor.shutdown()
            store.close()

    asyncio.run(_run())


def test_background_writer_has_bounded_service_under_critical_load() -> None:
    executor = async_store._PrioritizedWriterExecutor()
    gate = threading.Event()
    started = threading.Event()
    execution_order: list[str] = []

    def _blocker() -> None:
        started.set()
        gate.wait(timeout=2)
        execution_order.append("blocker")

    try:
        blocker = executor.submit(_blocker, priority=ControllerStorePriority.CRITICAL)
        assert started.wait(timeout=1)
        critical = [
            executor.submit(
                lambda index=index: execution_order.append(f"critical-{index}"),
                priority=ControllerStorePriority.CRITICAL,
            )
            for index in range(60)
        ]
        background = executor.submit(
            lambda: execution_order.append("background"),
            priority=ControllerStorePriority.BACKGROUND,
        )
        gate.set()

        background.result(timeout=1)
        assert execution_order.index("background") <= 9
        blocker.result(timeout=1)
        for future in critical:
            future.result(timeout=1)
    finally:
        gate.set()
        executor.shutdown()


def test_background_writer_has_bounded_service_under_replenishing_critical_load() -> None:
    executor = async_store._PrioritizedWriterExecutor()
    gate = threading.Event()
    started = threading.Event()
    execution_order: list[str] = []

    def _blocker() -> None:
        started.set()
        gate.wait(timeout=2)
        execution_order.append("blocker")

    def _critical(index: int) -> None:
        execution_order.append(f"critical-{index}")
        if index < 59:
            executor.submit(
                lambda: _critical(index + 1),
                priority=ControllerStorePriority.CRITICAL,
            )

    try:
        blocker = executor.submit(_blocker, priority=ControllerStorePriority.CRITICAL)
        assert started.wait(timeout=1)
        first = executor.submit(lambda: _critical(0), priority=ControllerStorePriority.CRITICAL)
        background = executor.submit(
            lambda: execution_order.append("background"),
            priority=ControllerStorePriority.BACKGROUND,
        )
        gate.set()

        background.result(timeout=1)
        assert execution_order.index("background") <= 9
        assert executor.wait_idle(timeout=1)
        assert len([item for item in execution_order if item.startswith("critical-")]) == 60
        blocker.result(timeout=1)
        first.result(timeout=1)
    finally:
        gate.set()
        executor.shutdown()


def test_background_writer_has_bounded_service_under_mixed_replenishment() -> None:
    executor = async_store._PrioritizedWriterExecutor()
    gate = threading.Event()
    started = threading.Event()
    execution_order: list[str] = []

    def _blocker() -> None:
        started.set()
        gate.wait(timeout=2)
        execution_order.append("blocker")

    def _normal(index: int) -> None:
        execution_order.append(f"normal-{index}")
        executor.submit(
            lambda: execution_order.append(f"critical-{index}"),
            priority=ControllerStorePriority.CRITICAL,
        )

    try:
        blocker = executor.submit(_blocker, priority=ControllerStorePriority.CRITICAL)
        assert started.wait(timeout=1)
        normal = [
            executor.submit(
                lambda index=index: _normal(index),
                priority=ControllerStorePriority.NORMAL,
            )
            for index in range(40)
        ]
        background = executor.submit(
            lambda: execution_order.append("background"),
            priority=ControllerStorePriority.BACKGROUND,
        )
        gate.set()

        background.result(timeout=1)
        assert execution_order.index("background") <= 17
        assert executor.wait_idle(timeout=1)
        blocker.result(timeout=1)
        for future in normal:
            future.result(timeout=1)
    finally:
        gate.set()
        executor.shutdown()


@pytest.mark.parametrize("preloaded_critical", [0, 1, 7, 8])
def test_background_writer_fairness_bound_includes_preloaded_and_replenished_critical(
    preloaded_critical: int,
) -> None:
    executor = async_store._PrioritizedWriterExecutor()
    gate = threading.Event()
    started = threading.Event()
    execution_order: list[str] = []

    def _blocker() -> None:
        started.set()
        gate.wait(timeout=2)
        execution_order.append("blocker")

    def _normal(index: int) -> None:
        execution_order.append(f"normal-{index}")
        executor.submit(
            lambda: execution_order.append(f"replenished-critical-{index}"),
            priority=ControllerStorePriority.CRITICAL,
        )

    try:
        blocker = executor.submit(_blocker, priority=ControllerStorePriority.CRITICAL)
        assert started.wait(timeout=1)
        critical = [
            executor.submit(
                lambda index=index: execution_order.append(f"preloaded-critical-{index}"),
                priority=ControllerStorePriority.CRITICAL,
            )
            for index in range(preloaded_critical)
        ]
        normal = [
            executor.submit(
                lambda index=index: _normal(index),
                priority=ControllerStorePriority.NORMAL,
            )
            for index in range(40)
        ]
        background = executor.submit(
            lambda: execution_order.append("background"),
            priority=ControllerStorePriority.BACKGROUND,
        )
        gate.set()

        background.result(timeout=1)
        assert execution_order.index("background") <= 17
        assert executor.wait_idle(timeout=1)
        blocker.result(timeout=1)
        for future in critical:
            future.result(timeout=1)
        for future in normal:
            future.result(timeout=1)
    finally:
        gate.set()
        executor.shutdown()


def test_background_admission_bypasses_normal_waiters(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        previous_executor = async_store._EXECUTOR
        previous_admission = async_store._WRITER_ADMISSION
        executor = async_store._PrioritizedWriterExecutor()
        async_store._EXECUTOR = executor
        async_store._WRITER_ADMISSION = async_store._WriterAdmission(
            maximum=2,
            normal_maximum=1,
            background_maximum=1,
        )
        gate = threading.Event()
        started = threading.Event()
        execution_order: list[str] = []
        blocker: asyncio.Task | None = None
        normals: list[asyncio.Task] = []
        try:

            def _blocker(_store) -> None:
                started.set()
                gate.wait(timeout=2)
                execution_order.append("blocker")

            blocker = asyncio.create_task(
                controller_store_call_profiled(store.path, _blocker, operation_name="admission-blocker")
            )
            await _wait_for_thread_events([started])
            normals = [
                asyncio.create_task(
                    controller_store_call_profiled(
                        store.path,
                        lambda _store, index=index: execution_order.append(f"normal-{index}"),
                        operation_name=f"normal-{index}",
                    )
                )
                for index in range(10)
            ]
            await asyncio.sleep(0.02)
            background = asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    lambda _store: execution_order.append("background"),
                    operation_name="device_heartbeat",
                    priority=ControllerStorePriority.BACKGROUND,
                )
            )
            await asyncio.sleep(0)
            gate.set()

            _value, metrics = await asyncio.wait_for(background, timeout=1)
            await asyncio.gather(blocker, *normals)
            assert execution_order.index("background") <= 3
            assert execution_order.index("background") < execution_order.index("normal-2")
            assert metrics.admission_ms >= 0
        finally:
            gate.set()
            pending = ([blocker] if blocker is not None else []) + normals
            await asyncio.gather(*pending, return_exceptions=True)
            async_store._EXECUTOR = previous_executor
            async_store._WRITER_ADMISSION = previous_admission
            executor.shutdown()
            store.close()

    asyncio.run(_run())


def test_writer_shutdown_drains_queued_work_and_rejects_new_submissions() -> None:
    executor = async_store._PrioritizedWriterExecutor()
    gate = threading.Event()
    started = threading.Event()
    completed: list[int] = []

    def _blocker() -> None:
        started.set()
        gate.wait(timeout=2)

    blocker = executor.submit(_blocker, priority=ControllerStorePriority.NORMAL)
    assert started.wait(timeout=1)
    futures = [
        executor.submit(
            lambda index=index: completed.append(index),
            priority=ControllerStorePriority.NORMAL,
        )
        for index in range(20)
    ]
    shutdown = threading.Thread(target=executor.shutdown)
    shutdown.start()
    try:
        time.sleep(0.02)
        assert shutdown.is_alive()
        gate.set()
        shutdown.join(timeout=1)
        assert not shutdown.is_alive()
        blocker.result(timeout=1)
        assert all(future.done() and not future.cancelled() for future in futures)
        assert sorted(completed) == list(range(20))
        with pytest.raises(RuntimeError, match="shut down"):
            executor.submit(lambda: None, priority=ControllerStorePriority.NORMAL)
    finally:
        gate.set()
        shutdown.join(timeout=1)


def test_server_stop_waits_for_writer_admission_waiters(tmp_path) -> None:
    async def _run() -> None:
        previous_executor = async_store._EXECUTOR
        previous_admission = async_store._WRITER_ADMISSION
        previous_lifecycle = async_store._WRITER_LIFECYCLE
        executor = async_store._PrioritizedWriterExecutor()
        async_store._EXECUTOR = executor
        async_store._WRITER_ADMISSION = async_store._WriterAdmission(
            maximum=1,
            normal_maximum=1,
            background_maximum=1,
        )
        async_store._WRITER_LIFECYCLE = async_store._WriterLifecycle()
        first_gate = threading.Event()
        first_started = threading.Event()
        second_gate = threading.Event()
        second_started = threading.Event()
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        try:
            first = asyncio.create_task(
                controller_store_call_profiled(
                    server.store.path,
                    lambda _store: (first_started.set(), first_gate.wait(timeout=2)),
                    operation_name="shutdown-running",
                )
            )
            await _wait_for_thread_events([first_started])
            second = asyncio.create_task(
                controller_store_call_profiled(
                    server.store.path,
                    lambda _store: (second_started.set(), second_gate.wait(timeout=2)),
                    operation_name="shutdown-admission-waiter",
                )
            )
            await asyncio.sleep(0.02)
            stopping = asyncio.create_task(server.stop())
            await asyncio.sleep(0.02)
            with pytest.raises(RuntimeError, match="draining"):
                await controller_store_call_profiled(
                    server.store.path,
                    lambda _store: None,
                    operation_name="shutdown-late-write",
                )

            first_gate.set()
            await _wait_for_thread_events([second_started])
            await asyncio.sleep(0.02)
            assert not stopping.done()
            second_gate.set()
            await asyncio.gather(first, second)
            await asyncio.wait_for(stopping, timeout=1)
        finally:
            first_gate.set()
            second_gate.set()
            async_store._EXECUTOR = previous_executor
            async_store._WRITER_ADMISSION = previous_admission
            async_store._WRITER_LIFECYCLE = previous_lifecycle
            executor.shutdown()

    asyncio.run(_run())


def test_server_stop_closes_writer_and_store_after_executor_drain_failure(tmp_path) -> None:
    class _FailingExecutor:
        async def drain(self) -> None:
            raise RuntimeError("worker drain failed")

    async def _run() -> None:
        previous_lifecycle = async_store._WRITER_LIFECYCLE
        async_store._WRITER_LIFECYCLE = async_store._WriterLifecycle()
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            ),
            executor=_FailingExecutor(),  # type: ignore[arg-type]
        )
        async_store.controller_store_open(server.store.path)
        try:
            with pytest.raises(RuntimeError, match="worker drain failed"):
                await server.stop()
            with pytest.raises(RuntimeError, match="draining"):
                await controller_store_call_profiled(
                    server.store.path,
                    lambda _store: None,
                    operation_name="shutdown-after-drain-failure",
                )
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                server.store.connection.execute("SELECT 1")
        finally:
            async_store._WRITER_LIFECYCLE = previous_lifecycle

    asyncio.run(_run())


def test_server_stop_closes_writer_and_store_when_cancelled(tmp_path) -> None:
    class _BlockingExecutor:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def drain(self) -> None:
            self.started.set()
            await asyncio.Future()

    async def _run() -> None:
        previous_lifecycle = async_store._WRITER_LIFECYCLE
        async_store._WRITER_LIFECYCLE = async_store._WriterLifecycle()
        executor = _BlockingExecutor()
        writer_gate = threading.Event()
        writer_started = threading.Event()
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            ),
            executor=executor,  # type: ignore[arg-type]
        )
        async_store.controller_store_open(server.store.path)
        try:
            writer = asyncio.create_task(
                controller_store_call_profiled(
                    server.store.path,
                    lambda _store: (writer_started.set(), writer_gate.wait(timeout=2)),
                    operation_name="shutdown-cancel-running-writer",
                )
            )
            await _wait_for_thread_events([writer_started])
            stopping = asyncio.create_task(server.stop())
            await asyncio.wait_for(executor.started.wait(), timeout=1)
            stopping.cancel()
            await asyncio.sleep(0.02)
            assert not stopping.done()
            with pytest.raises(RuntimeError, match="draining"):
                await controller_store_call_profiled(
                    server.store.path,
                    lambda _store: None,
                    operation_name="shutdown-after-cancellation",
                )
            writer_gate.set()
            await writer
            with pytest.raises(asyncio.CancelledError):
                await stopping
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                server.store.connection.execute("SELECT 1")
        finally:
            writer_gate.set()
            async_store._WRITER_LIFECYCLE = previous_lifecycle

    asyncio.run(_run())


def test_server_stop_preserves_external_cancellation_during_telemetry_cleanup(tmp_path) -> None:
    async def _run() -> None:
        previous_lifecycle = async_store._WRITER_LIFECYCLE
        async_store._WRITER_LIFECYCLE = async_store._WriterLifecycle()
        telemetry_started = asyncio.Event()
        telemetry_cleanup_started = asyncio.Event()
        allow_telemetry_cleanup = asyncio.Event()

        async def _telemetry() -> None:
            telemetry_started.set()
            try:
                await asyncio.Future()
            finally:
                telemetry_cleanup_started.set()
                await allow_telemetry_cleanup.wait()

        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        async_store.controller_store_open(server.store.path)
        server._runtime_telemetry_task = asyncio.create_task(_telemetry())  # noqa: SLF001
        try:
            await asyncio.wait_for(telemetry_started.wait(), timeout=1)
            stopping = asyncio.create_task(server.stop())
            await asyncio.wait_for(telemetry_cleanup_started.wait(), timeout=1)
            stopping.cancel()
            await asyncio.sleep(0.02)
            assert not stopping.done()
            allow_telemetry_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await stopping
            with pytest.raises(RuntimeError, match="draining"):
                await controller_store_call_profiled(
                    server.store.path,
                    lambda _store: None,
                    operation_name="shutdown-after-telemetry-cancellation",
                )
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                server.store.connection.execute("SELECT 1")
        finally:
            allow_telemetry_cleanup.set()
            async_store._WRITER_LIFECYCLE = previous_lifecycle

    asyncio.run(_run())


def test_cancel_while_waiting_for_writer_admission_releases_load(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        gate = threading.Event()
        started = threading.Event()

        def _blocker(_store) -> None:
            started.set()
            gate.wait(timeout=2)

        blocker = asyncio.create_task(
            controller_store_call_profiled(store.path, _blocker, operation_name="admission-blocker")
        )
        await _wait_for_thread_events([started])
        admitted = [
            asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    lambda _store: None,
                    operation_name=f"admitted-{index}",
                )
            )
            for index in range(95)
        ]
        waiting = asyncio.create_task(
            controller_store_call_profiled(
                store.path,
                lambda _store: None,
                operation_name="cancelled-before-admission",
            )
        )
        await asyncio.sleep(0.02)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        gate.set()
        await asyncio.gather(blocker, *admitted)
        _value, metrics = await controller_store_call_profiled(
            store.path,
            lambda _store: None,
            operation_name="after-admission-cancel",
        )
        assert metrics.queue_depth == 0
        assert metrics.admitted_queue_depth == 0
        store.close()

    asyncio.run(_run())


def test_cancelled_queued_calls_do_not_leak_queue_depth(tmp_path) -> None:
    async def _run() -> None:
        store = _active_store(tmp_path)
        try:
            writer_gate = threading.Event()
            writer_started = threading.Event()

            def _block_writer(_store) -> None:
                writer_started.set()
                writer_gate.wait(timeout=2)

            active_writer = asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    _block_writer,
                    operation_name="writer-blocker",
                )
            )
            await _wait_for_thread_events([writer_started])
            cancelled_writer = asyncio.create_task(
                controller_store_call_profiled(
                    store.path,
                    lambda _store: None,
                    operation_name="writer-cancelled",
                )
            )
            await asyncio.sleep(0)
            cancelled_writer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_writer
            writer_gate.set()
            await active_writer
            _result, writer_metrics = await controller_store_call_profiled(
                store.path,
                lambda _store: None,
                operation_name="writer-after-cancel",
            )
            assert writer_metrics.queue_depth == 0

            reader_gate = threading.Event()
            reader_started = [threading.Event() for _ in range(4)]

            def _block_reader(index: int):
                def _operation(_store) -> None:
                    reader_started[index].set()
                    reader_gate.wait(timeout=2)

                return _operation

            active_readers = [
                asyncio.create_task(
                    controller_store_read_profiled(
                        store.path,
                        _block_reader(index),
                        operation_name=f"reader-blocker-{index}",
                    )
                )
                for index in range(4)
            ]
            await _wait_for_thread_events(reader_started)
            cancelled_reader = asyncio.create_task(
                controller_store_read_profiled(
                    store.path,
                    lambda _store: None,
                    operation_name="reader-cancelled",
                )
            )
            await asyncio.sleep(0)
            cancelled_reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_reader
            reader_gate.set()
            await asyncio.gather(*active_readers)
            _result, reader_metrics = await controller_store_read_profiled(
                store.path,
                lambda reader: reader.session_row("session-fast"),
                operation_name="reader-after-cancel",
            )
            assert reader_metrics.queue_depth == 0
        finally:
            store.close()

    asyncio.run(_run())


def _command_statements(statements: list[str]) -> list[str]:
    return [statement for statement in statements if "COMMANDS" in statement.upper()]


def _data_statements(statements: list[str]) -> list[str]:
    prefixes = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")
    return [statement for statement in statements if statement.lstrip().upper().startswith(prefixes)]


async def _wait_for_thread_events(events: list[threading.Event]) -> None:
    for _ in range(200):
        if all(event.is_set() for event in events):
            return
        await asyncio.sleep(0.005)
    raise AssertionError("executor workers did not enter blocking operations")


async def _wait_for_writer_queue(
    executor: async_store._PrioritizedWriterExecutor,
    priority: ControllerStorePriority,
) -> None:
    for _ in range(200):
        with executor._condition:  # noqa: SLF001 - deterministic scheduler regression probe.
            if executor._queues[priority]:  # noqa: SLF001 - deterministic scheduler regression probe.
                return
        await asyncio.sleep(0.005)
    raise AssertionError(f"writer did not queue {priority.name} work")
