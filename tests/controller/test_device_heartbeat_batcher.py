"""Device heartbeat persistence is coalesced and lifecycle safe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.control_plane.registry import DeviceRegistry
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.device_heartbeat_batcher import DeviceHeartbeatBatcher
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.payloads import DeviceHeartbeat
from northcua_protocol.state import DeviceState


def _descriptor(device_id: str) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.LINUX,
        capabilities={Capability.DESKTOP_INPUT},
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
    )


@pytest.mark.asyncio
async def test_device_heartbeat_batcher_coalesces_latest_value(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(db_path)
    store.upsert_device(_descriptor("dev-1"), status=DeviceState.ONLINE, active_sessions=7)
    calls = 0

    async def _call(path: str | Path, operation, **_kwargs):
        nonlocal calls
        calls += 1
        worker_store = SQLiteControllerStore(path)
        try:
            return operation(worker_store)
        finally:
            worker_store.close()

    monkeypatch.setattr("northcua.controller.storage.device_heartbeat_batcher.controller_store_call", _call)
    batcher = DeviceHeartbeatBatcher(db_path, flush_interval_seconds=60)
    for step in range(500):
        batcher.publish(
            device_id="dev-1",
            last_heartbeat_at=f"2099-08-16T00:00:{step % 60:02d}+00:00",
            load_json=json.dumps({"step": step}),
        )
    await batcher.flush()

    row = next(row for row in store.device_rows() if row["device_id"] == "dev-1")
    assert calls == 1
    assert row["status"] == DeviceState.ONLINE.value
    assert row["active_sessions"] == 7
    assert json.loads(row["load_json"])["step"] == 499
    assert batcher.snapshot() == {
        "pending_devices": 0,
        "published": 500,
        "persisted": 1,
        "coalesced": 499,
        "failures": 0,
        "dropped": 0,
        "skipped": 0,
        "queue_age_ms_max": batcher.snapshot()["queue_age_ms_max"],
        "in_flight_publishes": 0,
        "max_batch_size": 1024,
        "max_write_batch_size": 128,
    }
    await batcher.aclose()
    store.close()


@pytest.mark.asyncio
async def test_device_heartbeat_batcher_bounds_each_sqlite_transaction(tmp_path: Path, monkeypatch) -> None:
    write_sizes: list[int] = []

    class _RecordingStore:
        def heartbeat_devices_batch(self, rows: list[tuple[str, str, str]]) -> int:
            write_sizes.append(len(rows))
            return len(rows)

    async def _call(_path: str | Path, operation, **_kwargs):
        return operation(_RecordingStore())

    monkeypatch.setattr("northcua.controller.storage.device_heartbeat_batcher.controller_store_call", _call)
    batcher = DeviceHeartbeatBatcher(
        tmp_path / "controller.sqlite",
        flush_interval_seconds=60,
        max_batch_size=1024,
        max_write_batch_size=128,
    )
    for index in range(300):
        batcher.publish(
            device_id=f"dev-{index}",
            last_heartbeat_at="2026-08-16T00:00:00+00:00",
            load_json="{}",
        )

    await batcher.flush()

    assert write_sizes == [128, 128, 44]
    assert batcher.snapshot()["persisted"] == 300
    assert batcher.snapshot()["pending_devices"] == 0
    await batcher.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", [DeviceState.DRAINING, DeviceState.OFFLINE])
async def test_late_device_heartbeat_cannot_revive_lifecycle_state(
    tmp_path: Path,
    terminal_state: DeviceState,
) -> None:
    db_path = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(db_path)
    store.upsert_device(_descriptor("dev-race"), status=DeviceState.ONLINE, active_sessions=5)
    batcher = DeviceHeartbeatBatcher(db_path, flush_interval_seconds=60)
    batcher.publish(
        device_id="dev-race",
        last_heartbeat_at="2026-08-16T00:00:00+00:00",
        load_json=json.dumps({"load": 0.7}),
    )
    store.update_device_status("dev-race", terminal_state)

    await batcher.flush()

    row = next(row for row in store.device_rows() if row["device_id"] == "dev-race")
    assert row["status"] == terminal_state.value
    assert row["active_sessions"] == 5
    assert json.loads(row["load_json"]) == {}
    assert batcher.snapshot()["persisted"] == 0
    assert batcher.snapshot()["skipped"] == 1
    await batcher.aclose()
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", [DeviceState.DRAINING, DeviceState.OFFLINE])
async def test_async_heartbeat_cannot_revive_terminal_device(
    tmp_path: Path,
    terminal_state: DeviceState,
) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    registry = DeviceRegistry(
        store=store,
        events=ControllerEventLog(store),
        heartbeat_persist_interval_seconds=0,
    )
    await registry.register_async(_descriptor("dev-draining"), trace_id="trace-register")
    if terminal_state == DeviceState.DRAINING:
        await registry.mark_draining_async("dev-draining", trace_id="trace-terminal")
    else:
        await registry.mark_offline_async("dev-draining", trace_id="trace-terminal")

    await registry.heartbeat_async(
        DeviceHeartbeat(device_id="dev-draining", active_sessions=1),
        trace_id="trace-heartbeat",
    )
    await registry._heartbeat_batcher.flush()  # noqa: SLF001 - lifecycle race verification

    device = registry.get("dev-draining")
    assert device is not None
    assert device.state == terminal_state
    assert registry.available() == []
    row = next(row for row in store.device_rows() if row["device_id"] == "dev-draining")
    assert row["status"] == terminal_state.value
    await registry.aclose()
    registry.events.close()
    store.close()


@pytest.mark.parametrize("terminal_state", [DeviceState.DRAINING, DeviceState.OFFLINE])
def test_sync_heartbeat_cannot_revive_terminal_device(tmp_path: Path, terminal_state: DeviceState) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    registry = DeviceRegistry(
        store=store,
        events=ControllerEventLog(store),
        heartbeat_persist_interval_seconds=0,
    )
    registry.register(_descriptor("dev-sync-terminal"), trace_id="trace-register")
    if terminal_state == DeviceState.DRAINING:
        registry.mark_draining("dev-sync-terminal", trace_id="trace-terminal")
    else:
        registry.mark_offline("dev-sync-terminal", trace_id="trace-terminal")

    assert not registry.heartbeat(
        DeviceHeartbeat(device_id="dev-sync-terminal", active_sessions=1),
        trace_id="trace-heartbeat",
    )

    device = registry.get("dev-sync-terminal")
    assert device is not None
    assert device.state == terminal_state
    assert registry.available() == []
    row = next(row for row in store.device_rows() if row["device_id"] == "dev-sync-terminal")
    assert row["status"] == terminal_state.value
    registry.events.close()
    store.close()


@pytest.mark.asyncio
async def test_stale_batch_cannot_overwrite_new_registration(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(db_path)
    store.upsert_device(
        _descriptor("dev-reconnect"),
        status=DeviceState.ONLINE,
        active_sessions=1,
        load_json=json.dumps({"generation": "old"}),
    )
    batcher = DeviceHeartbeatBatcher(db_path, flush_interval_seconds=60)
    batcher.publish(
        device_id="dev-reconnect",
        last_heartbeat_at="2026-08-16T00:00:00+00:00",
        load_json=json.dumps({"generation": "stale"}),
    )
    store.upsert_device(
        _descriptor("dev-reconnect"),
        status=DeviceState.ONLINE,
        active_sessions=9,
        load_json=json.dumps({"generation": "new"}),
    )

    await batcher.flush()

    row = next(row for row in store.device_rows() if row["device_id"] == "dev-reconnect")
    assert row["active_sessions"] == 9
    assert json.loads(row["load_json"])["generation"] == "new"
    assert batcher.snapshot()["persisted"] == 0
    assert batcher.snapshot()["skipped"] == 1
    await batcher.aclose()
    store.close()


@pytest.mark.asyncio
async def test_permanent_store_failure_is_bounded(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    async def _failed_call(_path: str | Path, _operation, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError("persistent sqlite failure")

    monkeypatch.setattr("northcua.controller.storage.device_heartbeat_batcher.controller_store_call", _failed_call)
    batcher = DeviceHeartbeatBatcher(
        tmp_path / "controller.sqlite",
        flush_interval_seconds=0,
        max_retry_attempts=2,
        retry_backoff_seconds=0,
    )
    batcher.publish(
        device_id="dev-failed",
        last_heartbeat_at="2026-08-16T00:00:00+00:00",
        load_json="{}",
    )
    task = batcher._flush_task  # noqa: SLF001 - focused retry lifecycle assertion
    assert task is not None
    await task

    assert calls == 2
    assert batcher.snapshot()["failures"] == 2
    assert batcher.snapshot()["dropped"] == 1
    assert batcher.snapshot()["in_flight_publishes"] == 0
    assert batcher.snapshot()["pending_devices"] == 0
    await batcher.aclose()


@pytest.mark.asyncio
async def test_publish_and_close_are_bounded_while_store_is_blocked(tmp_path: Path, monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_call(_path: str | Path, _operation, **_kwargs):
        started.set()
        await release.wait()
        return 0

    monkeypatch.setattr("northcua.controller.storage.device_heartbeat_batcher.controller_store_call", _blocked_call)
    batcher = DeviceHeartbeatBatcher(
        tmp_path / "controller.sqlite",
        flush_interval_seconds=0,
        flush_timeout_seconds=0.01,
    )
    batcher.publish(
        device_id="dev-blocked",
        last_heartbeat_at="2026-08-16T00:00:00+00:00",
        load_json="{}",
    )
    await started.wait()

    await asyncio.wait_for(batcher.aclose(), timeout=0.1)
    batcher.publish(
        device_id="dev-too-late",
        last_heartbeat_at="2026-08-16T00:00:01+00:00",
        load_json="{}",
    )
    release.set()

    assert batcher.snapshot()["pending_devices"] == 0
    assert batcher.snapshot()["dropped"] == 2
    assert batcher.snapshot()["in_flight_publishes"] == 0


@pytest.mark.asyncio
async def test_cancelled_server_stop_waits_for_heartbeat_close(tmp_path: Path, monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_call(_path: str | Path, _operation, **_kwargs):
        started.set()
        await release.wait()
        return 1

    monkeypatch.setenv("NORTHCUA_CONTROLLER_DEVICE_HEARTBEAT_FLUSH_INTERVAL_SECONDS", "0")
    monkeypatch.setattr("northcua.controller.storage.device_heartbeat_batcher.controller_store_call", _blocked_call)
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            heartbeat_persist_interval_seconds=0,
        )
    )
    await server.registry.register_async(_descriptor("dev-stop"), trace_id="trace-register")
    await server.registry.heartbeat_async(
        DeviceHeartbeat(device_id="dev-stop", active_sessions=1),
        trace_id="trace-heartbeat",
    )
    await started.wait()

    stop_task = asyncio.create_task(server.stop())
    await asyncio.sleep(0)
    stop_task.cancel()
    await asyncio.sleep(0)
    assert not stop_task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stop_task, timeout=2)
