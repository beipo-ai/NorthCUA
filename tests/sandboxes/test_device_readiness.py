"""Device readiness probe regression tests."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from northcua.sandboxes.contracts import SandboxReadinessState
from northcua.sandboxes.device.readiness import ControllerSQLiteDeviceReadinessProbe
from northcua_protocol.state import DeviceState


def _create_controller_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE devices (
                device_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                descriptor_json TEXT NOT NULL DEFAULT '{}',
                last_heartbeat_at TEXT,
                active_sessions INTEGER NOT NULL DEFAULT 0,
                load_json TEXT NOT NULL DEFAULT '{}'
            )
            """)
        connection.execute(
            "INSERT INTO devices(device_id, status) VALUES (?, ?)",
            ("ready-device", DeviceState.ONLINE.value),
        )


@pytest.mark.asyncio
async def test_controller_sqlite_readiness_returns_per_call_diagnostics(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    _create_controller_db(db_path)
    probe = ControllerSQLiteDeviceReadinessProbe(
        db_path,
        poll_interval_seconds=0.001,
        max_poll_interval_seconds=0.001,
    )

    ready_result, missing_result = await asyncio.gather(
        probe.wait_for_ready("ready-device", timeout_seconds=0.05),
        probe.wait_for_ready("missing-device", timeout_seconds=0.01),
    )

    assert ready_result.state == SandboxReadinessState.READY
    assert ready_result.probe_count == 1
    assert ready_result.last_status == DeviceState.ONLINE.value
    assert ready_result.last_error == ""
    assert missing_result.state == SandboxReadinessState.TIMED_OUT
    assert missing_result.probe_count >= 1
    assert missing_result.last_status == "missing"
    assert missing_result.last_error == ""
