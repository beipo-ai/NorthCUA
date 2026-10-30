from __future__ import annotations

import asyncio
import json
import time

import pytest

from northcua.batch.runner.event_loop_watchdog import RunnerEventLoopWatchdog


@pytest.mark.asyncio
async def test_runner_event_loop_watchdog_records_blocked_loop_stack(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_RUNNER_LOOP_LAG_THRESHOLD_SECONDS", "0.02")
    monkeypatch.setenv("NORTHCUA_BATCH_RUNNER_LOOP_LAG_CHECK_INTERVAL_SECONDS", "0.01")
    monkeypatch.setenv("NORTHCUA_BATCH_RUNNER_LOOP_LAG_TICK_INTERVAL_SECONDS", "0.01")
    monkeypatch.setenv("NORTHCUA_BATCH_RUNNER_LOOP_LAG_LOG_INTERVAL_SECONDS", "0")

    async with RunnerEventLoopWatchdog(output_dir=tmp_path, batch_id="batch-watchdog"):
        await asyncio.sleep(0.02)
        time.sleep(0.2)
        path = tmp_path / "batch_runner_runtime_events.jsonl"
        for _ in range(20):
            if path.exists():
                break
            await asyncio.sleep(0.02)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[-1]["batch_id"] == "batch-watchdog"
    assert rows[-1]["message"] == "runner_event_loop_lag"
    assert rows[-1]["lag_seconds"] >= 0.02
    assert any("test_runner_event_loop_watchdog_records_blocked_loop_stack" in line for line in rows[-1]["stack"])


@pytest.mark.asyncio
async def test_runner_event_loop_watchdog_can_be_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_RUNNER_LOOP_LAG_THRESHOLD_SECONDS", "0")

    async with RunnerEventLoopWatchdog(output_dir=tmp_path, batch_id="batch-watchdog"):
        time.sleep(0.03)
        await asyncio.sleep(0.01)

    assert not (tmp_path / "batch_runner_runtime_events.jsonl").exists()
