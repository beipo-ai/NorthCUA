"""Tests for device-side telemetry stream reduction."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from northcua_device.logging import DeviceEventLogger
from northcua_device.telemetry import DeviceTelemetrySink


class _TelemetryConnection:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, envelope: object) -> None:
        self.sent.append(envelope)


async def _drain_sink(sink: DeviceTelemetrySink) -> None:
    queue = sink._queue  # type: ignore[attr-defined]
    if queue is not None:
        await queue.join()


def _event_names(path: Path) -> list[str]:
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


def _streamed_event_names(connection: _TelemetryConnection) -> list[str]:
    return [getattr(envelope.payload, "event_type") for envelope in connection.sent]


def test_terminal_sampled_zero_keeps_local_logs_but_streams_terminal_only(tmp_path) -> None:
    async def _run() -> None:
        connection = _TelemetryConnection()
        sink = DeviceTelemetrySink(
            device_id="dev-1",
            local_events=DeviceEventLogger(tmp_path),
            stream_mode="terminal_sampled",
            sample_rate=0.0,
        )
        await sink.start(connection=connection, trace_id="trace-1")  # type: ignore[arg-type]

        sink.write("device_command_received", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("device_driver_call_started", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("command_result_ready", command_id="cmd-1", session_id="sess", tool_id="click", status="succeeded")
        sink.write("command_result_sent", command_id="cmd-1", session_id="sess", tool_id="click", status="succeeded")
        await _drain_sink(sink)
        await sink.stop()

        assert _streamed_event_names(connection) == [
            "device_command_received",
            "command_result_sent",
        ]
        local_events = _event_names(tmp_path / "device_runtime.jsonl")
        assert "device_command_received" in local_events
        assert "device_driver_call_started" in local_events
        assert "command_result_ready" in local_events
        assert "command_result_sent" in local_events

    asyncio.run(_run())


def test_terminal_sampled_streams_minimal_lifecycle_for_unsampled_command(tmp_path) -> None:
    async def _run() -> None:
        connection = _TelemetryConnection()
        sink = DeviceTelemetrySink(
            device_id="dev-1",
            local_events=DeviceEventLogger(tmp_path),
            stream_mode="terminal_sampled",
            sample_rate=0.0,
        )
        await sink.start(connection=connection, trace_id="trace-1")  # type: ignore[arg-type]

        sink.write("device_command_received", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("device_command_dequeued", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("command_started", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("device_driver_call_started", command_id="cmd-1", session_id="sess", tool_id="click")
        await _drain_sink(sink)
        await sink.stop()

        assert _streamed_event_names(connection) == ["device_command_received", "command_started"]

    asyncio.run(_run())


def test_terminal_sampled_streams_all_events_for_sampled_command(tmp_path) -> None:
    async def _run() -> None:
        connection = _TelemetryConnection()
        sink = DeviceTelemetrySink(
            device_id="dev-1",
            local_events=DeviceEventLogger(tmp_path),
            stream_mode="terminal_sampled",
            sample_rate=1.0,
        )
        await sink.start(connection=connection, trace_id="trace-1")  # type: ignore[arg-type]

        sink.write("device_command_received", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("device_driver_call_started", command_id="cmd-1", session_id="sess", tool_id="click")
        sink.write("command_result_ready", command_id="cmd-1", session_id="sess", tool_id="click", status="succeeded")
        sink.write("command_result_sent", command_id="cmd-1", session_id="sess", tool_id="click", status="succeeded")
        await _drain_sink(sink)
        await sink.stop()

        assert _streamed_event_names(connection) == [
            "device_command_received",
            "device_driver_call_started",
            "command_result_ready",
            "command_result_sent",
        ]

    asyncio.run(_run())


def test_terminal_sampled_always_streams_failures_even_when_not_sampled(tmp_path) -> None:
    async def _run() -> None:
        connection = _TelemetryConnection()
        sink = DeviceTelemetrySink(
            device_id="dev-1",
            local_events=DeviceEventLogger(tmp_path),
            stream_mode="terminal_sampled",
            sample_rate=0.0,
        )
        await sink.start(connection=connection, trace_id="trace-1")  # type: ignore[arg-type]

        sink.write("device_driver_call_finished", command_id="cmd-1", tool_id="click", status="failed", error="boom")
        sink.write("command_result_sent", command_id="cmd-1", tool_id="click", status="failed", error="boom")
        await _drain_sink(sink)
        await sink.stop()

        assert _streamed_event_names(connection) == ["device_driver_call_finished", "command_result_sent"]

    asyncio.run(_run())


def test_raw_and_off_modes_are_explicit_escape_hatches(tmp_path) -> None:
    async def _run() -> None:
        raw_connection = _TelemetryConnection()
        raw_sink = DeviceTelemetrySink(
            device_id="dev-1",
            local_events=DeviceEventLogger(tmp_path / "raw"),
            stream_mode="raw",
            sample_rate=0.0,
        )
        await raw_sink.start(connection=raw_connection, trace_id="trace-1")  # type: ignore[arg-type]
        raw_sink.write("device_command_received", command_id="cmd-1")
        await _drain_sink(raw_sink)
        await raw_sink.stop()
        assert _streamed_event_names(raw_connection) == ["device_command_received"]

        off_connection = _TelemetryConnection()
        off_sink = DeviceTelemetrySink(
            device_id="dev-1",
            local_events=DeviceEventLogger(tmp_path / "off"),
            stream_mode="off",
            sample_rate=1.0,
        )
        await off_sink.start(connection=off_connection, trace_id="trace-1")  # type: ignore[arg-type]
        off_sink.write("command_result_sent", command_id="cmd-1", status="failed", error="boom")
        await _drain_sink(off_sink)
        await off_sink.stop()
        assert _streamed_event_names(off_connection) == []

    asyncio.run(_run())
