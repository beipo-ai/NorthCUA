from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from northcua.diagnostics.transport import strict_violations, summarize


def test_summarize_device_transport_reads_wal_and_reports_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "case"
    run_dir.mkdir()
    database = run_dir / "controller.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE events (entity_id TEXT, event_type TEXT, created_at TEXT)")
    connection.execute(
        "CREATE TABLE commands (status TEXT, protocol_command_id TEXT, request_json TEXT, result_json TEXT)"
    )
    connection.commit()

    connection.executemany(
        "INSERT INTO events VALUES (?, 'command_result_envelope_received', ?)",
        [
            ("command-ok", "2026-07-16T00:00:00.100000+00:00"),
            ("command-failed", "2026-07-16T00:00:01.200000+00:00"),
        ],
    )
    connection.executemany(
        "INSERT INTO commands VALUES (?, ?, ?, ?)",
        [
            (
                "succeeded",
                "command-ok",
                json.dumps({"tool_id": "click"}),
                json.dumps({"status": "succeeded", "result": {"ok": True}}),
            ),
            (
                "failed",
                "command-failed",
                json.dumps({"tool_id": "shell.exec_readonly"}),
                json.dumps(
                    {
                        "status": "failed",
                        "result": {"returncode": 2},
                        "error": "shell command failed",
                        "metadata": {"tool_stage_timings": {"backend_action_ms": 1}},
                    }
                ),
            ),
        ],
    )
    connection.commit()
    assert database.with_name("controller.sqlite-wal").is_file()

    runtime_log = run_dir / "batch" / "case-device_runtime.jsonl"
    runtime_log.parent.mkdir()
    events = []
    for command_id in ("command-ok", "command-failed"):
        for event_name in (
            "device_command_received",
            "device_command_dequeued",
            "device_command_prepare_started",
            "device_command_prepare_finished",
            "device_driver_call_started",
            "device_driver_call_finished",
            "device_result_json_serialize_started",
            "device_result_json_serialize_finished",
            "device_websocket_send_started",
            "device_websocket_send_finished",
        ):
            events.append(
                {
                    "timestamp": "2026-07-16T00:00:00+00:00",
                    "event": event_name,
                    "command_id": command_id,
                    "queue_depth_before": 0,
                    "queue_wait_ms": 3,
                }
            )
    events.extend(
        [
            {
                "timestamp": "2026-07-16T00:00:00+00:00",
                "event": "command_result_sent",
                "command_id": "command-ok",
                "send_elapsed_ms": 1,
                "envelope_bytes": 1024,
            },
            {
                "timestamp": "2026-07-16T00:00:01+00:00",
                "event": "command_result_sent",
                "command_id": "command-failed",
                "send_elapsed_ms": 2,
                "envelope_bytes": 2048,
            },
            {"timestamp": "2026-07-16T00:00:02+00:00", "event": "device_telemetry_dropped"},
            {"timestamp": "2026-07-16T00:00:03+00:00", "event": "connection_failed"},
        ]
    )
    runtime_log.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    try:
        result = summarize(tmp_path)
    finally:
        connection.close()

    assert result["commands"] == {"failed": 1, "succeeded": 1}
    assert result["failure_classes"] == {"shell_nonzero_rc_2": 1}
    assert result["result_telemetry_leaks"] == 1
    assert result["transport_failure_events"] == {"device_telemetry_dropped": 1}
    assert result["connection_events"] == {"connection_failed": 1}
    assert result["result_delivery_coverage"] == {
        "device_sends": 2,
        "controller_receives": 2,
        "matched_sends": 2,
        "unmatched_sends": 0,
        "unmatched_receives": 0,
        "duplicate_send_ids": 0,
        "duplicate_receive_ids": 0,
    }
    assert result["command_table_coverage"] == {
        "protocol_commands": 2,
        "duplicate_command_ids": 0,
        "missing_controller_results": 0,
        "unexpected_controller_results": 0,
    }
    assert all(coverage["missing"] == 0 for coverage in result["device_lifecycle_coverage"].values())
    assert result["send_to_controller_ingress_ms"] == {
        "count": 2,
        "min": 100.0,
        "p50": 100.0,
        "p95": 200.0,
        "max": 200.0,
    }
    violations = strict_violations(result, expected_databases=1)
    assert "device_telemetry_dropped=1" in violations
    assert "result_telemetry_leaks=1" in violations


def test_strict_transport_tolerates_small_cross_host_clock_skew() -> None:
    payload = {
        "agent_trace_files": 1,
        "controller_databases": 1,
        "commands": {"succeeded": 1},
        "send_to_controller_ingress_ms": {"count": 1, "min": -35.6},
    }

    assert strict_violations(payload, expected_databases=1) == []


def test_strict_transport_rejects_implausible_negative_ingress_latency() -> None:
    payload = {
        "agent_trace_files": 1,
        "controller_databases": 1,
        "commands": {"succeeded": 1},
        "send_to_controller_ingress_ms": {"count": 1, "min": -1_001.0},
    }

    assert strict_violations(payload, expected_databases=1) == [
        "send_to_controller_ingress_ms.min=-1001.0, tolerance_ms=1000"
    ]


def test_strict_transport_requires_agent_trace_evidence() -> None:
    payload = {
        "controller_databases": 1,
        "commands": {"succeeded": 1},
        "send_to_controller_ingress_ms": {"count": 1, "min": 0.0},
    }

    assert strict_violations(payload, expected_databases=1) == ["agent_trace_files=0"]
