from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import cast

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.maintenance.batch_health_report import generate_batch_health_report
from northcua.maintenance.controller_event_metrics import process_executor_metrics, readiness_stage_metrics


def test_health_report_returns_stable_operator_metrics(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir()
    batch_db = batch_dir / "batch_state.sqlite"
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="do two"),
        ],
    )
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.TIMED_OUT,
            attempts=1,
            controller_task_id="controller-new",
            error="case-level controller timeout after 2400s",
        )
    )
    repository.upsert_state(
        BatchTaskState(
            task_id="case-2",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-current",
            session_id="session-current",
            device_id="device-current",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    _controller_db(controller_db)
    _insert_controller_row(
        controller_db, "controller-current", "case-2", "session-current", "device-current", "screenshot"
    )
    _insert_controller_row(controller_db, "controller-old", "case-1", "session-old", "device-old", "screenshot")
    command_start = time.time() - 30
    _insert_command(
        controller_db,
        command_id="cmd-click-recent",
        session_id="session-current",
        device_id="device-current",
        status="succeeded",
        request_json={"tool_id": "click", "parameters": {"x": 1, "y": 2}},
        created_at=command_start,
        updated_at=command_start + 12,
    )
    _insert_command(
        controller_db,
        command_id="cmd-evaluator-client_bash_readonly-failed",
        session_id="session-current",
        device_id="device-current",
        status="failed",
        request_json={"tool_id": "shell.exec", "parameters": {"cmd": "ls"}},
        created_at=command_start + 1,
        updated_at=command_start + 4,
        result_json={"error": "readonly shell probe failed"},
    )
    _prepared_sandbox_table(batch_db)
    _insert_prepared_sandbox(
        batch_db,
        task_id="case-2",
        status="assigned",
        now=command_start,
        expires_at=command_start + 900,
        assigned_at=command_start - 8,
        prepare_seconds=18.0,
    )
    _insert_prepared_sandbox(
        batch_db,
        task_id="case-prepared",
        status="prepared",
        now=command_start,
        expires_at=command_start + 900,
        prepared_at=command_start - 60,
        prepare_seconds=22.0,
    )
    _insert_prepared_sandbox(
        batch_db,
        task_id="case-preparing",
        status="preparing",
        now=command_start,
        expires_at=command_start + 900,
        prepare_started_at=command_start - 45,
    )
    _insert_prepared_sandbox(
        batch_db,
        task_id="case-stale-assigned",
        status="assigned",
        now=command_start,
        expires_at=command_start + 900,
        assigned_at=command_start - 20,
        prepare_seconds=12.0,
    )
    _insert_prepared_sandbox(
        batch_db,
        task_id="case-expired-active",
        status="prepared",
        now=command_start,
        expires_at=command_start - 1,
        prepared_at=command_start - 120,
        prepare_seconds=16.0,
        error="expired while still prepared",
    )
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO telemetry_events(entity_kind, entity_id, event_type, phase, status, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "llm",
                "model-1",
                "llm_fallback_attempt_failed",
                "main_agent",
                "error",
                json.dumps({"error": "model timeout", "fallback": True}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO telemetry_events(entity_kind, entity_id, event_type, phase, status, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "command",
                "probe-1",
                "command_result_received",
                "tool",
                "succeeded",
                json.dumps({"tool_id": "capture_accessibility_tree", "elapsed_ms": 2400}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO telemetry_events(entity_kind, entity_id, event_type, phase, status, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "task",
                "case-timeout",
                "failure_recorded",
                "controller_submit",
                "failed",
                json.dumps({"error": "case-level controller timeout after 12000s"}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-runtime",
                "controller",
                "127.0.0.1:9192",
                "controller_runtime_telemetry",
                json.dumps({"event_loop_lag_ms": 125, "recv_q_max": 7, "recv_q_p95": 3, "socket_count": 5}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-websocket",
                "connection",
                "conn-1",
                "websocket_handler_started",
                json.dumps(
                    {
                        "websocket_open_handshake_ms": 42,
                        "websocket_handler_start_delay_ms": 45,
                        "websocket_response_status_code": 101,
                        "remote_address": "127.0.0.1:12345",
                    }
                ),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-first",
                "connection",
                "conn-1",
                "websocket_first_envelope_received",
                json.dumps({"first_envelope_ms": 63, "message_type": "device.hello"}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-stage",
                "device",
                "device-current",
                "device_readiness_stage",
                json.dumps(
                    {
                        "stage": "websocket_open",
                        "device_id": "device-current",
                        "websocket_open_handshake_ms": 42,
                        "first_envelope_ms": 63,
                    }
                ),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO telemetry_events(entity_kind, entity_id, event_type, phase, status, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "sandbox",
                "sandbox-current",
                "status_changed",
                "prepared_sandbox",
                "bootstrapping",
                json.dumps(
                    {
                        "message": "sandbox_stage_timing",
                        "details": {
                            "stage": "device_readiness_wait",
                            "duration_ms": 123,
                            "status": "succeeded",
                            "device_id": "device-current",
                            "sandbox_id": "sandbox-current",
                        },
                    }
                ),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-worker",
                "worker_job",
                "job-1",
                "process_executor_metric",
                json.dumps({"metric_name": "worker_connect", "value_ms": 44}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-worker",
                "worker_job",
                "job-1",
                "process_executor_metric",
                json.dumps({"metric_name": "worker_ready", "value_ms": 88}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-worker",
                "worker_job",
                "job-1",
                "process_executor_metric",
                json.dumps({"metric_name": "worker_command_roundtrip", "value_ms": 12, "metadata": {"count": 3}}),
                time.time(),
            ),
        )
        connection.execute(
            "INSERT INTO worker_jobs(job_id, task_id, attempt_id, session_id, lease_id, device_id, agent_kind, status, worker_pid, failure_kind, error, artifact_manifest_json, heartbeat_payload_json, created_at, updated_at, heartbeat_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "job-1",
                "case-2",
                "case-2:1",
                "session-current",
                "lease-session-current",
                "device-current",
                "desktop",
                "running",
                1234,
                "",
                "",
                "{}",
                json.dumps({"rss_bytes": 2048, "cpu_time_seconds": 1.5}),
                time.time(),
                time.time(),
                time.time(),
            ),
        )
    (batch_dir / "batch_attempt_events.jsonl").write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "message": "sandbox device did not become ready within timeout",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (batch_dir / "telemetry_events.jsonl").write_text(
        json.dumps(
            {
                "created_at": time.time(),
                "payload_json": json.dumps({"artifact": {"mime_type": "image/jpeg"}}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    now = time.time()
    _write_jsonl(
        batch_dir / "sandbox-logs" / "direct" / "case-never" / "sandbox-never-device_runtime.jsonl",
        [
            {
                "timestamp": str(now),
                "event": "connection_failed",
                "sandbox_id": "sandbox-never",
                "server_url": "wss://northcua.com",
                "error": "timed out during opening handshake",
            }
        ],
    )
    _write_jsonl(
        batch_dir / "sandbox-logs" / "prepared" / "case-hello" / "sandbox-hello-device_runtime.jsonl",
        [
            {"timestamp": str(now), "event": "connected", "device_id": "device-hello"},
            {"timestamp": str(now + 1), "event": "device_hello_sent", "device_id": "device-hello"},
            {
                "timestamp": str(now + 2),
                "event": "connection_failed",
                "device_id": "device-hello",
                "error": "no close frame received or sent",
            },
        ],
    )
    _write_jsonl(
        batch_dir / "sandbox-logs" / "direct" / "case-ran" / "sandbox-ran-device_runtime.jsonl",
        [
            {"timestamp": str(now), "event": "connected", "device_id": "device-ran"},
            {"timestamp": str(now + 1), "event": "device_hello_sent", "device_id": "device-ran"},
            {"timestamp": str(now + 2), "event": "task_accepted", "device_id": "device-ran"},
            {"timestamp": str(now + 3), "event": "lease_started", "device_id": "device-ran"},
            {"timestamp": str(now + 4), "event": "device_command_received", "device_id": "device-ran"},
            {"timestamp": str(now + 5), "event": "lease_released", "device_id": "device-ran"},
        ],
    )

    report = generate_batch_health_report(
        batch_db=batch_db,
        controller_db=controller_db,
        run_dir=batch_dir,
        restart_time=str(time.time() - 60),
        recent_window_minutes=60,
    )

    assert report["batch"]["status_counts"]["timed_out"] == 1
    assert report["batch"]["eta"]["remaining_total"] == 1
    assert report["batch"]["eta"]["basis"] == "recent_window"
    assert report["batch"]["eta"]["recent_terminal_per_hour"] > 0
    assert report["batch"]["eta"]["eta_at"]
    assert report["batch"]["failure_root_causes"]["case_level_timeout"] == 1
    assert report["controller"]["live_membership"]["current_active"] == 1
    assert report["controller"]["live_membership"]["orphan_controller_task_id_mismatch_live"] == 1
    assert report["controller"]["disabled_window_tool_count_since_restart"] == 0
    assert report["controller"]["telemetry_event_health_since_restart"]["llm_error"] == 1
    assert report["controller"]["telemetry_event_health_since_restart"]["llm_fallback"] == 1
    assert report["controller"]["telemetry_event_health_since_restart"]["case_level_timeout"] == 1
    assert report["controller"]["command_metrics"]["counts_by_tool_status"]["click:succeeded"] == 1
    assert report["controller"]["command_metrics"]["recent_window_by_tool_status"]["click:succeeded"]["count"] == 1
    assert report["controller"]["command_metrics"]["problem_commands"][0]["tool"] == "shell.exec"
    assert report["controller"]["command_metrics"]["non_evaluator_shell_exec_total"] == 0
    assert report["controller"]["runtime_event_metrics"]["event_loop_lag_ms"]["p95"] == 125
    assert report["controller"]["runtime_event_metrics"]["recent_recv_q_max"]["max"] == 7
    assert report["controller"]["websocket_handshake_metrics"]["accepted_total"] == 1
    assert report["controller"]["websocket_handshake_metrics"]["recent_open_handshake_ms"]["p95"] == 42
    assert report["controller"]["websocket_handshake_metrics"]["recent_first_envelope_ms"]["p95"] == 63
    assert report["controller"]["readiness_stage_metrics"]["stage_counts"]["websocket_open"] == 1
    assert report["controller"]["readiness_stage_metrics"]["stage_counts"]["device_readiness_wait"] == 1
    assert (
        report["controller"]["readiness_stage_metrics"]["recent_by_stage"]["device_readiness_wait"]["duration_ms"][
            "p95"
        ]
        == 123
    )
    assert report["controller"]["process_executor_metrics"]["metric_counts"]["worker_connect"] == 1
    assert report["controller"]["process_executor_metrics"]["metric_counts"]["worker_ready"] == 1
    assert report["controller"]["process_executor_metrics"]["metric_counts"]["worker_command_roundtrip"] == 1
    assert report["controller"]["process_executor_metrics"]["logical_metric_counts"]["worker_command_roundtrip"] == 3
    assert report["controller"]["process_executor_metrics"]["worker_command_roundtrip_event_count"] == 1
    assert report["controller"]["process_executor_metrics"]["worker_command_roundtrip_command_count"] == 3
    assert report["controller"]["process_executor_metrics"]["worker_connect_ms"]["p95"] == 44
    assert report["controller"]["process_executor_metrics"]["worker_ready_ms"]["p95"] == 88
    assert report["controller"]["process_executor_metrics"]["worker_command_roundtrip_ms"]["p95"] == 12
    assert report["controller"]["worker_job_health"]["status_counts"]["running"] == 1
    assert report["controller"]["worker_job_health"]["rss_bytes"]["max"] == 2048
    assert report["prepared_sandboxes"]["active_unassigned_total"] == 2
    assert report["prepared_sandboxes"]["assigned_to_current_active_total"] == 1
    assert report["prepared_sandboxes"]["stale_assigned_total"] == 1
    assert report["prepared_sandboxes"]["expired_but_active_total"] == 1
    assert report["prepared_sandboxes"]["prepare_seconds_distribution"]["count"] == 4
    assert report["attempt_events"]["device_ready_failure_windows"]["60m"] == 1
    assert report["artifacts"]["mime_counts_since_restart"]["image/jpeg"] == 1
    assert report["worker_logs"]["since_restart"]["classification_counts"]["never_opened"] == 1
    assert report["worker_logs"]["since_restart"]["classification_counts"]["hello_no_lease"] == 1
    assert report["worker_logs"]["since_restart"]["classification_counts"]["ran_released"] == 1
    assert report["worker_logs"]["since_restart"]["connection_error_counts"]["opening_handshake_timeout"] == 1
    assert report["worker_logs"]["since_restart"]["connection_error_counts"]["no_close_frame"] == 1
    assert report["worker_logs"]["recent_window"]["path_kind_counts"]["direct"] == 2
    assert report["worker_logs"]["recent_window"]["path_kind_counts"]["prepared"] == 1
    assert "command latency recent" in report["markdown"]
    assert "controller loop lag" in report["markdown"]
    assert "websocket handshake" in report["markdown"]
    assert "readiness stages" in report["markdown"]
    assert "worker logs" in report["markdown"]
    assert "process executor" in report["markdown"]
    assert "prepared detail" in report["markdown"]
    assert "disabled window tools" in report["markdown"]
    assert "ETA: remaining=1" in report["markdown"]


def test_process_executor_metrics_support_epoch_cutoff_and_legacy_startup(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_db(controller_db)
    now = time.time()
    with sqlite3.connect(controller_db) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-old",
                "worker_job",
                "job-old",
                "process_executor_metric",
                json.dumps({"metric_name": "worker_startup", "value_ms": 10}),
                str(now - 600),
            ),
        )
        connection.execute(
            "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "trace-new",
                "worker_job",
                "job-new",
                "process_executor_metric",
                json.dumps({"metric_name": "worker_startup", "value_ms": 20}),
                str(now),
            ),
        )

        metrics = process_executor_metrics(connection, restart_ts=now - 60, recent_window_minutes=60)

    connect_ms = cast(dict[str, object], metrics["worker_connect_ms"])
    ready_ms = cast(dict[str, object], metrics["worker_ready_ms"])
    assert metrics["sample_count"] == 1
    assert connect_ms["p95"] == 20
    assert ready_ms["count"] == 0


def test_readiness_stage_metrics_classify_timeout_stage(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_db(controller_db)
    now = time.time()
    with sqlite3.connect(controller_db) as connection:
        connection.row_factory = sqlite3.Row
        _insert_readiness_stage_event(connection, "device-b", "websocket_open", now)
        _insert_readiness_stage_event(connection, "device-c", "websocket_open", now)
        _insert_readiness_stage_event(connection, "device-c", "first_heartbeat", now + 1)
        for device_id in ("device-a", "device-b", "device-c"):
            _insert_readiness_wait_timeout(connection, device_id, now + 10)

        metrics = readiness_stage_metrics(connection, restart_ts=now - 60, recent_window_minutes=60)

    assert metrics["timeout_classification_counts"] == {
        "opening_handshake_timeout": 1,
        "first_heartbeat_timeout": 1,
        "capability_ready_timeout": 1,
    }
    assert metrics["recent_timeout_classification_counts"] == metrics["timeout_classification_counts"]


def test_readiness_recent_timeout_classes_use_full_stage_history(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_db(controller_db)
    now = time.time()
    with sqlite3.connect(controller_db) as connection:
        connection.row_factory = sqlite3.Row
        _insert_readiness_stage_event(connection, "device-recent", "websocket_open", now - 120)
        _insert_readiness_wait_timeout(connection, "device-recent", now)

        metrics = readiness_stage_metrics(connection, restart_ts=now - 180, recent_window_minutes=1)

    assert metrics["timeout_classification_counts"] == {"first_heartbeat_timeout": 1}
    assert metrics["recent_timeout_classification_counts"] == {"first_heartbeat_timeout": 1}


def test_health_report_separates_evaluator_accepted_infeasible_from_failed(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-accepted-infeasible"
    batch_dir.mkdir()
    batch_db = batch_dir / "batch_state.sqlite"
    manifest = BatchManifest(
        batch_id="batch-accepted-infeasible",
        tasks=[
            BatchTaskSpec(task_id="case-accepted", agent_kind="desktop", query="use unavailable hardware"),
            BatchTaskSpec(task_id="case-real-failed", agent_kind="desktop", query="do impossible badly"),
            BatchTaskSpec(task_id="case-succeeded", agent_kind="desktop", query="do one"),
        ],
    )
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-accepted",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-accepted",
            error="missing hardware",
        )
    )
    repository.upsert_state(
        BatchTaskState(
            task_id="case-real-failed",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-failed",
            error="agent did the wrong thing",
        )
    )
    repository.upsert_state(BatchTaskState(task_id="case-succeeded", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    _controller_db(controller_db)
    _insert_controller_task_result(
        controller_db,
        task_id="controller-accepted",
        case_id="case-accepted",
        status="failed",
        result={
            "task_id": "controller-accepted",
            "state": "failed",
            "result": {
                "evaluation_report": {
                    "overall_verdict": "success",
                    "outcome_label": "success",
                    "controllability": "uncontrollable",
                }
            },
        },
    )
    _insert_controller_task_result(
        controller_db,
        task_id="controller-failed",
        case_id="case-real-failed",
        status="failed",
        result={
            "task_id": "controller-failed",
            "state": "failed",
            "result": {
                "evaluation_report": {
                    "overall_verdict": "failure",
                    "outcome_label": "failure",
                    "controllability": "controllable",
                }
            },
        },
    )

    report = generate_batch_health_report(
        batch_db=batch_db,
        controller_db=controller_db,
        run_dir=batch_dir,
        restart_time=str(time.time() - 60),
        recent_window_minutes=60,
    )

    assert report["batch"]["status_counts"]["failed"] == 2
    assert report["batch"]["accepted_infeasible_total"] == 1
    assert report["batch"]["accepted_infeasible_raw_failed_total"] == 1
    assert report["batch"]["actionable_failed_total"] == 1
    assert report["batch"]["task_success_total"] == 1
    assert "accepted_infeasible: 1" in report["markdown"]
    assert "actionable_failed: 1" in report["markdown"]


def _controller_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                agent_kind TEXT NOT NULL,
                submitter_id TEXT NOT NULL DEFAULT '',
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                device_id TEXT,
                lease_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE leases (
                lease_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                expires_at TEXT,
                generation INTEGER NOT NULL
            );
            CREATE TABLE devices (
                device_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                last_heartbeat_at TEXT,
                active_sessions INTEGER NOT NULL DEFAULT 0,
                load_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE commands (
                command_id TEXT PRIMARY KEY,
                protocol_command_id TEXT NOT NULL DEFAULT '',
                idempotency_key TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                lease_id TEXT NOT NULL DEFAULT '',
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE worker_jobs (
                job_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                agent_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_pid INTEGER NOT NULL DEFAULT 0,
                failure_kind TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                artifact_manifest_json TEXT NOT NULL DEFAULT '{}',
                heartbeat_payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                heartbeat_at TEXT,
                finished_at TEXT
            );
            CREATE TABLE telemetry_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL DEFAULT '',
                batch_id TEXT NOT NULL DEFAULT '',
                case_id TEXT NOT NULL DEFAULT '',
                trace_id TEXT NOT NULL DEFAULT '',
                entity_kind TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)


def _insert_readiness_stage_event(
    connection: sqlite3.Connection,
    device_id: str,
    stage: str,
    created_at: float,
) -> None:
    connection.execute(
        "INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "trace-" + device_id,
            "device",
            device_id,
            "device_readiness_stage",
            json.dumps({"stage": stage, "device_id": device_id}),
            str(created_at),
        ),
    )


def _insert_readiness_wait_timeout(
    connection: sqlite3.Connection,
    device_id: str,
    created_at: float,
) -> None:
    connection.execute(
        "INSERT INTO telemetry_events(entity_kind, entity_id, event_type, phase, status, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "sandbox",
            "sandbox-" + device_id,
            "status_changed",
            "prepared_sandbox",
            "failed",
            json.dumps(
                {
                    "message": "sandbox_stage_timing",
                    "details": {
                        "stage": "device_readiness_wait",
                        "status": "failed",
                        "readiness": "timed_out",
                        "device_id": device_id,
                        "sandbox_id": "sandbox-" + device_id,
                        "duration_ms": 120_000,
                    },
                }
            ),
            str(created_at),
        ),
    )


def _insert_controller_task_result(
    path: Path,
    *,
    task_id: str,
    case_id: str,
    status: str,
    result: dict[str, object],
) -> None:
    now = "2026-05-15T00:00:00+00:00"
    request_json = json.dumps({"metadata": {"batch_id": "batch-accepted-infeasible", "batch_task_id": case_id}})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO tasks(task_id, trace_id, agent_kind, request_json, status, result_json, created_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, "trace", "desktop", request_json, status, json.dumps(result), now, now),
        )


def _insert_controller_row(
    path: Path,
    task_id: str,
    case_id: str,
    session_id: str,
    device_id: str,
    tool_name: str,
) -> None:
    now = "2026-05-15T00:00:00+00:00"
    request_json = json.dumps({"metadata": {"batch_id": "batch-1", "batch_task_id": case_id}})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO tasks(task_id, trace_id, agent_kind, request_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, "trace", "desktop", request_json, "running", now),
        )
        connection.execute(
            "INSERT INTO sessions(session_id, task_id, device_id, lease_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, task_id, device_id, "lease-" + session_id, "running", now, now),
        )
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at) VALUES (?, ?, ?, ?)",
            (device_id, "online", "{}", now),
        )
        connection.execute(
            "INSERT INTO commands(command_id, idempotency_key, session_id, lease_id, device_id, status, request_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "cmd-" + task_id,
                "idem-" + task_id,
                session_id,
                "lease-" + session_id,
                device_id,
                "finished",
                json.dumps({"tool_name": tool_name}),
                now,
                "2026-05-15T00:00:03+00:00",
            ),
        )


def _insert_command(
    path: Path,
    *,
    command_id: str,
    session_id: str,
    device_id: str,
    status: str,
    request_json: dict[str, object],
    created_at: float,
    updated_at: float,
    result_json: dict[str, object] | None = None,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO commands(command_id, idempotency_key, session_id, lease_id, device_id, status, request_json, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                command_id,
                "idem-" + command_id,
                session_id,
                "lease-" + session_id,
                device_id,
                status,
                json.dumps(request_json),
                json.dumps(result_json or {}),
                str(created_at),
                str(updated_at),
            ),
        )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepared_sandbox_table(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prepared_sandboxes (
                task_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                runner_id TEXT NOT NULL,
                depth TEXT NOT NULL,
                provider TEXT NOT NULL,
                template TEXT NOT NULL,
                sandbox_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                reservation_token TEXT NOT NULL,
                reserved_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                prepare_started_at REAL NOT NULL,
                prepared_at REAL NOT NULL,
                assigned_at REAL NOT NULL,
                released_at REAL NOT NULL,
                prepare_seconds REAL NOT NULL,
                error TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(task_id, attempt)
            )
            """)


def _insert_prepared_sandbox(
    path: Path,
    *,
    task_id: str,
    status: str,
    now: float,
    expires_at: float,
    attempt: int = 1,
    reserved_at: float | None = None,
    prepare_started_at: float = 0.0,
    prepared_at: float = 0.0,
    assigned_at: float = 0.0,
    released_at: float = 0.0,
    prepare_seconds: float = 0.0,
    error: str = "",
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO prepared_sandboxes(
                task_id, attempt, status, runner_id, depth, provider, template,
                sandbox_id, device_id, lease_id, reservation_token, reserved_at,
                updated_at, expires_at, prepare_started_at, prepared_at, assigned_at,
                released_at, prepare_seconds, error, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                attempt,
                status,
                "runner-1",
                "full_init",
                "e2b",
                "template",
                "sandbox-" + task_id,
                "device-" + task_id,
                "lease-" + task_id,
                "reservation-" + task_id,
                reserved_at if reserved_at is not None else now - 90,
                now,
                expires_at,
                prepare_started_at,
                prepared_at,
                assigned_at,
                released_at,
                prepare_seconds,
                error,
                "{}",
            ),
        )
