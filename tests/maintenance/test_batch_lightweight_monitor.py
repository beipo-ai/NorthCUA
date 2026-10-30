from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from northcua.maintenance.batch_lightweight_monitor import generate_lightweight_batch_monitor_report


def test_lightweight_monitor_uses_sql_aggregates_and_bounded_jsonl_tail(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch" / "run-1"
    run_dir.mkdir(parents=True)
    batch_db = run_dir / "batch_state.sqlite"
    controller_db = tmp_path / "controller.sqlite"
    _create_batch_db(batch_db)
    _create_controller_db(controller_db)
    now = datetime.now(timezone.utc)
    launch = now - timedelta(minutes=15)
    _write_attempt_events(run_dir / "batch_attempt_events.jsonl", launch=launch, now=now)
    _write_runtime_events(tmp_path / "agent-artifacts" / "session-1" / "runtime.jsonl", now=now)
    (run_dir / "prepared_sandbox_events.jsonl").write_text(
        json.dumps({"timestamp": now.isoformat(), "message": "prepared_sandbox_assigned"}) + "\n",
        encoding="utf-8",
    )

    report = generate_lightweight_batch_monitor_report(
        batch_db=batch_db,
        controller_db=controller_db,
        run_dir=run_dir,
        launch_time=launch.isoformat(),
        tail_mb=1,
        include_processes=False,
        slow_controller_counts=True,
    )

    assert report["batch"]["total"] == 4
    assert report["batch"]["succeeded"] == 1
    assert report["batch"]["submitted"] == 1
    assert report["prepared_sandboxes"]["status_counts"] == {"assigned": 1, "prepared": 1}
    assert report["controller"]["tables"]["tasks"]["running"] == 1
    assert report["attempt_events"]["terminal_counts"] == {"failed": 1, "succeeded": 2}
    assert report["attempt_events"]["throughput_windows"]["20"]["terminal_delta"] == 3
    assert report["attempt_events"]["submit_by_second"]["controller_submit_open_started"]["max"] == 2
    assert report["attempt_events"]["preflight"]["backpressure_wait_seconds"]["p95"] == 0.2
    assert report["controller"]["runtime"]["operation_timings"]["storage.lease_generation_query"]["count"] == 2
    assert (
        report["controller"]["runtime"]["operation_timings"]["storage.lease_generation_query"]["latest"]["p95_ms"]
        == 12.0
    )
    assert report["controller"]["worker_heartbeat"]["sampled_workers"] == 3
    assert report["controller"]["worker_heartbeat"]["step_index"] == {
        "known": 1,
        "unknown": 1,
        "no_heartbeat_payload": 1,
    }
    assert report["controller"]["worker_heartbeat"]["active_command_workers"] == 1
    assert report["controller"]["worker_heartbeat"]["rss"]["current_mib"]["max"] == 123.0
    assert report["controller"]["worker_heartbeat"]["rss"]["peak_mib"]["max"] == 456.0
    assert report["runtime_jsonl"]["files_read"] == 1
    assert report["runtime_jsonl"]["llm_calls"] == {"started": 2, "timed_out": 1, "completed": 1}
    assert report["runtime_jsonl"]["llm_call_wall_seconds"]["p50"] == 900.0
    assert report["runtime_jsonl"]["llm_completed_wall_seconds"]["max"] == 17.5
    assert report["runtime_jsonl"]["llm_timeout_wall_seconds"]["max"] == 900.0
    assert report["runtime_jsonl"]["fallback"]["failed"]["0:gpt-5.5:primary"] == 1
    assert report["runtime_jsonl"]["fallback"]["selected"]["1:gpt-5.5-previous-primary:fallback"] == 1
    assert report["runtime_jsonl"]["request_size_bytes"]["max"] == 2_000_000.0
    assert "llm_timeouts_present" in report["summary"]["issues"]
    assert "LLM wall-clock:" in report["markdown"]
    assert report["runtime"]["processes"] == {}


def _create_batch_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE batch_tasks (
                task_id TEXT,
                status TEXT,
                state_json TEXT,
                last_failure_category TEXT,
                retry_budget_remaining INTEGER
            )
            """)
        conn.executemany(
            "INSERT INTO batch_tasks VALUES (?, ?, ?, ?, ?)",
            [
                ("case-1", "succeeded", "{}", None, 3),
                ("case-2", "failed", "{}", "controller_submit", 2),
                ("case-3", "submitted", json.dumps({"owner_id": "runner-current"}), None, 3),
                ("case-4", "pending", "{}", None, 3),
            ],
        )
        conn.execute("""
            CREATE TABLE prepared_sandboxes (
                task_id TEXT,
                status TEXT,
                expires_at REAL
            )
            """)
        conn.executemany(
            "INSERT INTO prepared_sandboxes VALUES (?, ?, ?)",
            [("case-3", "assigned", 9999999999), ("case-5", "prepared", 9999999999)],
        )


def _create_controller_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        for table in ("tasks", "sessions", "worker_jobs", "devices", "leases", "commands"):
            conn.execute(f"CREATE TABLE {table} (status TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('running')")
        conn.execute("INSERT INTO sessions VALUES ('running')")
        conn.execute("INSERT INTO worker_jobs VALUES ('running')")
        conn.execute("ALTER TABLE worker_jobs ADD COLUMN heartbeat_payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE worker_jobs ADD COLUMN heartbeat_at REAL NOT NULL DEFAULT 0")
        conn.execute(
            "UPDATE worker_jobs SET heartbeat_payload_json = ?, heartbeat_at = ? WHERE status = 'running'",
            (
                json.dumps(
                    {
                        "phase": "running",
                        "step_index": 3,
                        "active_command_id": "cmd-live",
                        "rss_bytes": 456 * 1024 * 1024,
                        "peak_rss_bytes": 456 * 1024 * 1024,
                        "current_rss_bytes": 123 * 1024 * 1024,
                    }
                ),
                123.0,
            ),
        )
        conn.execute(
            "INSERT INTO worker_jobs(status, heartbeat_payload_json, heartbeat_at) VALUES (?, ?, ?)",
            ("running", json.dumps({"phase": "runner_started", "step_index": 0}), 122.0),
        )
        conn.execute(
            "INSERT INTO worker_jobs(status, heartbeat_payload_json, heartbeat_at) VALUES (?, ?, ?)",
            ("queued", "{}", 121.0),
        )
        conn.execute("INSERT INTO devices VALUES ('online')")
        conn.execute("INSERT INTO leases VALUES ('leased')")
        conn.execute("INSERT INTO commands VALUES ('running')")
        conn.execute("""
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY,
                trace_id TEXT,
                entity_type TEXT,
                entity_id TEXT,
                event_type TEXT,
                payload_json TEXT,
                created_at TEXT
            )
            """)
        payload = {
            "event_loop_lag_ms": 50,
            "socket_count": 12,
            "controller_operation_timings": {
                "storage.lease_generation_query": {
                    "count": 2,
                    "sampled_count": 2,
                    "dropped_samples": 0,
                    "p50_ms": 8.0,
                    "p90_ms": 12.0,
                    "p95_ms": 12.0,
                    "max_ms": 12.0,
                }
            },
        }
        conn.execute(
            """
            INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("trace", "controller", "runtime", "controller_runtime_telemetry", json.dumps(payload), "2026-01-01"),
        )


def _write_attempt_events(path: Path, *, launch: datetime, now: datetime) -> None:
    rows = [
        {"timestamp": (launch - timedelta(seconds=1)).isoformat(), "message": "attempt_cleanup_finished"},
        {
            "timestamp": (now - timedelta(minutes=9)).isoformat(),
            "message": "controller_submit_preflight_finished",
            "details": {"backpressure_wait_seconds": 0.1, "effective_submit_jitter_seconds": 30},
        },
        {
            "timestamp": (now - timedelta(minutes=9)).isoformat(),
            "message": "controller_submit_preflight_finished",
            "details": {"backpressure_wait_seconds": 0.2, "effective_submit_jitter_seconds": 60},
        },
        {"timestamp": (now - timedelta(minutes=8)).isoformat(), "message": "controller_submit_open_started"},
        {"timestamp": (now - timedelta(minutes=8)).isoformat(), "message": "controller_submit_open_started"},
        {
            "timestamp": (now - timedelta(minutes=7)).isoformat(),
            "message": "attempt_cleanup_finished",
            "details": {"terminal_status": "succeeded"},
        },
        {
            "timestamp": (now - timedelta(minutes=6)).isoformat(),
            "message": "attempt_cleanup_finished",
            "details": {"terminal_status": "failed"},
        },
        {
            "timestamp": (now - timedelta(minutes=4)).isoformat(),
            "message": "attempt_cleanup_finished",
            "details": {"terminal_status": "succeeded"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_runtime_events(path: Path, *, now: datetime) -> None:
    path.parent.mkdir(parents=True)
    rows = [
        {
            "ts": (now - timedelta(minutes=8)).isoformat(),
            "message": "trace_capture_before_model",
            "extra": {"request_size_bytes": 2_000_000, "api_type": "openai_responses"},
        },
        {
            "ts": (now - timedelta(minutes=8)).isoformat(),
            "message": "llm_fallback_attempt_started",
            "extra": {
                "candidate_index": 0,
                "candidate_name": "gpt-5.5",
                "is_primary": True,
                "model": "gpt-5.5",
                "api_type": "openai_responses",
            },
        },
        {
            "ts": (now - timedelta(minutes=8)).isoformat(),
            "message": "single_model_step_llm_call_started",
            "extra": {
                "timeout_seconds": 900.0,
                "configured_timeout_seconds": 300,
                "configured_max_retries": 2,
                "model": "gpt-5.5",
                "api_type": "openai_responses",
            },
        },
        {
            "ts": (now - timedelta(minutes=7)).isoformat(),
            "message": "single_model_step_llm_call_timed_out",
            "extra": {
                "elapsed_seconds": 900.0,
                "timeout_seconds": 900.0,
                "configured_timeout_seconds": 300,
                "configured_max_retries": 2,
                "model": "gpt-5.5",
                "api_type": "openai_responses",
            },
        },
        {
            "ts": (now - timedelta(minutes=7)).isoformat(),
            "message": "llm_fallback_attempt_failed",
            "extra": {
                "candidate_index": 0,
                "candidate_name": "gpt-5.5",
                "is_primary": True,
                "error_type": "ModelStepLLMTimeoutError",
            },
        },
        {
            "ts": (now - timedelta(minutes=7)).isoformat(),
            "message": "llm_fallback_attempt_selected",
            "extra": {
                "candidate_index": 1,
                "candidate_name": "gpt-5.5-previous-primary",
                "is_primary": False,
            },
        },
        {
            "ts": (now - timedelta(minutes=6)).isoformat(),
            "message": "single_model_step_llm_call_started",
            "extra": {
                "timeout_seconds": 900.0,
                "configured_timeout_seconds": 300,
                "configured_max_retries": 2,
                "model": "gpt-5.5",
                "api_type": "openai_responses",
            },
        },
        {
            "ts": (now - timedelta(minutes=5)).isoformat(),
            "message": "single_model_step_llm_call_completed",
            "extra": {
                "elapsed_seconds": 17.5,
                "timeout_seconds": 900.0,
                "configured_timeout_seconds": 300,
                "configured_max_retries": 2,
                "model": "gpt-5.5",
                "api_type": "openai_responses",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
