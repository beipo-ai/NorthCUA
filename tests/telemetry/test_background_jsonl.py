"""Background JSONL writer hot-path safety tests."""

from __future__ import annotations

import json
import time

from northcua.telemetry.background_jsonl import BackgroundJsonlWriter


def test_background_jsonl_writer_flushes_and_closes_idempotently(tmp_path) -> None:
    writer = BackgroundJsonlWriter(queue_max_size=8, shutdown_timeout_seconds=1.0)
    path = tmp_path / "events.jsonl"

    assert writer.append_json(path, {"event": "one"})
    assert writer.append_text(path, json.dumps({"event": "two"}) + "\n")
    writer.flush()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"event": "one"}, {"event": "two"}]

    writer.close()
    writer.flush()
    assert writer.stats().closed is True
    assert writer.append_json(path, {"event": "after-close"}) is False


def test_background_jsonl_writer_flushes_periodically_without_explicit_flush(tmp_path) -> None:
    writer = BackgroundJsonlWriter(
        queue_max_size=8,
        shutdown_timeout_seconds=1.0,
        flush_interval_seconds=0.01,
        flush_row_count=100,
    )
    path = tmp_path / "events.jsonl"

    try:
        assert writer.append_json(path, {"event": "visible"})

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if path.exists() and path.read_text(encoding="utf-8").strip():
                break
            time.sleep(0.01)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows == [{"event": "visible"}]
    finally:
        writer.close()


def test_background_jsonl_writer_flushes_after_row_threshold(tmp_path) -> None:
    writer = BackgroundJsonlWriter(
        queue_max_size=8,
        shutdown_timeout_seconds=1.0,
        flush_interval_seconds=10.0,
        flush_row_count=1,
    )
    path = tmp_path / "events.jsonl"

    try:
        assert writer.append_json(path, {"event": "threshold"})

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if path.exists() and path.read_text(encoding="utf-8").strip():
                break
            time.sleep(0.01)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows == [{"event": "threshold"}]
    finally:
        writer.close()
