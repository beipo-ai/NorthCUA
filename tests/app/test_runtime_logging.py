from __future__ import annotations

import json
import logging
from pathlib import Path

from northcua.app.logging import configure_runtime_logging, log_runtime_event, runtime_logging_context
from northcua.core.config import ArtifactConfig
from northcua.runtime.artifacts.store import NorthCUAArtifactStore


def test_runtime_jsonl_handler_writes_structured_records(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-1", ArtifactConfig())
    configure_runtime_logging(store)

    logger = logging.getLogger("northcua.tests")
    with runtime_logging_context(log_path=store.runtime_log_path, session_id="sess-1"):
        log_runtime_event(
            logger,
            "hello runtime log",
            level=logging.DEBUG,
            round_num=2,
            status="continue",
            trace_dir=Path("/tmp/northcua-trace"),
        )

    payload = json.loads(store.runtime_log_path.read_text(encoding="utf-8").strip().splitlines()[-1])

    assert payload["message"] == "hello runtime log"
    assert payload["level"] == "DEBUG"
    assert payload["session_id"] == "sess-1"
    assert payload["extra"]["event"] == "hello runtime log"
    assert payload["extra"]["round_num"] == 2
    assert payload["extra"]["status"] == "continue"
    assert payload["extra"]["trace_dir"] == "/tmp/northcua-trace"


def test_runtime_jsonl_handler_sanitizes_reserved_extra_keys(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-2", ArtifactConfig())
    configure_runtime_logging(store)

    logger = logging.getLogger("northcua.tests")
    with runtime_logging_context(log_path=store.runtime_log_path, session_id="sess-2"):
        log_runtime_event(
            logger,
            "reserved keys",
            message="human-readable detail",
            module="fake_module",
        )

    payload = json.loads(store.runtime_log_path.read_text(encoding="utf-8").strip().splitlines()[-1])

    assert payload["message"] == "reserved keys"
    assert payload["extra"]["field_message"] == "human-readable detail"
    assert payload["extra"]["field_module"] == "fake_module"


def test_runtime_logging_uses_single_handler_across_multiple_artifact_stores(tmp_path) -> None:
    store_one = NorthCUAArtifactStore.for_session(tmp_path, "sess-1", ArtifactConfig())
    store_two = NorthCUAArtifactStore.for_session(tmp_path, "sess-2", ArtifactConfig())
    configure_runtime_logging(store_one)
    configure_runtime_logging(store_two)

    logger = logging.getLogger("northcua.tests")
    with runtime_logging_context(log_path=store_one.runtime_log_path, session_id="sess-1"):
        log_runtime_event(logger, "first")
    with runtime_logging_context(log_path=store_two.runtime_log_path, session_id="sess-2"):
        log_runtime_event(logger, "second")

    assert len(store_one.runtime_log_path.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert len(store_two.runtime_log_path.read_text(encoding="utf-8").strip().splitlines()) == 1
