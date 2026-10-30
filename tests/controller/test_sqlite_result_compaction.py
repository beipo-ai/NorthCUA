"""Regression coverage for controller SQLite result compaction."""

from __future__ import annotations

import base64
import json

from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua.controller.storage.queue import TaskQueue
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef, StorageKind
from northcua_protocol.payloads import CommandRequest, CommandResult, TaskFinished, TaskRequest
from northcua_protocol.state import CommandState, TaskState


def _store(tmp_path) -> SQLiteControllerStore:
    return SQLiteControllerStore(tmp_path / "controller.sqlite")


def _tree_payload() -> dict[str, object]:
    return {
        "status": "success",
        "scope": "focused",
        "tree": {
            "role": "window",
            "name": "Root",
            "children": [{"role": "button", "name": f"Button {index}", "children": []} for index in range(100)],
        },
    }


def test_command_result_sqlite_copy_omits_a11y_tree_and_image_data_url(tmp_path) -> None:
    store = _store(tmp_path)
    request = CommandRequest(
        command_id="cmd-a11y",
        idempotency_key="idem-a11y",
        session_id="sess-a",
        lease_id="lease-a",
        device_id="dev-a",
        tool_id="capture_accessibility_tree",
    )
    store.claim_command(request)
    data_url = f"data:image/png;base64,{base64.b64encode(b'fake-png').decode('ascii')}"
    screenshot_ref = ArtifactRef.from_inline_bytes(
        artifact_id="cmd-a11y-screenshot_url",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-png",
        mime_type="image/png",
    )
    live_result = CommandResult(
        command_id=request.command_id,
        session_id=request.session_id,
        lease_id=request.lease_id,
        device_id=request.device_id,
        status=CommandState.SUCCEEDED,
        result={
            **_tree_payload(),
            "screenshot_url": data_url,
            "screenshot_path": "/agent-artifacts/sess/query_0001/action_step0_tool_1.png",
        },
        artifacts=[screenshot_ref],
    )

    assert store.record_command_result(live_result) is None

    persisted = CommandResult.model_validate_json(store.command_row("cmd-a11y")["result_json"])
    assert isinstance(persisted.result, dict)
    assert persisted.result["status"] == "success"
    assert persisted.result["scope"] == "focused"
    assert persisted.result["screenshot_path"].endswith("action_step0_tool_1.png")
    assert persisted.result["screenshot_url"] == ""
    assert persisted.result["screenshot_url_omitted"]["kind"] == "image_data_url"
    assert persisted.result["tree"]["omitted"] is True
    assert persisted.result["tree"]["kind"] == "accessibility_tree"
    assert persisted.result["tree"]["node_count"] == 101
    assert len(persisted.artifacts) == 1
    assert persisted.artifacts[0].artifact_id == screenshot_ref.artifact_id
    assert persisted.artifacts[0].storage_kind == StorageKind.DEVICE_LOCAL
    assert persisted.artifacts[0].uri.startswith("agent-artifacts://sha256/")
    assert persisted.artifacts[0].inline_data is None
    assert persisted.result["sqlite_omitted_artifacts"][0]["artifact_id"] == screenshot_ref.artifact_id
    result_json = store.command_row("cmd-a11y")["result_json"]
    assert "Button 99" not in result_json
    assert "data:image/png;base64" not in result_json
    assert screenshot_ref.inline_data not in result_json
    store.close()


def test_task_finished_sqlite_copy_is_compacted_but_live_payload_is_unchanged(tmp_path) -> None:
    store = _store(tmp_path)
    queue = TaskQueue(store=store, events=ControllerEventLog(store))
    request = TaskRequest(
        task_id="task-a11y",
        trace_id="trace-a11y",
        agent_kind="desktop",
        request="inspect app",
    )
    queue.accept(request, session_id="sess-a11y", submitter_id="submitter-a")
    payload = TaskFinished(
        task_id=request.task_id,
        trace_id=request.trace_id,
        session_id="sess-a11y",
        state=TaskState.FINISHED,
        result=_tree_payload(),
    )

    queue.finish(payload)

    assert isinstance(payload.result, dict)
    assert isinstance(payload.result["tree"], dict)
    assert payload.result["tree"]["children"][99]["name"] == "Button 99"
    persisted = TaskFinished.model_validate_json(store.task_row("task-a11y")["result_json"])
    assert isinstance(persisted.result, dict)
    assert persisted.result["tree"]["omitted"] is True
    assert persisted.result["tree"]["kind"] == "accessibility_tree"
    assert persisted.result["tree"]["node_count"] == 101
    assert "Button 99" not in store.task_row("task-a11y")["result_json"]
    store.close()


def test_artifact_catalog_sqlite_copy_omits_inline_data(tmp_path) -> None:
    store = _store(tmp_path)
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="shot-inline",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-image-bytes",
        mime_type="image/png",
    )

    store.record_artifact_ref(artifact, trace_id="trace-artifact", source="command_output")

    row = store.artifact_rows(trace_id="trace-artifact")[0]
    artifact_json = json.loads(row["artifact_json"])
    assert row["sha256"] == artifact.sha256
    assert row["size_bytes"] == artifact.size_bytes
    assert artifact_json["artifact_id"] == artifact.artifact_id
    assert artifact_json["inline_data_omitted"] is True
    assert "inline_data" not in artifact_json
    assert artifact.inline_data not in row["artifact_json"]
    store.close()
