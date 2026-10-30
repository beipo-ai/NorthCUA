"""Shared fixtures for experiment trace verifier tests."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from northcua.batch.contracts import BatchTaskState
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef, ArtifactScope, RetentionPolicy

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def seed_agent_artifact_session(
    tmp_path: Path,
    spec,
    state: BatchTaskState,
    *,
    root_name: str = "agent-artifacts",
) -> Path:
    session_root = tmp_path / root_name / state.session_id
    session_root.mkdir(parents=True, exist_ok=True)
    (session_root / "steps.jsonl").write_text('{"session_step":1}\n', encoding="utf-8")
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    (session_root / "runtime.jsonl").write_text('{"event":"running"}\n', encoding="utf-8")
    screenshot = session_root / "query_0001" / "action_step1_history_overlay_1.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    screenshot.write_bytes(_TINY_PNG)
    evaluator_root = session_root / "evaluations" / spec.trace_expectations.evaluator_kind / "round_0001"
    evaluator_root.mkdir(parents=True, exist_ok=True)
    (evaluator_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (evaluator_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (evaluator_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")
    return session_root


def move_seeded_artifact_to_controller_blob(
    *,
    db_path: Path,
    blob_dir: Path,
    artifact_id: str,
    payload: bytes,
    mime_type: str,
) -> None:
    blob_dir.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256(payload).hexdigest()
    (blob_dir / sha256).write_bytes(payload)
    artifact = ArtifactRef.from_controller_blob(
        artifact_id=artifact_id,
        kind=ArtifactKind.JSON,
        uri=f"controller-blob://{sha256}",
        mime_type=mime_type,
        sha256=sha256,
        size_bytes=len(payload),
        artifact_scope=ArtifactScope.TASK,
        retention_policy=RetentionPolicy.TASK,
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE artifacts
            SET storage_kind=?,
                uri=?,
                sha256=?,
                size_bytes=?,
                artifact_json=?
            WHERE original_artifact_id=?
            """,
            (
                artifact.storage_kind.value,
                artifact.uri,
                artifact.sha256,
                artifact.size_bytes,
                artifact.model_dump_json(),
                artifact_id,
            ),
        )
        connection.commit()


def seed_controller_artifact_db(db_path: Path, state: BatchTaskState) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript("""
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            original_artifact_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            trace_id TEXT NOT NULL,
            task_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            command_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            storage_kind TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            uri TEXT NOT NULL DEFAULT '',
            artifact_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
    artifacts = [
        (
            f"{state.controller_task_id}-steps",
            "task_output",
            ArtifactKind.JSON,
            b'{"session_step":1}\n',
            "application/x-ndjson",
        ),
        (
            f"{state.controller_task_id}-llm-trace",
            "task_output",
            ArtifactKind.JSON,
            b'{"trace":"main"}\n',
            "application/x-ndjson",
        ),
        (
            f"{state.controller_task_id}-evaluation-desktop_round_completion-round_0001-final_report",
            "task_output",
            ArtifactKind.JSON,
            b'{"overall_verdict":"success"}',
            "application/json",
        ),
        (
            f"{state.controller_task_id}-evaluation-desktop_round_completion-round_0001-evidence_manifest",
            "task_output",
            ArtifactKind.JSON,
            b'{"refs":[]}',
            "application/json",
        ),
        (
            f"{state.controller_task_id}-evaluation-desktop_round_completion-round_0001-llm_trace",
            "task_output",
            ArtifactKind.JSON,
            b'{"trace":"eval"}\n',
            "application/x-ndjson",
        ),
        (
            f"{state.controller_task_id}-session-file-query_0001%2Faction_step1_history_overlay_1.png",
            "task_output",
            ArtifactKind.SCREENSHOT,
            _TINY_PNG,
            "image/png",
        ),
        ("call_1-screenshot_url", "command_output", ArtifactKind.SCREENSHOT, _TINY_PNG, "image/png"),
    ]
    for index, (artifact_id, source, kind, data, mime_type) in enumerate(artifacts):
        artifact = ArtifactRef.from_inline_bytes(
            artifact_id=artifact_id,
            kind=kind,
            data=data,
            mime_type=mime_type,
            artifact_scope=ArtifactScope.TASK,
            retention_policy=RetentionPolicy.TASK,
        )
        connection.execute(
            """
            INSERT INTO artifacts(
                artifact_id,
                original_artifact_id,
                source,
                trace_id,
                task_id,
                session_id,
                command_id,
                kind,
                storage_kind,
                mime_type,
                sha256,
                size_bytes,
                uri,
                artifact_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"catalog-{index}",
                artifact.artifact_id,
                source,
                state.trace_id,
                state.controller_task_id,
                state.session_id,
                "command-1" if source == "command_output" else "",
                artifact.kind.value,
                artifact.storage_kind.value,
                artifact.mime_type,
                artifact.sha256,
                artifact.size_bytes,
                artifact.uri,
                artifact.model_dump_json(),
                f"2026-04-29T00:00:0{index}Z",
            ),
        )
    connection.commit()
    connection.close()
