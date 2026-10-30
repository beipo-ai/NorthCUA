"""Controller artifact fallback behavior for SQLite-compacted rows."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
from northcua.controller.storage.result_compaction import compact_artifact_ref_json_for_sqlite
from northcua.experiments.verification.controller_artifacts import ControllerArtifactMaterializer
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef


def test_controller_artifact_materializer_skips_sqlite_omitted_inline_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    state = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id=f"{state.controller_task_id}-session-file-query_0001%2Faction_step1.png",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-image",
        mime_type="image/png",
    )
    _seed_compacted_artifact_row(db_path=db_path, state=state, artifact=artifact)

    materialized = ControllerArtifactMaterializer().materialize(
        output_dir=tmp_path,
        states=[state],
        evaluator_kind="desktop_round_completion",
        db_path=db_path,
    )

    assert materialized is None
    assert not (tmp_path / "materialized").exists()


def _seed_compacted_artifact_row(*, db_path: Path, state: BatchTaskState, artifact: ArtifactRef) -> None:
    with sqlite3.connect(db_path) as connection:
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
                "catalog-1",
                artifact.artifact_id,
                "task_output",
                state.trace_id,
                state.controller_task_id,
                state.session_id,
                "",
                artifact.kind.value,
                artifact.storage_kind.value,
                artifact.mime_type,
                artifact.sha256,
                artifact.size_bytes,
                artifact.uri,
                compact_artifact_ref_json_for_sqlite(artifact),
                "2026-05-12T00:00:00Z",
            ),
        )
        connection.commit()
