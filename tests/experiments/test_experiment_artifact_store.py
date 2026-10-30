"""Experiment artifact store tests."""

from __future__ import annotations

from pathlib import Path

from northcua.experiments.artifacts import ArtifactIndex, ArtifactIndexRow, LocalExperimentArtifactStore


def test_local_experiment_artifact_store_puts_and_lists_files(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    store = LocalExperimentArtifactStore(tmp_path / "store")

    uri = store.put(source=source, relative_path="case-1/source.txt")

    assert store.exists(uri)
    assert store.resolve_uri(uri).read_text(encoding="utf-8") == "hello"
    assert uri in store.list("case-1")


def test_local_experiment_artifact_store_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    store = LocalExperimentArtifactStore(tmp_path / "store")

    for relative_path in ("../outside.txt", "/tmp/outside.txt"):
        try:
            store.put(source=source, relative_path=relative_path)
        except ValueError as exc:
            assert "artifact" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {relative_path}")


def test_artifact_index_round_trips_rows(tmp_path: Path) -> None:
    index = ArtifactIndex(tmp_path / "artifact_index.jsonl")
    row = ArtifactIndexRow(task_id="task", kind="screenshot", uri="file:///tmp/s.png", path="/tmp/s.png")

    index.write([row])

    assert index.read() == [row]
