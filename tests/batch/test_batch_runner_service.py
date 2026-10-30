"""Batch runner service persistence helpers."""

from __future__ import annotations

import json

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchRunResumeConfig, BatchTaskSpec
from northcua.batch.runner.service import _redacted_argv, _write_run_config_snapshots


def test_run_config_snapshot_written_inside_batch_and_external_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_BACKLOG", "4096")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("HTTPS_PROXY", "https://user:pass@example.test")
    config = BatchRunConfig(
        manifest=BatchManifest(
            batch_id="batch-snap",
            tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="hello")],
        ),
        output_dir=tmp_path / "batch",
        token="controller-token",
    )
    batch_output_dir = config.output_dir / config.manifest.batch_id
    batch_output_dir.mkdir(parents=True)
    resume_config = BatchRunResumeConfig.from_run_config(config)

    _write_run_config_snapshots(config=config, resume_config=resume_config, batch_output_dir=batch_output_dir)

    local_latest = batch_output_dir / "run_config_snapshots" / "run_config.latest.json"
    external_latest = tmp_path / "run_config_snapshots" / "batch-snap.latest.json"
    assert local_latest.exists()
    assert external_latest.exists()
    local_payload = json.loads(local_latest.read_text(encoding="utf-8"))
    external_payload = json.loads(external_latest.read_text(encoding="utf-8"))
    assert local_payload == external_payload
    assert local_payload["batch_id"] == "batch-snap"
    assert local_payload["runtime_config"]["token"] == "<redacted>"
    assert local_payload["runtime_config"]["token_fingerprint"]
    assert local_payload["environment"]["NORTHCUA_CONTROLLER_WEBSOCKET_BACKLOG"] == "4096"
    assert local_payload["environment"]["OPENAI_API_KEY"] == "<redacted>"
    assert local_payload["environment"]["HTTPS_PROXY"] == "<redacted>"


def test_run_config_snapshot_redacts_secret_cli_arguments() -> None:
    assert _redacted_argv(["northcua", "batch", "resume", "--token", "secret", "--api-key=abc"]) == [
        "northcua",
        "batch",
        "resume",
        "--token",
        "<redacted>",
        "--api-key=<redacted>",
    ]
