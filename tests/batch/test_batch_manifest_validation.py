"""Batch manifest resume validation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchTaskSpec
from northcua.batch.post_run import PostRunHookSpec
from northcua.batch.runner.service import _validate_resume_identity_snapshot
from northcua.batch.runner.validation import _manifest_compare_payload, validate_existing_manifest


def test_manifest_compare_payload_sorts_required_capabilities() -> None:
    manifest = BatchManifest(
        batch_id="capability-order",
        tasks=[
            BatchTaskSpec(
                task_id="task-1",
                agent_kind="desktop",
                query="Run",
                required_capabilities={"desktop.screenshot", "a11y.tree", "desktop.input"},
            )
        ],
    )

    task_payload = _manifest_compare_payload(manifest)["tasks"][0]

    assert task_payload["required_capabilities"] == ["a11y.tree", "desktop.input", "desktop.screenshot"]


def test_existing_manifest_validation_ignores_required_capability_order(tmp_path: Path) -> None:
    requested = BatchManifest(
        batch_id="resume-capability-order",
        tasks=[
            BatchTaskSpec(
                task_id="task-1",
                agent_kind="desktop",
                query="Run",
                required_capabilities={"desktop.screenshot", "a11y.tree", "desktop.input"},
            )
        ],
    )
    persisted = json.loads(requested.model_dump_json())
    persisted["tasks"][0]["required_capabilities"] = ["desktop.input", "desktop.screenshot", "a11y.tree"]

    (tmp_path / "batch_state.sqlite").touch()
    (tmp_path / "manifest.json").write_text(json.dumps(persisted), encoding="utf-8")

    validate_existing_manifest(output_dir=tmp_path, manifest=requested)


def test_manifest_compare_payload_includes_empty_post_run_hooks_for_compatibility() -> None:
    manifest = BatchManifest(
        batch_id="post-run-default",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="Run")],
    )

    task_payload = _manifest_compare_payload(manifest)["tasks"][0]

    assert task_payload["post_run_hooks"] == []


def test_manifest_compare_payload_preserves_post_run_hook_contract() -> None:
    manifest = BatchManifest(
        batch_id="post-run-hook",
        tasks=[
            BatchTaskSpec(
                task_id="task-1",
                agent_kind="desktop",
                query="Run",
                post_run_hooks=[PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")],
            )
        ],
    )

    task_payload = _manifest_compare_payload(manifest)["tasks"][0]

    assert task_payload["post_run_hooks"][0]["hook_id"] == "reward"
    assert task_payload["post_run_hooks"][0]["kind"] == "command_json"


def test_selected_retry_identity_accepts_runtime_only_retry_budget_override(tmp_path: Path) -> None:
    identity = BatchManifest(
        batch_id="selected-retry-identity",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="Run", max_retries=0)],
    )
    persisted_bytes = identity.model_dump_json(indent=2).encode()
    (tmp_path / "batch_state.sqlite").touch()
    (tmp_path / "manifest.json").write_bytes(persisted_bytes)
    runtime = identity.model_copy(update={"tasks": [identity.tasks[0].model_copy(update={"max_retries": 2})]})

    _validate_resume_identity_snapshot(
        output_dir=tmp_path,
        identity_manifest=identity,
        identity_manifest_sha256=hashlib.sha256(persisted_bytes).hexdigest(),
        runtime_manifest=runtime,
    )

    assert (tmp_path / "manifest.json").read_bytes() == persisted_bytes


def test_selected_retry_identity_rejects_manifest_byte_drift(tmp_path: Path) -> None:
    identity = BatchManifest(
        batch_id="selected-retry-identity",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="Run")],
    )
    original_bytes = identity.model_dump_json().encode()
    (tmp_path / "batch_state.sqlite").touch()
    (tmp_path / "manifest.json").write_text(identity.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="persisted manifest changed"):
        _validate_resume_identity_snapshot(
            output_dir=tmp_path,
            identity_manifest=identity,
            identity_manifest_sha256=hashlib.sha256(original_bytes).hexdigest(),
            runtime_manifest=identity,
        )
