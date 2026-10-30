"""Batch run config persistence tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchRunResumeConfig, BatchTaskSpec
from northcua.batch.preparation import PreparedSandboxPolicy


def test_batch_resume_config_can_omit_persisted_token() -> None:
    manifest = BatchManifest(
        batch_id="no-token-persist",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Open page")],
    )

    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(manifest=manifest, token="secret-token", persist_token=False)
    )

    assert snapshot.token == ""
    assert snapshot.persist_token is False


def test_batch_resume_config_fingerprints_manifest_default_token() -> None:
    manifest = BatchManifest(
        batch_id="default-token-fingerprint",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Open page")],
    )
    manifest.defaults.token = "default-token"

    snapshot = BatchRunResumeConfig.from_run_config(BatchRunConfig(manifest=manifest, persist_token=False))

    assert snapshot.token == ""
    assert snapshot.token_fingerprint


def test_batch_resume_config_preserves_token_persistence_policy() -> None:
    manifest = BatchManifest(
        batch_id="no-token-persist",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Open page")],
    )
    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(manifest=manifest, token="secret-token", persist_token=False)
    )

    merged = snapshot.merge_overrides(token="resume-token")

    assert merged.token == "resume-token"
    assert merged.persist_token is False


def test_prepared_sandbox_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PreparedSandboxPolicy.model_validate({"enabled": True, "buffer_size": 4})


def test_batch_resume_config_preserves_prepared_warmup_policy() -> None:
    manifest = BatchManifest(
        batch_id="prepared-warmup-resume",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Count files")],
    )
    policy = PreparedSandboxPolicy(
        enabled=True,
        buffer=10,
        prepare_concurrency=3,
        initial_warmup_wait_seconds=4.5,
        claim_pending_wait_seconds=0.25,
        claim_wait_seconds=30,
    )
    snapshot = BatchRunResumeConfig.from_run_config(BatchRunConfig(manifest=manifest, prepared_sandbox_policy=policy))

    merged = snapshot.merge_overrides(concurrency=2)

    assert merged.prepared_sandbox_policy.initial_warmup_wait_seconds == 4.5
    assert merged.prepared_sandbox_policy.claim_pending_wait_seconds == 0.25
    assert merged.prepared_sandbox_policy.claim_wait_seconds == 30


def test_batch_resume_config_preserves_telemetry_fields(tmp_path: Path) -> None:
    manifest = BatchManifest(
        batch_id="telemetry-resume",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Count files")],
    )
    controller_db_path = tmp_path / "controller.sqlite"
    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(
            manifest=manifest,
            telemetry_experiment_id="experiment-1",
            telemetry_controller_db_path=controller_db_path,
        )
    )

    merged = snapshot.merge_overrides(concurrency=2)

    assert merged.telemetry_experiment_id == "experiment-1"
    assert merged.telemetry_controller_db_path == controller_db_path


def test_batch_resume_config_can_override_controller_db_path(tmp_path: Path) -> None:
    snapshot = BatchRunResumeConfig()
    controller_db_path = tmp_path / "controller.sqlite"

    merged = snapshot.merge_overrides(telemetry_controller_db_path=controller_db_path)

    assert merged.telemetry_controller_db_path == controller_db_path


def test_batch_resume_config_preserves_submit_throttle() -> None:
    manifest = BatchManifest(
        batch_id="submit-throttle",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Count files")],
    )
    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(
            manifest=manifest,
            controller_submit_concurrency=20,
            controller_submit_jitter_seconds=2.0,
        )
    )

    merged = snapshot.merge_overrides()

    assert merged.controller_submit_concurrency == 20
    assert merged.controller_submit_jitter_seconds == 2.0


def test_batch_resume_config_preserves_device_readiness_timeout() -> None:
    manifest = BatchManifest(
        batch_id="readiness-timeout",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Count files")],
    )
    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(
            manifest=manifest,
            device_readiness_timeout_seconds=720.0,
        )
    )

    merged = snapshot.merge_overrides()
    overridden = snapshot.merge_overrides(device_readiness_timeout_seconds=300.0)

    assert merged.device_readiness_timeout_seconds == 720.0
    assert overridden.device_readiness_timeout_seconds == 300.0


def test_batch_resume_config_preserves_infra_retry_backoff() -> None:
    manifest = BatchManifest(
        batch_id="infra-backoff",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Count files")],
    )
    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(
            manifest=manifest,
            infra_retry_backoff_seconds=[30.0, 90.0, 180.0],
        )
    )

    merged = snapshot.merge_overrides()
    overridden = snapshot.merge_overrides(infra_retry_backoff_seconds=[5.0, 10.0])

    assert merged.infra_retry_backoff_seconds == [30.0, 90.0, 180.0]
    assert overridden.infra_retry_backoff_seconds == [5.0, 10.0]


def test_batch_resume_config_preserves_controller_status_concurrency() -> None:
    manifest = BatchManifest(
        batch_id="status-concurrency",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Count files")],
    )
    snapshot = BatchRunResumeConfig.from_run_config(
        BatchRunConfig(
            manifest=manifest,
            controller_submit_concurrency=20,
            controller_status_concurrency=4,
        )
    )

    merged = snapshot.merge_overrides()
    overridden = snapshot.merge_overrides(controller_status_concurrency=2)

    assert merged.controller_status_concurrency == 4
    assert overridden.controller_status_concurrency == 2
