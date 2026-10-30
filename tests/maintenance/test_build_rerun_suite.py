from __future__ import annotations

import json
from pathlib import Path

import yaml

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.manifest.loader import load_batch_manifest
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.experiments.cases.contracts import CaseSpec, CaseSuiteSpec
from northcua.maintenance.build_rerun_suite import build_rerun_manifest
from northcua.sandboxes.contracts import SandboxProviderSpec


def test_build_rerun_manifest_collects_non_success_and_rewrites_template(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "batch-1"
    source_dir.mkdir(parents=True)
    source_manifest = BatchManifest(
        batch_id="batch-1",
        defaults={"concurrency": 4, "sandbox": {"provider": "e2b", "template": "old-template"}},
        tasks=[
            BatchTaskSpec(
                task_id="case-success",
                agent_kind="desktop",
                query="success",
                sandbox=SandboxProviderSpec(provider="e2b", template="old-template"),
            ),
            BatchTaskSpec(
                task_id="case-failed",
                agent_kind="desktop",
                query="failed",
                metadata={"original": "yes"},
                sandbox=SandboxProviderSpec(provider="e2b", template="old-template"),
            ),
        ],
    )
    (source_dir / "manifest.json").write_text(source_manifest.model_dump_json(indent=2), encoding="utf-8")
    batch_db = source_dir / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(source_manifest)
    repository.upsert_state(BatchTaskState(task_id="case-success", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    repository.upsert_state(BatchTaskState(task_id="case-failed", status=BatchTaskStatus.FAILED, attempts=1))
    with repository._connection:  # noqa: SLF001 - fixture needs deterministic dedupe timestamps.
        repository._connection.execute("UPDATE batch_tasks SET updated_at=10 WHERE task_id='case-failed'")
    newer_dir = tmp_path / "source" / "batch-2"
    newer_dir.mkdir()
    newer_manifest = source_manifest.model_copy(
        update={
            "batch_id": "batch-2",
            "tasks": [
                BatchTaskSpec(
                    task_id="case-failed",
                    agent_kind="desktop",
                    query="failed latest",
                    metadata={"original": "latest"},
                    sandbox=SandboxProviderSpec(provider="e2b", template="old-template"),
                )
            ],
        }
    )
    (newer_dir / "manifest.json").write_text(newer_manifest.model_dump_json(indent=2), encoding="utf-8")
    newer_batch_db = newer_dir / "batch_state.sqlite"
    newer_repository = SQLiteBatchStateRepository(newer_batch_db)
    newer_repository.initialize_manifest(newer_manifest)
    newer_repository.upsert_state(BatchTaskState(task_id="case-failed", status=BatchTaskStatus.FAILED, attempts=1))
    with newer_repository._connection:  # noqa: SLF001 - fixture needs deterministic dedupe timestamps.
        newer_repository._connection.execute("UPDATE batch_tasks SET updated_at=20 WHERE task_id='case-failed'")

    output_manifest = tmp_path / "rerun" / "manifest.json"
    report = build_rerun_manifest(
        run_root=tmp_path,
        output_manifest=output_manifest,
        rerun_batch_id="rerun-batch",
        source_batch_dbs=[batch_db, newer_batch_db],
        new_template="new-template",
        new_template_profile="new-profile",
        concurrency=40,
    )

    manifest = load_batch_manifest(output_manifest)
    assert report["deduped_task_count"] == 1
    assert [task.task_id for task in manifest.tasks] == ["case-failed"]
    assert manifest.defaults.concurrency == 40
    assert manifest.defaults.sandbox.template == "new-template"
    assert manifest.tasks[0].sandbox is not None
    assert manifest.tasks[0].sandbox.template == "new-template"
    assert manifest.tasks[0].metadata["original"] == "latest"
    assert manifest.tasks[0].metadata["template_profile"] == "new-profile"
    assert manifest.tasks[0].metadata["rerun_source"] == "non_success_collection"
    assert (
        json.loads(output_manifest.with_suffix(".json.report.json").read_text(encoding="utf-8"))["new_template"]
        == "new-template"
    )


def test_build_rerun_manifest_excludes_any_case_that_succeeded_in_any_source(tmp_path: Path) -> None:
    first_dir = tmp_path / "source" / "batch-1"
    first_dir.mkdir(parents=True)
    first_manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[
            BatchTaskSpec(task_id="case-later-success", agent_kind="desktop", query="failed first"),
            BatchTaskSpec(task_id="case-earlier-success", agent_kind="desktop", query="success first"),
            BatchTaskSpec(task_id="case-keep", agent_kind="desktop", query="failed always"),
        ],
    )
    (first_dir / "manifest.json").write_text(first_manifest.model_dump_json(indent=2), encoding="utf-8")
    first_db = first_dir / "batch_state.sqlite"
    first_repo = SQLiteBatchStateRepository(first_db)
    first_repo.initialize_manifest(first_manifest)
    first_repo.upsert_state(BatchTaskState(task_id="case-later-success", status=BatchTaskStatus.FAILED, attempts=1))
    first_repo.upsert_state(
        BatchTaskState(task_id="case-earlier-success", status=BatchTaskStatus.SUCCEEDED, attempts=1)
    )
    first_repo.upsert_state(BatchTaskState(task_id="case-keep", status=BatchTaskStatus.FAILED, attempts=1))
    with first_repo._connection:  # noqa: SLF001 - fixture needs deterministic dedupe timestamps.
        first_repo._connection.execute("UPDATE batch_tasks SET updated_at=10")

    second_dir = tmp_path / "source" / "batch-2"
    second_dir.mkdir()
    second_manifest = BatchManifest(
        batch_id="batch-2",
        tasks=[
            BatchTaskSpec(task_id="case-later-success", agent_kind="desktop", query="success later"),
            BatchTaskSpec(task_id="case-earlier-success", agent_kind="desktop", query="failed later"),
            BatchTaskSpec(task_id="case-keep", agent_kind="desktop", query="failed latest"),
        ],
    )
    (second_dir / "manifest.json").write_text(second_manifest.model_dump_json(indent=2), encoding="utf-8")
    second_db = second_dir / "batch_state.sqlite"
    second_repo = SQLiteBatchStateRepository(second_db)
    second_repo.initialize_manifest(second_manifest)
    second_repo.upsert_state(BatchTaskState(task_id="case-later-success", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    second_repo.upsert_state(BatchTaskState(task_id="case-earlier-success", status=BatchTaskStatus.FAILED, attempts=1))
    second_repo.upsert_state(BatchTaskState(task_id="case-keep", status=BatchTaskStatus.FAILED, attempts=1))
    with second_repo._connection:  # noqa: SLF001 - fixture needs deterministic dedupe timestamps.
        second_repo._connection.execute("UPDATE batch_tasks SET updated_at=20")

    output_manifest = tmp_path / "rerun" / "manifest.json"
    report = build_rerun_manifest(
        run_root=tmp_path,
        output_manifest=output_manifest,
        rerun_batch_id="rerun-batch",
        source_batch_dbs=[first_db, second_db],
        new_template="new-template",
    )

    manifest = load_batch_manifest(output_manifest)
    assert [task.task_id for task in manifest.tasks] == ["case-keep"]
    assert manifest.tasks[0].query == "failed latest"
    assert report["source_status_records"] == 6
    assert report["source_non_success_records"] == 4
    assert report["source_success_records"] == 2
    assert report["excluded_success_task_count"] == 2


def test_build_rerun_manifest_can_write_r19_case_suite_from_old_desktop_full_specs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "batch-1"
    source_dir.mkdir(parents=True)
    source_manifest = BatchManifest(
        batch_id="batch-1",
        defaults={
            "concurrency": 4,
            "sandbox": {
                "provider": "e2b",
                "template": "northcua-e2b-desktop-full",
                "resolution": [1280, 900],
                "resources": {"timeout_seconds": 7200},
            },
        },
        tasks=[
            BatchTaskSpec(task_id="case-success", agent_kind="desktop", query="success"),
            BatchTaskSpec(task_id="case-failed", agent_kind="desktop", query="failed"),
        ],
    )
    (source_dir / "manifest.json").write_text(source_manifest.model_dump_json(indent=2), encoding="utf-8")
    old_suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "old-suite",
            "controller": {
                "db_path": str(tmp_path / "old" / "controller.sqlite"),
                "blob_dir": str(tmp_path / "old" / "controller-blobs"),
                "artifact_root_dir": str(tmp_path / "old" / "agent-artifacts"),
            },
            "sandbox": {
                "provider": "e2b",
                "template": "northcua-e2b-desktop-full",
                "resolution": [1280, 900],
                "resources": {"timeout_seconds": 7200},
            },
            "prepared_sandbox_policy": {
                "enabled": True,
                "depth": "full_init",
                "buffer": 16,
                "prepare_concurrency": 5,
                "ttl_seconds": 1800,
                "claim_wait_seconds": 45,
            },
            "tasks": [
                CaseSpec(
                    task_id="case-success",
                    query="success",
                    app_profile="browser.chrome",
                    template_profile="desktop-full",
                ),
                CaseSpec(
                    task_id="case-failed",
                    query="failed",
                    app_profile="browser.chrome",
                    template_profile="desktop-full",
                    metadata={"source_run_id": "old-run"},
                ),
            ],
        }
    )
    (source_dir.parent / "case_suite.json").write_text(old_suite.model_dump_json(indent=2), encoding="utf-8")
    batch_db = source_dir / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(source_manifest)
    repository.upsert_state(BatchTaskState(task_id="case-success", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    repository.upsert_state(BatchTaskState(task_id="case-failed", status=BatchTaskStatus.FAILED, attempts=1))

    output_manifest = tmp_path / "rerun" / "manifest.json"
    output_case_suite = tmp_path / "rerun" / "suite.yaml"
    report = build_rerun_manifest(
        run_root=tmp_path,
        output_manifest=output_manifest,
        output_case_suite=output_case_suite,
        rerun_suite_id="rerun-suite",
        rerun_batch_id="rerun-batch",
        source_batch_dbs=[batch_db],
        new_template="fixture-osworld-base",
        new_template_profile="osworld-gnome-full",
        concurrency=40,
    )

    manifest = load_batch_manifest(output_manifest)
    suite_raw = yaml.safe_load(output_case_suite.read_text(encoding="utf-8"))
    assert report["output_case_suite"] == str(output_case_suite)
    assert [task.task_id for task in manifest.tasks] == ["case-failed"]
    assert manifest.defaults.sandbox.template == "fixture-osworld-base"
    assert manifest.defaults.sandbox.resolution == (1920, 1080)
    assert manifest.defaults.sandbox.resources.cpu_count == 8
    assert manifest.defaults.sandbox.resources.memory_mb == 16384
    assert suite_raw["suite_id"] == "rerun-suite"
    assert suite_raw["concurrency"] == 40
    assert suite_raw["controller"]["db_path"] is None
    assert suite_raw["controller"]["blob_dir"] is None
    assert suite_raw["controller"]["artifact_root_dir"] is None
    assert suite_raw["sandbox"]["template"] == "fixture-osworld-base"
    assert suite_raw["sandbox"]["resolution"] == [1920, 1080]
    assert suite_raw["sandbox"]["resources"]["cpu_count"] == 8
    assert suite_raw["sandbox"]["resources"]["memory_mb"] == 16384
    assert suite_raw["prepared_sandbox_policy"]["buffer"] == 50
    assert suite_raw["prepared_sandbox_policy"]["claim_wait_seconds"] == 3600.0
    assert [task["task_id"] for task in suite_raw["tasks"]] == ["case-failed"]
    assert suite_raw["tasks"][0]["template_profile"] == "osworld-gnome-full"
    assert suite_raw["tasks"][0]["metadata"]["rerun_source_status"] == "failed"
    assert suite_raw["tasks"][0]["metadata"]["template_alias"] == "fixture-osworld-base"
