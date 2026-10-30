from __future__ import annotations

import asyncio
import json
import multiprocessing
import stat
from pathlib import Path

import pytest

from northcua.agentinfra.context.observability import ArtifactContextTraceSink
from northcua.agentinfra.context.persistence import ArtifactCheckpointRepository
from northcua.agentkernel.context.domain.enums import (
    CompactionOutcome,
    CompactionTrigger,
    SummaryInvocationKind,
    SummaryTracePayloadMode,
)
from northcua.agentkernel.context.domain.models import (
    ContextCheckpoint,
    ContextCompactionEvent,
    ContextOwnerKey,
    ContextSummaryDocumentV1,
    ModelInvocationTraceEnvelope,
)


def _owner(agent: str = "root") -> ContextOwnerKey:
    return ContextOwnerKey(user_id="u", session_id="s", task_epoch="t", logical_agent_path=agent)


def _checkpoint(owner: ContextOwnerKey, revision: int) -> ContextCheckpoint:
    return ContextCheckpoint(
        checkpoint_id=f"checkpoint-{revision}",
        owner=owner,
        policy_version="v1",
        policy_fingerprint="policy",
        prompt_fingerprint="prompt",
        model_fingerprint="model",
        revision=revision,
        epoch=revision,
        covered_segment_ids=("segment-1",),
        source_digest="digest",
        summary=ContextSummaryDocumentV1(goal="goal"),
        summary_trace_id="trace",
    )


def _cas_process_worker(root: str, start: object, results: object) -> None:
    owner = _owner()
    start.wait()  # type: ignore[attr-defined]
    committed = asyncio.run(
        ArtifactCheckpointRepository(Path(root)).compare_and_swap(
            owner,
            expected_revision=0,
            checkpoint=_checkpoint(owner, 1),
        )
    )
    results.put(committed)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_artifact_repository_cas_and_restart(tmp_path: Path) -> None:
    owner = _owner()
    repository = ArtifactCheckpointRepository(tmp_path)
    async with repository.lease(owner, "digest"):
        assert await repository.compare_and_swap(owner, expected_revision=0, checkpoint=_checkpoint(owner, 1))
    assert not await repository.compare_and_swap(owner, expected_revision=0, checkpoint=_checkpoint(owner, 2))

    restarted = ArtifactCheckpointRepository(tmp_path)
    loaded = await restarted.load(owner)

    assert loaded is not None
    assert loaded.revision == 1
    assert loaded.owner == owner
    checkpoint_path = repository._checkpoint_path(owner)
    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_artifact_repository_cas_is_atomic_without_external_lease(tmp_path: Path) -> None:
    owner = _owner()
    repositories = [ArtifactCheckpointRepository(tmp_path), ArtifactCheckpointRepository(tmp_path)]

    results = await asyncio.gather(
        *(
            repository.compare_and_swap(owner, expected_revision=0, checkpoint=_checkpoint(owner, 1))
            for repository in repositories
        )
    )

    assert sorted(results) == [False, True]
    assert (await ArtifactCheckpointRepository(tmp_path).load(owner)).revision == 1


@pytest.mark.asyncio
async def test_repository_load_observes_commits_from_another_instance(tmp_path: Path) -> None:
    owner = _owner()
    first = ArtifactCheckpointRepository(tmp_path)
    second = ArtifactCheckpointRepository(tmp_path)
    assert await first.compare_and_swap(owner, expected_revision=0, checkpoint=_checkpoint(owner, 1))
    assert (await first.load(owner)).revision == 1
    assert await second.compare_and_swap(owner, expected_revision=1, checkpoint=_checkpoint(owner, 2))

    assert (await first.load(owner)).revision == 2


def test_artifact_repository_cas_is_atomic_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [context.Process(target=_cas_process_worker, args=(str(tmp_path), start, results)) for _ in range(2)]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(observed) == [False, True]


@pytest.mark.asyncio
async def test_checkpoint_namespaces_isolate_agents(tmp_path: Path) -> None:
    root_owner = _owner("root")
    child_owner = _owner("root/child")
    repository = ArtifactCheckpointRepository(tmp_path)
    async with repository.lease(root_owner, "digest"):
        await repository.compare_and_swap(root_owner, expected_revision=0, checkpoint=_checkpoint(root_owner, 1))

    assert await repository.load(child_owner) is None


@pytest.mark.asyncio
async def test_corrupt_checkpoint_is_preserved_before_safe_rebuild(tmp_path: Path) -> None:
    owner = _owner()
    repository = ArtifactCheckpointRepository(tmp_path)
    owner_dir = repository._owner_dir(owner)
    owner_dir.mkdir(parents=True)
    (owner_dir / "checkpoint.json").write_text("{broken", encoding="utf-8")

    async with repository.lease(owner, "digest"):
        assert await repository.load(owner) is None
        assert await repository.compare_and_swap(owner, expected_revision=0, checkpoint=_checkpoint(owner, 1))

    assert list(owner_dir.glob("checkpoint.corrupt.*.json"))
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in owner_dir.glob("checkpoint.corrupt.*.json"))
    assert (await repository.load(owner)).revision == 1


def test_trace_sink_separates_files_and_redacts_sensitive_payloads(tmp_path: Path) -> None:
    sink = ArtifactContextTraceSink(
        summary_trace_path=tmp_path / "context_summary_llm_trace.jsonl",
        compaction_path=tmp_path / "context_compaction.jsonl",
        payload_mode=SummaryTracePayloadMode.FULL_REDACTED,
    )
    owner = _owner()
    sink.record_summary(
        ModelInvocationTraceEnvelope(
            trace_id="trace",
            invocation_kind=SummaryInvocationKind.CONTEXT_SUMMARY,
            compaction_event_id="event",
            checkpoint_id="checkpoint",
            owner=owner,
            source_digest="digest",
            model="model",
            provider="provider",
            prompt_fingerprint="prompt",
            request={"image": "data:image/png;base64,SECRET", "encrypted_content": "SECRET"},
            response={
                "summary": "safe",
                "raw": 'authorization: Bearer abc.def {"awsSecretAccessKey":"abc123"}',
            },
            status="succeeded",
        )
    )
    sink.record_event(
        ContextCompactionEvent(
            event_id="event",
            logical_step_id="step",
            owner=owner,
            trigger=CompactionTrigger.SOFT_BUDGET,
            outcome=CompactionOutcome.CHECKPOINT_COMMITTED,
            policy_fingerprint="policy",
            status="done",
        )
    )

    summary_payload = json.loads((tmp_path / "context_summary_llm_trace.jsonl").read_text())
    audit_payload = json.loads((tmp_path / "context_compaction.jsonl").read_text())

    assert summary_payload["request"]["image"] == "[REDACTED_IMAGE_DATA_URL]"
    assert summary_payload["request"]["encrypted_content"] == "[REDACTED]"
    assert "abc.def" not in summary_payload["response"]["raw"]
    assert "abc123" not in summary_payload["response"]["raw"]
    assert audit_payload["event_id"] == "event"
    assert not (tmp_path / "llm_trace.jsonl").exists()
