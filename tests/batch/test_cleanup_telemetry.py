from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchTaskSpec, BatchTaskState
from northcua.batch.runner.telemetry_helpers import release_with_telemetry


class _Timer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _CleanupEvents:
    def __init__(self, *, fail_writes: bool) -> None:
        self.fail_writes = fail_writes
        self.states: list[BatchTaskState] = []

    def phase_timer(self, **kwargs: object) -> _Timer:
        _ = kwargs
        return _Timer()

    async def upsert_task_async(self, state: BatchTaskState) -> bool:
        if self.fail_writes:
            raise RuntimeError("state write failed")
        self.states.append(state)
        return True


def _config(tmp_path: Path) -> BatchRunConfig:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="Open page")
    return BatchRunConfig(manifest=BatchManifest(batch_id="cleanup-test", tasks=[task]), output_dir=tmp_path)


@pytest.mark.asyncio
async def test_release_success_survives_cleanup_status_write_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    task = config.manifest.tasks[0]
    state = BatchTaskState(task_id=task.task_id)
    released = False

    async def release(**kwargs: Any) -> None:
        nonlocal released
        _ = kwargs
        released = True

    result = await release_with_telemetry(
        release=release,
        coordinator=object(),  # type: ignore[arg-type]
        config=config,
        task=task,
        state=state,
        lease=None,
        lease_context=None,
        output_dir=tmp_path,
        attempt=1,
        events=_CleanupEvents(fail_writes=True),  # type: ignore[arg-type]
    )

    assert released
    assert result.ok
    assert result.cleanup_status == "ok"


@pytest.mark.asyncio
async def test_release_failure_survives_cleanup_status_write_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    task = config.manifest.tasks[0]
    state = BatchTaskState(task_id=task.task_id)

    async def release(**kwargs: Any) -> None:
        _ = kwargs
        raise RuntimeError("provider terminate failed")

    result = await release_with_telemetry(
        release=release,
        coordinator=object(),  # type: ignore[arg-type]
        config=config,
        task=task,
        state=state,
        lease=None,
        lease_context=None,
        output_dir=tmp_path,
        attempt=1,
        events=_CleanupEvents(fail_writes=True),  # type: ignore[arg-type]
    )

    assert not result.ok
    assert result.cleanup_status.startswith("failed:")
