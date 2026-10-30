"""High-concurrency status polling stays bounded and batched."""

from __future__ import annotations

import asyncio

import pytest

from northcua.batch.runner.controller import ControllerTaskWatcher
from northcua_protocol.payloads import TaskStatusSnapshot
from northcua_protocol.state import TaskState


class _BatchingSubmitter:
    calls: list[list[str]] = []

    def __init__(self, _url: str) -> None:
        pass

    async def __aenter__(self) -> "_BatchingSubmitter":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        _ = token
        type(self).calls.append(list(task_ids))
        return [
            TaskStatusSnapshot(
                task_id=task_id,
                trace_id=f"trace-{task_id}",
                agent_kind="desktop",
                state=TaskState.RUNNING,
            )
            for task_id in task_ids
        ]


@pytest.mark.asyncio
async def test_status_poller_batches_hundreds_of_task_waiters() -> None:
    _BatchingSubmitter.calls = []
    watcher = ControllerTaskWatcher(
        _BatchingSubmitter,
        status_poll_batch_window_seconds=0.01,
        status_poll_batch_size=500,
    )

    snapshots = await asyncio.gather(
        *[
            watcher._status_poller.snapshot(  # noqa: SLF001 - focused concurrency contract
                controller_url="ws://controller",
                token="read-token",
                task_id=f"task-{index}",
                deadline=None,
            )
            for index in range(423)
        ]
    )

    assert len(_BatchingSubmitter.calls) == 1
    assert len(_BatchingSubmitter.calls[0]) == 423
    assert {snapshot.task_id for snapshot in snapshots if snapshot is not None} == {
        f"task-{index}" for index in range(423)
    }


@pytest.mark.asyncio
async def test_status_poller_chunks_above_controller_batch_limit() -> None:
    _BatchingSubmitter.calls = []
    watcher = ControllerTaskWatcher(
        _BatchingSubmitter,
        status_poll_batch_window_seconds=0,
        status_poll_batch_size=200,
    )

    await asyncio.gather(
        *[
            watcher._status_poller.snapshot(  # noqa: SLF001 - focused concurrency contract
                controller_url="ws://controller",
                token="read-token",
                task_id=f"task-{index}",
                deadline=None,
            )
            for index in range(423)
        ]
    )

    assert [len(call) for call in _BatchingSubmitter.calls] == [200, 200, 23]


@pytest.mark.asyncio
async def test_status_poller_deduplicates_same_task_waiters() -> None:
    _BatchingSubmitter.calls = []
    watcher = ControllerTaskWatcher(
        _BatchingSubmitter,
        status_poll_batch_window_seconds=0.01,
    )

    first, second = await asyncio.gather(
        *[
            watcher._status_poller.snapshot(  # noqa: SLF001 - focused concurrency contract
                controller_url="ws://controller",
                token="read-token",
                task_id="same-task",
                deadline=None,
            )
            for _ in range(2)
        ]
    )

    assert _BatchingSubmitter.calls == [["same-task"]]
    assert first is not None and second is not None
    assert first.task_id == second.task_id == "same-task"


@pytest.mark.asyncio
async def test_status_poller_drops_canceled_waiter_before_flush() -> None:
    _BatchingSubmitter.calls = []
    watcher = ControllerTaskWatcher(
        _BatchingSubmitter,
        status_poll_batch_window_seconds=0.05,
    )
    snapshot = asyncio.create_task(
        watcher._status_poller.snapshot(  # noqa: SLF001 - focused cancellation contract
            controller_url="ws://controller",
            token="read-token",
            task_id="canceled-task",
            deadline=None,
        )
    )
    await asyncio.sleep(0)
    snapshot.cancel()
    with pytest.raises(asyncio.CancelledError):
        await snapshot
    await asyncio.sleep(0.06)

    assert _BatchingSubmitter.calls == []
    assert watcher._status_poller._pending == {}  # noqa: SLF001
    await watcher.aclose()


@pytest.mark.asyncio
async def test_status_poller_close_cancels_pending_flush_without_request() -> None:
    _BatchingSubmitter.calls = []
    watcher = ControllerTaskWatcher(
        _BatchingSubmitter,
        status_poll_batch_window_seconds=60,
    )
    snapshot = asyncio.create_task(
        watcher._status_poller.snapshot(  # noqa: SLF001 - focused shutdown contract
            controller_url="ws://controller",
            token="read-token",
            task_id="pending-task",
            deadline=None,
        )
    )
    await asyncio.sleep(0)
    await watcher.aclose()
    with pytest.raises(asyncio.CancelledError):
        await snapshot

    assert _BatchingSubmitter.calls == []
    assert watcher._status_poller._pending == {}  # noqa: SLF001
    assert watcher._status_poller._flush_tasks == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_status_poller_close_is_bounded_and_shared_when_request_ignores_cancel() -> None:
    class _NonCooperativeSubmitter(_BatchingSubmitter):
        async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
            del task_ids, token
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)
                return []

    watcher = ControllerTaskWatcher(
        _NonCooperativeSubmitter,
        status_poll_batch_window_seconds=0,
        status_poll_shutdown_timeout_seconds=0.01,
    )
    snapshot = asyncio.create_task(
        watcher._status_poller.snapshot(  # noqa: SLF001
            controller_url="ws://controller",
            token="read-token",
            task_id="stuck-task",
            deadline=None,
        )
    )
    await asyncio.sleep(0)

    first_close = asyncio.create_task(watcher.aclose())
    await asyncio.sleep(0)
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close
    await asyncio.wait_for(watcher.aclose(), timeout=0.1)

    with pytest.raises(asyncio.CancelledError):
        await snapshot
    await asyncio.sleep(0.06)
