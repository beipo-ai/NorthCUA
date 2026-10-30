"""Executor isolation tests for asynchronous repository helpers."""

from __future__ import annotations

import asyncio
import os
import select
import signal
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from northcua.batch.state import async_repository


@pytest.fixture(autouse=True)
def _reset_prepared_executor() -> Iterator[None]:
    async_repository._shutdown_prepared_repository_executor_for_testing()
    yield
    async_repository._shutdown_prepared_repository_executor_for_testing()


@pytest.mark.asyncio
async def test_prepared_repository_call_is_not_blocked_by_default_executor() -> None:
    loop = asyncio.get_running_loop()
    default_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="blocked-default")
    loop.set_default_executor(default_executor)
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def block_default_executor() -> None:
        blocker_started.set()
        release_blocker.wait(timeout=5.0)

    blocker = asyncio.create_task(asyncio.to_thread(block_default_executor))
    try:
        for _ in range(100):
            if blocker_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert blocker_started.is_set()

        thread_name = await asyncio.wait_for(
            async_repository.prepared_repository_call(lambda: threading.current_thread().name),
            timeout=0.5,
        )

        assert thread_name.startswith("northcua-prepared-repository")
    finally:
        release_blocker.set()
        await blocker
        default_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_prepared_repository_calls_are_serialized() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first_operation() -> str:
        first_started.set()
        release_first.wait(timeout=5.0)
        return "first"

    def second_operation() -> str:
        second_started.set()
        return "second"

    first = asyncio.create_task(async_repository.prepared_repository_call(first_operation))
    for _ in range(100):
        if first_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert first_started.is_set()

    second = asyncio.create_task(async_repository.prepared_repository_call(second_operation))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    assert await first == "first"
    assert await second == "second"


@pytest.mark.asyncio
async def test_generic_repository_call_keeps_using_default_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    calls: list[object] = []

    async def fake_to_thread(operation: object) -> object:
        calls.append(operation)
        return sentinel

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    assert await async_repository.repository_call(lambda: None) is sentinel
    assert len(calls) == 1


def test_prepared_repository_instances_share_process_executor() -> None:
    first = async_repository._prepared_repository_executor_for_process()
    second = async_repository._prepared_repository_executor_for_process()

    assert first is second
    assert first._max_workers == 1  # type: ignore[attr-defined]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_prepared_repository_executor_resets_in_forked_child() -> None:
    read_fd, write_fd = os.pipe()
    async_repository._prepared_executor_lock.acquire()
    try:
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                executor = async_repository._prepared_repository_executor_for_process()
                executor.submit(lambda: None).result(timeout=0.5)
                os.write(write_fd, b"ok")
                os._exit(0)
            except BaseException:
                os._exit(2)
        os.close(write_fd)
        readable, _, _ = select.select([read_fd], [], [], 2.0)
    finally:
        async_repository._prepared_executor_lock.release()

    if not readable:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
        pytest.fail("forked child deadlocked on inherited prepared executor lock")
    assert os.read(read_fd, 2) == b"ok"
    os.close(read_fd)
    _, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_prepared_repository_threads_converge_after_shutdown() -> None:
    prefix = "northcua-prepared-repository"
    baseline = {thread.ident for thread in threading.enumerate() if thread.name.startswith(prefix)}

    for _ in range(3):
        executor = async_repository._prepared_repository_executor_for_process()
        executor.submit(lambda: None).result(timeout=0.5)
        async_repository._shutdown_prepared_repository_executor_for_testing()

    remaining = {thread.ident for thread in threading.enumerate() if thread.name.startswith(prefix)}
    assert remaining <= baseline


@pytest.mark.asyncio
async def test_cancelled_queued_prepared_call_does_not_execute() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    queued_executed = threading.Event()

    def first_operation() -> None:
        first_started.set()
        release_first.wait(timeout=5.0)

    first = asyncio.create_task(async_repository.prepared_repository_call(first_operation))
    for _ in range(100):
        if first_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert first_started.is_set()

    queued = asyncio.create_task(async_repository.prepared_repository_call(lambda: queued_executed.set()))
    await asyncio.sleep(0.01)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    release_first.set()
    await first
    await asyncio.sleep(0.01)
    assert not queued_executed.is_set()


@pytest.mark.asyncio
async def test_cancelled_running_prepared_call_finishes_before_next_operation() -> None:
    running_started = threading.Event()
    release_running = threading.Event()
    order: list[str] = []

    def running_operation() -> None:
        running_started.set()
        release_running.wait(timeout=5.0)
        order.append("running-finished")

    running = asyncio.create_task(async_repository.prepared_repository_call(running_operation))
    for _ in range(100):
        if running_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert running_started.is_set()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    next_call = asyncio.create_task(async_repository.prepared_repository_call(lambda: order.append("next")))
    await asyncio.sleep(0.01)
    assert order == []

    release_running.set()
    await next_call
    assert order == ["running-finished", "next"]
