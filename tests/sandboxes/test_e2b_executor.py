"""Concurrency coverage for the process-shared E2B SDK executors."""

from __future__ import annotations

import asyncio
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from northcua.sandboxes.contracts import (
    SandboxCommandResult,
    SandboxCommandSpec,
    SandboxHandle,
    SandboxLifecycleState,
    SandboxProviderSpec,
)
from northcua.sandboxes.provider import e2b_executor as e2b_executor_module
from northcua.sandboxes.provider import e2b_runtime as e2b_runtime_module
from northcua.sandboxes.provider import e2b_runtime_support as e2b_runtime_support_module
from northcua.sandboxes.provider.e2b import E2BSandboxProvider
from northcua.sandboxes.provider.e2b_executor import (
    E2BExecutorCall,
    _executor,
    _shutdown_executors_for_testing,
    cancel_e2b_reaper_campaign,
    drain_e2b_reaper,
    executor_telemetry,
    executor_workers,
    reaper_telemetry,
    submit_e2b_reaper_campaign,
    submit_e2b_reaper_fallback_job,
    submit_e2b_reaper_job,
)


@pytest.fixture(autouse=True)
def _isolated_executors(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _shutdown_executors_for_testing()
    monkeypatch.delenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", raising=False)
    monkeypatch.delenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", raising=False)
    yield
    _shutdown_executors_for_testing()


async def _wait_for(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for executor state")
        await asyncio.sleep(0.005)


def _wait_for_sync(predicate: Callable[[], bool], *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


def test_reaper_fallback_uses_one_process_shared_bounded_pool() -> None:
    release = threading.Event()
    started = 0
    started_lock = threading.Lock()

    def blocked_cleanup() -> None:
        nonlocal started
        with started_lock:
            started += 1
        assert release.wait(timeout=5)

    for index in range(300):
        submit_e2b_reaper_fallback_job(
            operation=f"fallback-{index}",
            job=blocked_cleanup,
        )

    assert _wait_for_sync(lambda: started == 4)
    fallback_threads = [
        thread for thread in threading.enumerate() if thread.name.startswith("northcua-e2b-reaper-fallback")
    ]
    assert len(fallback_threads) == 4
    assert reaper_telemetry()["reaper_fallback_running"] == 300
    assert reaper_telemetry()["reaper_fallback_workers"] == 4

    release.set()
    assert _wait_for_sync(lambda: reaper_telemetry()["reaper_fallback_running"] == 0, timeout=5)


def test_executor_worker_defaults_and_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    assert executor_workers("control") == 64
    assert executor_workers("command") == 256

    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "9999")
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "9999")

    assert executor_workers("control") == 256
    assert executor_workers("command") == 512

    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "invalid")

    assert executor_workers("control") == 64
    assert executor_workers("command") == 256


def test_provider_instances_share_process_executors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "2")

    first = E2BSandboxProvider()
    second = E2BSandboxProvider()

    assert first is not second
    assert _executor("control") is _executor("control")
    assert _executor("control")._max_workers == 2  # type: ignore[attr-defined]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_reinitializes_an_inherited_locked_executor_guard() -> None:
    read_fd, write_fd = os.pipe()
    e2b_executor_module._executor_lock.acquire()
    try:
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                _executor("control").submit(lambda: None).result(timeout=0.5)
                os.write(write_fd, b"ok")
                os._exit(0)
            except Exception:
                # The parent asserts this explicit child failure through its exit status.
                os._exit(2)
        os.close(write_fd)
        readable, _, _ = select.select([read_fd], [], [], 2.0)
    finally:
        e2b_executor_module._executor_lock.release()

    if not readable:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
        pytest.fail("forked child deadlocked on inherited executor lock")
    assert os.read(read_fd, 2) == b"ok"
    os.close(read_fd)
    _, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_executor_threads_converge_after_repeated_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "2")
    baseline = {thread.ident for thread in threading.enumerate() if thread.name.startswith("northcua-e2b-")}

    for _ in range(3):
        _executor("control").submit(lambda: None).result(timeout=0.5)
        _shutdown_executors_for_testing()

    remaining = {thread.ident for thread in threading.enumerate() if thread.name.startswith("northcua-e2b-")}
    assert remaining <= baseline


def test_executor_threads_do_not_block_normal_process_exit() -> None:
    code = """
import asyncio
from northcua.sandboxes.provider.e2b_executor import E2BExecutorCall
asyncio.run(E2BExecutorCall("command", lambda: None).run())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0, completed.stderr


def test_reaper_drain_fences_accepted_jobs() -> None:
    release = threading.Event()
    completed: list[int] = []

    assert submit_e2b_reaper_job(operation="first", job=lambda: release.wait(timeout=2))
    assert submit_e2b_reaper_job(operation="second", job=lambda: completed.append(2))

    drain_result: list[bool] = []
    drain_thread = threading.Thread(
        target=lambda: drain_result.append(drain_e2b_reaper(timeout_seconds=2)),
    )
    drain_thread.start()
    time.sleep(0.02)
    assert drain_thread.is_alive()
    release.set()
    drain_thread.join(timeout=2)

    assert drain_result == [True]
    assert completed == [2]
    assert reaper_telemetry()["reaper_completed_total"] == 2


def test_reaper_stop_waits_for_synchronous_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    assert drain_e2b_reaper(timeout_seconds=1.0, stop=True)

    def fallback() -> None:
        started.set()
        release.wait(timeout=2)
        completed.set()

    submitter = threading.Thread(
        target=lambda: submit_e2b_reaper_job(
            operation="sync-fallback",
            job=fallback,
            fallback_to_caller_thread=True,
        )
    )
    submitter.start()
    assert started.wait(timeout=1)
    submitter.join(timeout=1)
    assert not submitter.is_alive()

    drain_result: list[bool] = []
    drain = threading.Thread(target=lambda: drain_result.append(drain_e2b_reaper(timeout_seconds=2.0, stop=True)))
    drain.start()
    time.sleep(0.02)
    assert drain.is_alive()
    release.set()
    drain.join(timeout=1)

    assert completed.is_set()
    assert drain_result == [True]
    assert reaper_telemetry()["reaper_fallback_running"] == 0


def test_reaper_stop_linearizes_before_concurrent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert submit_e2b_reaper_job(operation="prime-worker", job=lambda: None)
    assert drain_e2b_reaper(timeout_seconds=1.0)
    sentinel_blocked = threading.Event()
    release_sentinel = threading.Event()
    fallback_started = threading.Event()
    original_put = e2b_executor_module._reaper_queue.put

    def blocked_put(item: object, *args: object, **kwargs: object) -> None:
        if isinstance(item, e2b_executor_module._ReaperStop):
            sentinel_blocked.set()
            assert release_sentinel.wait(timeout=2)
        original_put(item, *args, **kwargs)

    monkeypatch.setattr(e2b_executor_module._reaper_queue, "put", blocked_put)
    drain_result: list[bool] = []
    drain = threading.Thread(target=lambda: drain_result.append(drain_e2b_reaper(timeout_seconds=2.0, stop=True)))
    drain.start()
    assert sentinel_blocked.wait(timeout=1)

    submitter = threading.Thread(
        target=lambda: submit_e2b_reaper_job(
            operation="concurrent-fallback",
            job=fallback_started.set,
            fallback_to_caller_thread=True,
        )
    )
    submitter.start()
    time.sleep(0.02)
    assert not fallback_started.is_set()

    release_sentinel.set()
    drain.join(timeout=1)
    submitter.join(timeout=1)

    assert drain_result == [True]
    assert _wait_for_sync(fallback_started.is_set, timeout=1)
    assert reaper_telemetry()["reaper_fallback_running"] == 0


def test_reaper_atexit_drains_cancel_storm_cleanup(tmp_path: Path) -> None:
    marker = tmp_path / "reaped.txt"
    code = f"""
import time
from pathlib import Path
from northcua.sandboxes.provider.e2b_executor import submit_e2b_reaper_job
marker = Path({str(marker)!r})
for index in range(32):
    def cleanup(index=index):
        time.sleep(0.002)
        with marker.open("a", encoding="utf-8") as stream:
            stream.write(f"{{index}}\\n")
    assert submit_e2b_reaper_job(operation="cancelled_create", job=cleanup)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(int(line) for line in marker.read_text(encoding="utf-8").splitlines()) == list(range(32))


def test_parallel_reaper_drains_cancellation_storm_within_one_deadline() -> None:
    completed: list[int] = []
    lock = threading.Lock()

    for index in range(64):

        def cleanup(index: int = index) -> None:
            time.sleep(0.05)
            with lock:
                completed.append(index)

        assert submit_e2b_reaper_job(operation="cancelled_create", job=cleanup)

    started_at = time.monotonic()
    assert drain_e2b_reaper(timeout_seconds=1.0)
    elapsed = time.monotonic() - started_at

    assert sorted(completed) == list(range(64))
    assert elapsed < 0.75
    assert reaper_telemetry()["reaper_workers"] == 16


def test_reaper_campaign_drain_includes_delayed_jobs_without_worker_sleep() -> None:
    completed: list[int] = []

    for index in range(666):
        assert submit_e2b_reaper_campaign(
            operation="ambiguous-create",
            jobs=tuple(
                (delay, lambda index=index: (time.sleep(0.001), completed.append(index))) for delay in (0.0, 0.02, 0.04)
            ),
            dedupe_key=f"call-{index}",
        )

    assert drain_e2b_reaper(timeout_seconds=2.0)
    assert len(completed) == 666 * 3
    assert sum(thread.name == "northcua-e2b-reaper-scheduler" for thread in threading.enumerate()) == 1
    reaper_threads = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("northcua-e2b-reaper-") and not thread.name.startswith("northcua-e2b-reaper-fallback")
    ]
    assert len(reaper_threads) <= 17


def test_completed_campaign_cancels_future_passes_and_releases_drain() -> None:
    completed: list[str] = []
    key = "exact-call-token"

    def first() -> None:
        completed.append("first")
        assert cancel_e2b_reaper_campaign(key) == 2

    assert submit_e2b_reaper_campaign(
        operation="ambiguous-create",
        jobs=((0.0, first), (10.0, lambda: completed.append("second")), (20.0, lambda: completed.append("third"))),
        dedupe_key=key,
    )
    assert drain_e2b_reaper(timeout_seconds=1.0)
    assert completed == ["first"]


def test_late_create_is_reaped_after_provider_close_and_event_loop_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    scenario_finished = threading.Event()

    class _LateSandbox:
        sandbox_id = "sandbox-after-loop"

        def __init__(self) -> None:
            self.kill_calls = 0

        def kill(self, **_kwargs: object) -> None:
            self.kill_calls += 1

    sandbox = _LateSandbox()
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_PROVIDER_CLOSE_TIMEOUT_SECONDS", "0")

    def delayed_create(_spec: object) -> _LateSandbox:
        started.set()
        assert release.wait(timeout=5)
        return sandbox

    async def scenario() -> None:
        provider = E2BSandboxProvider()
        monkeypatch.setattr(provider, "_create_sync", delayed_create)
        create = asyncio.create_task(provider.create(SandboxProviderSpec(provider="e2b", template="template")))
        await asyncio.to_thread(started.wait, 2)
        create.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create
        await provider.aclose()

    def run_scenario() -> None:
        asyncio.run(scenario())
        scenario_finished.set()

    thread = threading.Thread(target=run_scenario)
    thread.start()
    assert started.wait(timeout=2)
    thread.join(timeout=2)
    assert scenario_finished.is_set()

    release.set()
    assert _wait_for_sync(lambda: sandbox.kill_calls == 1, timeout=2)
    assert sandbox.kill_calls == 1


def test_late_ambiguous_create_error_reconciles_after_event_loop_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    scenario_finished = threading.Event()
    create_call_token = ""
    killed: list[str] = []
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_PROVIDER_CLOSE_TIMEOUT_SECONDS", "0")

    def delayed_create(spec: SandboxProviderSpec) -> object:
        nonlocal create_call_token
        create_call_token = str(spec.metadata["create_call_token"])
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("The write operation timed out")

    async def scenario() -> None:
        provider = E2BSandboxProvider()
        monkeypatch.setattr(provider, "_create_sync", delayed_create)
        monkeypatch.setattr(
            provider,
            "_list_matching_orphan_sandboxes_sync",
            lambda expected: [
                type(
                    "InventoryRow",
                    (),
                    {
                        "sandbox_id": "orphan-after-loop",
                        "metadata": dict(expected),
                    },
                )()
            ],
        )
        monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
        create = asyncio.create_task(
            provider.create(
                SandboxProviderSpec(
                    provider="e2b",
                    template="template",
                    metadata={
                        "northcua": "true",
                        "batch_id": "batch-1",
                        "task_id": "task-1",
                        "attempt": "1",
                        "template": "template",
                        "create_token": "allocation-1",
                    },
                )
            )
        )
        await asyncio.to_thread(started.wait, 2)
        create.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create
        await provider.aclose()

    def run_scenario() -> None:
        asyncio.run(scenario())
        scenario_finished.set()

    thread = threading.Thread(target=run_scenario)
    thread.start()
    assert started.wait(timeout=2)
    thread.join(timeout=2)
    assert scenario_finished.is_set()

    release.set()
    assert _wait_for_sync(lambda: killed == ["orphan-after-loop"], timeout=2)
    assert create_call_token
    assert killed == ["orphan-after-loop"]


def test_late_ambiguous_create_rechecks_delayed_inventory_after_loop_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    scenario_finished = threading.Event()
    inventory_calls = 0
    killed: list[str] = []
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_PROVIDER_CLOSE_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(e2b_runtime_module, "_ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0, 0.03, 0.03))

    def delayed_create(_spec: SandboxProviderSpec) -> object:
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("The write operation timed out")

    async def scenario() -> None:
        nonlocal inventory_calls
        provider = E2BSandboxProvider()
        monkeypatch.setattr(provider, "_create_sync", delayed_create)

        def list_matching(expected: dict[str, str]) -> list[object]:
            nonlocal inventory_calls
            inventory_calls += 1
            if inventory_calls == 1:
                return []
            return [type("InventoryRow", (), {"sandbox_id": "delayed-orphan", "metadata": dict(expected)})()]

        monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
        monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
        create = asyncio.create_task(
            provider.create(
                SandboxProviderSpec(
                    provider="e2b",
                    template="template",
                    metadata={
                        "northcua": "true",
                        "batch_id": "batch-1",
                        "task_id": "task-1",
                        "attempt": "1",
                        "template": "template",
                        "create_token": "allocation-1",
                    },
                )
            )
        )
        await asyncio.to_thread(started.wait, 2)
        create.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create
        await provider.aclose()

    def run_scenario() -> None:
        asyncio.run(scenario())
        scenario_finished.set()

    thread = threading.Thread(target=run_scenario)
    thread.start()
    assert started.wait(timeout=2)
    thread.join(timeout=2)
    assert scenario_finished.is_set()
    release.set()

    assert _wait_for_sync(lambda: killed == ["delayed-orphan"], timeout=2)
    assert inventory_calls == 2


def test_late_ambiguous_create_falls_back_after_reaper_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    scenario_finished = threading.Event()
    inventory_calls = 0
    killed: list[str] = []
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_PROVIDER_CLOSE_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(e2b_runtime_module, "_ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0, 0.01))

    def delayed_create(_spec: SandboxProviderSpec) -> object:
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("The write operation timed out")

    async def scenario() -> None:
        nonlocal inventory_calls
        provider = E2BSandboxProvider()
        monkeypatch.setattr(provider, "_create_sync", delayed_create)

        def list_matching(expected: dict[str, str]) -> list[object]:
            nonlocal inventory_calls
            inventory_calls += 1
            if inventory_calls == 1:
                return []
            return [type("InventoryRow", (), {"sandbox_id": "stopped-reaper-orphan", "metadata": dict(expected)})()]

        monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
        monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
        create = asyncio.create_task(
            provider.create(
                SandboxProviderSpec(
                    provider="e2b",
                    template="template",
                    metadata={
                        "northcua": "true",
                        "batch_id": "batch-1",
                        "task_id": "task-1",
                        "attempt": "1",
                        "template": "template",
                        "create_token": "allocation-1",
                    },
                )
            )
        )
        await asyncio.to_thread(started.wait, 2)
        create.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create
        await provider.aclose()

    def run_scenario() -> None:
        asyncio.run(scenario())
        scenario_finished.set()

    thread = threading.Thread(target=run_scenario)
    thread.start()
    assert started.wait(timeout=2)
    thread.join(timeout=2)
    assert scenario_finished.is_set()
    assert drain_e2b_reaper(timeout_seconds=1.0, stop=True)

    release.set()
    assert _wait_for_sync(lambda: killed == ["stopped-reaper-orphan"], timeout=2)
    assert inventory_calls == 2
    assert reaper_telemetry()["reaper_fallback_running"] == 0
    assert reaper_telemetry()["reaper_fallback_started_total"] == 1
    assert reaper_telemetry()["reaper_fallback_completed_total"] == 1


def test_cli_sigterm_unwinds_async_command_and_drains_reaper(tmp_path: Path) -> None:
    marker = tmp_path / "sigterm-reaped.txt"
    code = f"""
import asyncio
import importlib
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from northcua.sandboxes.provider.e2b_executor import submit_e2b_reaper_job
cli_main = importlib.import_module("northcua.cli.main")
marker = Path({str(marker)!r})
class Parser:
    def parse_args(self):
        return SimpleNamespace(command_group="batch")
async def run_batch_command(_args):
    def cleanup():
        time.sleep(0.35)
        marker.write_text("reaped", encoding="utf-8")
    assert submit_e2b_reaper_job(operation="sigterm", job=cleanup)
    print("READY", flush=True)
    await asyncio.Event().wait()
fake = types.ModuleType("northcua.cli.batch_commands")
fake.run_batch_command = run_batch_command
sys.modules["northcua.cli.batch_commands"] = fake
cli_main.build_parser = Parser
cli_main.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    process.terminate()
    _, stderr = process.communicate(timeout=5)

    assert process.returncode == 128 + signal.SIGTERM, stderr
    assert marker.read_text(encoding="utf-8") == "reaped"


def test_cli_sigterm_cancels_once_and_allows_async_cleanup(tmp_path: Path) -> None:
    marker = tmp_path / "sigterm-cleanup.txt"
    code = f"""
import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
cli_main = importlib.import_module("northcua.cli.main")
marker = Path({str(marker)!r})
cli_main._SIGTERM_COMMAND_GRACE_SECONDS = 0.5
cli_main._SIGTERM_REAPER_GRACE_SECONDS = 0.2
class Parser:
    def parse_args(self):
        return SimpleNamespace(command_group="batch")
async def run_batch_command(_args):
    print("READY", flush=True)
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await asyncio.sleep(0.15)
        marker.write_text("cleaned", encoding="utf-8")
        raise
fake = types.ModuleType("northcua.cli.batch_commands")
fake.run_batch_command = run_batch_command
sys.modules["northcua.cli.batch_commands"] = fake
cli_main.build_parser = Parser
cli_main.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    process.terminate()
    _, stderr = process.communicate(timeout=3)

    assert process.returncode == 128 + signal.SIGTERM, stderr
    assert marker.read_text(encoding="utf-8") == "cleaned"


def test_cli_sigterm_has_hard_deadline_when_cancellation_is_swallowed(tmp_path: Path) -> None:
    marker = tmp_path / "sigterm-swallowed.txt"
    code = f"""
import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
cli_main = importlib.import_module("northcua.cli.main")
marker = Path({str(marker)!r})
cli_main._SIGTERM_COMMAND_GRACE_SECONDS = 0.2
cli_main._SIGTERM_REAPER_GRACE_SECONDS = 0.2
class Parser:
    def parse_args(self):
        return SimpleNamespace(command_group="batch")
async def run_batch_command(_args):
    print("READY", flush=True)
    while True:
        try:
            await asyncio.Event().wait()
        except BaseException:
            marker.write_text("swallowed", encoding="utf-8")
            while True:
                pass
fake = types.ModuleType("northcua.cli.batch_commands")
fake.run_batch_command = run_batch_command
sys.modules["northcua.cli.batch_commands"] = fake
cli_main.build_parser = Parser
cli_main.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    process.terminate()
    _, stderr = process.communicate(timeout=3)

    assert process.returncode == 128 + signal.SIGTERM, stderr
    assert marker.read_text(encoding="utf-8") == "swallowed"


def test_cli_sigterm_has_hard_deadline_when_loop_is_already_blocked() -> None:
    code = """
import importlib
import sys
import types
from types import SimpleNamespace
cli_main = importlib.import_module("northcua.cli.main")
cli_main._SIGTERM_COMMAND_GRACE_SECONDS = 0.2
cli_main._SIGTERM_REAPER_GRACE_SECONDS = 0.2
class Parser:
    def parse_args(self):
        return SimpleNamespace(command_group="batch")
async def run_batch_command(_args):
    print("READY", flush=True)
    while True:
        pass
fake = types.ModuleType("northcua.cli.batch_commands")
fake.run_batch_command = run_batch_command
sys.modules["northcua.cli.batch_commands"] = fake
cli_main.build_parser = Parser
cli_main.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    process.terminate()
    _, stderr = process.communicate(timeout=3)

    assert process.returncode == 128 + signal.SIGTERM, stderr


def test_cli_repeated_sigterm_does_not_deadlock_signal_handler() -> None:
    code = """
import importlib
import sys
import types
from types import SimpleNamespace
cli_main = importlib.import_module("northcua.cli.main")
cli_main._SIGTERM_COMMAND_GRACE_SECONDS = 0.2
cli_main._SIGTERM_REAPER_GRACE_SECONDS = 0.2
class Parser:
    def parse_args(self):
        return SimpleNamespace(command_group="batch")
async def run_batch_command(_args):
    print("READY", flush=True)
    while True:
        pass
fake = types.ModuleType("northcua.cli.batch_commands")
fake.run_batch_command = run_batch_command
sys.modules["northcua.cli.batch_commands"] = fake
cli_main.build_parser = Parser
cli_main.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    for _ in range(20):
        process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=3)

    assert process.returncode == 128 + signal.SIGTERM, stderr


def test_cli_sigterm_watchdog_bounds_rejected_reaper_fallback() -> None:
    code = """
import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from northcua.sandboxes.provider.e2b_executor import (
    drain_e2b_reaper,
    submit_e2b_reaper_fallback_job,
)
cli_main = importlib.import_module("northcua.cli.main")
cli_main._SIGTERM_COMMAND_GRACE_SECONDS = 0.2
cli_main._SIGTERM_REAPER_GRACE_SECONDS = 0.2
class Parser:
    def parse_args(self):
        return SimpleNamespace(command_group="batch")
async def run_batch_command(_args):
    assert drain_e2b_reaper(timeout_seconds=0.1, stop=True)
    print("READY", flush=True)
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        submit_e2b_reaper_fallback_job(
            operation="blocked-fallback",
            job=lambda: time.sleep(1.0),
        )
        raise
fake = types.ModuleType("northcua.cli.batch_commands")
fake.run_batch_command = run_batch_command
sys.modules["northcua.cli.batch_commands"] = fake
cli_main.build_parser = Parser
cli_main.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    started_at = time.monotonic()
    process.terminate()
    _, stderr = process.communicate(timeout=3)

    assert process.returncode == 128 + signal.SIGTERM, stderr
    assert time.monotonic() - started_at < 0.8


@pytest.mark.asyncio
async def test_executor_physical_queue_stays_bounded_across_cancellation_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "1")
    monkeypatch.setattr(e2b_executor_module, "_EXECUTOR_MAX_PENDING", {"control": 32, "command": 32})
    blocker_started = threading.Event()
    release = threading.Event()

    blocker = E2BExecutorCall("control", lambda: blocker_started.set() or release.wait(timeout=5))
    blocker_task = asyncio.create_task(blocker.run())
    await _wait_for(blocker_started.is_set)
    executor = _executor("control")

    for _wave in range(10):
        tasks = [asyncio.create_task(E2BExecutorCall("control", lambda: None).run()) for _ in range(100)]
        await asyncio.sleep(0.01)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert executor._work_queue.qsize() <= 32  # type: ignore[attr-defined]

    release.set()
    await blocker_task


@pytest.mark.asyncio
async def test_control_lane_runs_while_command_lane_is_saturated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    command_started = threading.Event()
    release_command = threading.Event()

    def blocked_command() -> str:
        command_started.set()
        release_command.wait(timeout=2)
        return "first"

    first = E2BExecutorCall("command", blocked_command)
    first_task = asyncio.create_task(first.run())
    await _wait_for(command_started.is_set)

    second_ran = threading.Event()
    second = E2BExecutorCall("command", lambda: second_ran.set() or "second")
    second_task = asyncio.create_task(second.run())
    await asyncio.sleep(0.02)
    assert second.timing().wait_state == "executor_queue"

    control = E2BExecutorCall("control", lambda: "control")
    assert await asyncio.wait_for(control.run(), timeout=0.5) == "control"

    release_command.set()
    assert await first_task == "first"
    assert await second_task == "second"
    assert second_ran.is_set()
    assert second.timing().executor_queue_seconds > 0


@pytest.mark.asyncio
async def test_executor_telemetry_tracks_bounded_lane_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(timeout=2)

    running_task = asyncio.create_task(E2BExecutorCall("command", blocked).run())
    await _wait_for(started.is_set)
    queued_task = asyncio.create_task(E2BExecutorCall("command", lambda: None).run())
    await _wait_for(lambda: executor_telemetry("command")["executor_queued"] == 1)

    active = executor_telemetry("command")
    assert active["executor_workers"] == 1
    assert active["executor_running"] == 1
    assert active["executor_queued"] == 1
    assert active["executor_submitted_total"] == 2

    queued_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued_task
    cancelled = executor_telemetry("command")
    assert cancelled["executor_queued"] == 0
    assert cancelled["executor_queued_cancelled_total"] == 1
    assert cancelled["executor_worker_claimed_waiter_cancelled_total"] == 0
    assert cancelled["executor_max_queue_seconds"] > 0

    release.set()
    await running_task
    completed = executor_telemetry("command")
    assert completed["executor_running"] == 0
    assert completed["executor_sdk_started_total"] == 1
    assert completed["executor_sdk_finished_total"] == 1
    assert completed["executor_sdk_failed_total"] == 0


@pytest.mark.asyncio
async def test_slow_sdk_call_emits_secret_free_lane_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "1")
    monkeypatch.setattr(e2b_runtime_support_module, "_EXECUTOR_TELEMETRY_DEADLINES_SECONDS", (0.01,))
    provider = E2BSandboxProvider()
    progress: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(provider, "_progress", lambda message, **fields: progress.append((message, fields)))
    started = threading.Event()
    release = threading.Event()

    def blocked() -> str:
        started.set()
        release.wait(timeout=2)
        return "secret-result"

    task = asyncio.create_task(provider._run_executor_call(lane="control", operation="telemetry_probe", call=blocked))
    await _wait_for(started.is_set)
    await _wait_for(lambda: any(message == "e2b sdk call: waiting" for message, _ in progress))
    release.set()
    assert await task == "secret-result"

    _, fields = next(item for item in progress if item[0] == "e2b sdk call: waiting")
    assert fields["wait_state"] == "sdk_call"
    assert fields["executor_workers"] == 1
    assert fields["executor_running"] == 1
    assert fields["executor_queued"] == 0
    assert fields["telemetry_deadline_seconds"] == 0.01
    assert "secret-result" not in str(progress)


@pytest.mark.asyncio
async def test_command_wait_progress_includes_lane_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    monkeypatch.setattr(e2b_runtime_module, "_BUILD_HEARTBEAT_SECONDS", 0.01)
    provider = E2BSandboxProvider()
    progress: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(provider, "_progress", lambda message, **fields: progress.append((message, fields)))
    started = threading.Event()
    release = threading.Event()

    class _Sandbox:
        sandbox_id = "sandbox-command-telemetry"

    sandbox = _Sandbox()
    provider._sandboxes[sandbox.sandbox_id] = sandbox  # type: ignore[assignment]

    def blocked(*_args: object, **_kwargs: object) -> SandboxCommandResult:
        started.set()
        release.wait(timeout=2)
        return SandboxCommandResult()

    monkeypatch.setattr(provider, "_run_command_sync", blocked)
    handle = SandboxHandle(
        sandbox_id=sandbox.sandbox_id,
        provider="e2b",
        template="template",
        state=SandboxLifecycleState.CREATED,
    )
    task = asyncio.create_task(provider.run_command(handle, SandboxCommandSpec(command="true")))
    await _wait_for(started.is_set)
    await _wait_for(lambda: any(message == "e2b command: run_command waiting" for message, _ in progress))
    release.set()
    assert (await task).ok

    _, fields = next(item for item in progress if item[0] == "e2b command: run_command waiting")
    assert fields["executor_lane"] == "command"
    assert fields["executor_workers"] == 1
    assert fields["executor_running"] == 1
    assert fields["executor_queued"] == 0
    assert "true" not in str(progress)


@pytest.mark.asyncio
async def test_queued_call_cancellation_prevents_sdk_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    first_started = threading.Event()
    release_first = threading.Event()
    queued_ran = threading.Event()

    def blocked_command() -> None:
        first_started.set()
        release_first.wait(timeout=2)

    first_task = asyncio.create_task(E2BExecutorCall("command", blocked_command).run())
    await _wait_for(first_started.is_set)
    queued_task = asyncio.create_task(E2BExecutorCall("command", queued_ran.set).run())
    await asyncio.sleep(0.02)
    queued_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued_task

    release_first.set()
    await first_task
    await asyncio.sleep(0.02)
    assert not queued_ran.is_set()


@pytest.mark.asyncio
async def test_executor_submission_queue_has_hard_bound_and_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    monkeypatch.setitem(e2b_executor_module._EXECUTOR_MAX_PENDING, "command", 32)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(timeout=3)

    running = asyncio.create_task(E2BExecutorCall("command", blocked).run())
    await _wait_for(started.is_set)
    submissions = [asyncio.create_task(E2BExecutorCall("command", lambda: None).run()) for _ in range(1000)]
    await asyncio.sleep(0.05)
    executor = _executor("command")
    assert executor._work_queue.qsize() <= 32  # type: ignore[attr-defined]

    for task in submissions:
        if not task.done():
            task.cancel()
    await asyncio.gather(*submissions, return_exceptions=True)
    assert executor_telemetry("command")["executor_submission_rejected_total"] >= 900

    release.set()
    await running
    assert _wait_for_sync(lambda: executor._work_queue.qsize() == 0, timeout=1)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_completed_late_create_bypasses_saturated_control_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "1")
    control_started = threading.Event()
    release_control = threading.Event()
    sandbox_killed = threading.Event()

    running_control = asyncio.create_task(
        E2BExecutorCall(
            "control",
            lambda: control_started.set() or release_control.wait(timeout=3),
        ).run()
    )
    await _wait_for(control_started.is_set)

    class _LateSandbox:
        sandbox_id = "late-create-control-saturated"

        def kill(self, **_kwargs: object) -> None:
            sandbox_killed.set()

    provider = E2BSandboxProvider()
    sandbox = _LateSandbox()
    create_task = asyncio.create_task(asyncio.sleep(0, result=sandbox))
    await create_task
    provider._track_late_create_cleanup(  # noqa: SLF001
        create_task,
        spec=SandboxProviderSpec(provider="e2b", template="template"),
    )

    assert await asyncio.to_thread(sandbox_killed.wait, 0.5)
    release_control.set()
    await running_control


@pytest.mark.asyncio
async def test_provider_close_uses_one_deadline_without_global_reaper_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_PROVIDER_CLOSE_TIMEOUT_SECONDS", "1")
    provider = E2BSandboxProvider()
    pending = asyncio.create_task(asyncio.sleep(5))
    provider._late_executor_completion_tasks.add(pending)  # noqa: SLF001

    started_at = time.monotonic()
    await provider.aclose()
    elapsed = time.monotonic() - started_at

    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    assert 0.9 <= elapsed < 1.35


@pytest.mark.asyncio
async def test_worker_claim_handoff_is_not_misclassified_as_queued_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    worker_claimed = threading.Event()
    enter_invoke = threading.Event()
    operation_ran = threading.Event()
    call = E2BExecutorCall("command", operation_ran.set)
    original_invoke = call._invoke

    def delayed_invoke() -> None:
        worker_claimed.set()
        enter_invoke.wait(timeout=2)
        original_invoke()

    call._invoke = delayed_invoke  # type: ignore[method-assign]
    task = asyncio.create_task(call.run())
    await _wait_for(worker_claimed.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    claimed = executor_telemetry("command")
    assert claimed["executor_queued_cancelled_total"] == 0
    assert claimed["executor_worker_claimed_waiter_cancelled_total"] == 1
    assert claimed["executor_sdk_started_total"] == 0

    enter_invoke.set()
    await _wait_for(operation_ran.is_set)
    await _wait_for(lambda: executor_telemetry("command")["executor_running"] == 0)
    finished = executor_telemetry("command")
    assert finished["executor_queued"] == 0
    assert finished["executor_sdk_started_total"] == 1
    assert finished["executor_sdk_finished_total"] == 1


@pytest.mark.asyncio
async def test_running_call_finishes_underlying_sdk_after_waiter_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def operation() -> None:
        started.set()
        release.wait(timeout=2)
        finished.set()

    task = asyncio.create_task(E2BExecutorCall("command", operation).run())
    await _wait_for(started.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cancelled = executor_telemetry("command")
    assert cancelled["executor_running"] == 1
    assert cancelled["executor_worker_claimed_waiter_cancelled_total"] == 1
    assert cancelled["executor_sdk_finished_total"] == 0

    release.set()
    await _wait_for(finished.is_set)
    await _wait_for(lambda: executor_telemetry("command")["executor_running"] == 0)
    completed = executor_telemetry("command")
    assert completed["executor_sdk_finished_total"] == 1
    assert completed["executor_worker_claimed_waiter_cancelled_total"] == 1


@pytest.mark.asyncio
async def test_cancelled_provider_command_owns_and_consumes_late_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    provider = E2BSandboxProvider()
    started = threading.Event()
    release = threading.Event()
    progress: list[tuple[str, dict[str, object]]] = []

    class _Sandbox:
        sandbox_id = "sandbox-late-command"

    sandbox = _Sandbox()
    provider._sandboxes[sandbox.sandbox_id] = sandbox  # type: ignore[assignment]
    monkeypatch.setattr(provider, "_progress", lambda message, **fields: progress.append((message, fields)))

    def fail_late(
        _sandbox: object,
        _command: SandboxCommandSpec,
        *,
        user: str | None = None,
    ) -> SandboxCommandResult:
        assert user is None
        started.set()
        release.wait(timeout=2)
        raise RuntimeError("late SDK failure")

    monkeypatch.setattr(provider, "_run_command_sync", fail_late)
    handle = SandboxHandle(
        sandbox_id=sandbox.sandbox_id,
        provider="e2b",
        template="template",
        state=SandboxLifecycleState.CREATED,
    )
    command = asyncio.create_task(provider.run_command(handle, SandboxCommandSpec(command="sleep")))
    await _wait_for(started.is_set)
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command

    assert len(provider._late_executor_completion_tasks) == 1  # noqa: SLF001
    release.set()
    await provider.aclose()

    assert provider._late_executor_completion_tasks == set()  # noqa: SLF001
    assert any(message == "e2b sdk call: late completion failed" for message, _ in progress)


@pytest.mark.asyncio
async def test_provider_terminate_is_not_starved_by_running_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_EXECUTOR_WORKERS", "1")
    provider = E2BSandboxProvider()
    command_started = threading.Event()
    release_command = threading.Event()
    killed = threading.Event()

    class _Sandbox:
        sandbox_id = "sandbox-1"

    sandbox = _Sandbox()
    provider._sandboxes[sandbox.sandbox_id] = sandbox  # type: ignore[assignment]

    def run_command(
        _sandbox: object,
        _command: SandboxCommandSpec,
        *,
        user: str | None = None,
    ) -> SandboxCommandResult:
        assert user is None
        command_started.set()
        release_command.wait(timeout=2)
        return SandboxCommandResult()

    monkeypatch.setattr(provider, "_run_command_sync", run_command)
    monkeypatch.setattr(provider, "_kill_sync", lambda _sandbox: killed.set())
    handle = SandboxHandle(
        sandbox_id=sandbox.sandbox_id,
        provider="e2b",
        template="template",
        state=SandboxLifecycleState.CREATED,
    )

    command_task = asyncio.create_task(provider.run_command(handle, SandboxCommandSpec(command="sleep")))
    await _wait_for(command_started.is_set)
    await asyncio.wait_for(provider.terminate(handle), timeout=0.5)
    assert killed.is_set()

    release_command.set()
    assert (await command_task).ok
    assert sandbox.sandbox_id not in provider._sandboxes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_progress_reports_queue_and_sdk_timing_without_operation_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_E2B_CONTROL_EXECUTOR_WORKERS", "1")
    provider = E2BSandboxProvider()
    progress: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(provider, "_progress", lambda message, **fields: progress.append((message, fields)))

    result = await provider._run_executor_call(
        lane="control",
        operation="test_operation",
        call=lambda: "sensitive-result",
    )

    assert result == "sensitive-result"
    message, fields = progress[-1]
    assert message == "e2b sdk call: completed"
    assert fields["operation"] == "test_operation"
    assert fields["executor_lane"] == "control"
    assert fields["wait_state"] == "sdk_call"
    assert fields["executor_queue_seconds"] >= 0
    assert fields["sdk_call_seconds"] >= 0
    assert fields["total_seconds"] >= 0
    assert "sensitive-result" not in str(fields)
