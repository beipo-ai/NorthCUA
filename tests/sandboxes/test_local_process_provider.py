"""Local worker ownership and failure boundaries, without GUI access."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from northcua.batch.contracts import BatchManifest
from northcua.sandboxes.contracts import SandboxHandle, SandboxProviderSpec
from northcua.sandboxes.contracts import SandboxDeviceBinding, SandboxCapacityProbeRequest
from northcua_protocol.capabilities import Capability
from northcua.sandboxes.provider.local_process import LocalProcessSandboxProvider

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Local process batches require POSIX")


def spec(tmp_path, **overrides):
    db = tmp_path / "controller.sqlite"
    db.touch()
    return SandboxProviderSpec(
        provider="local-process",
        template="browser",
        desktop=False,
        platform="browser",
        metadata={"controller_db": str(db)},
        **overrides,
    )


@pytest.mark.asyncio
async def test_allocation_identity_and_cross_instance_cleanup(tmp_path):
    provider = LocalProcessSandboxProvider(tmp_path / "state")
    handle = await provider.create(spec(tmp_path))
    other = LocalProcessSandboxProvider(provider.root)
    connected = await other.connect(
        SandboxProviderSpec(provider="local-process", template="browser", sandbox_id=handle.sandbox_id)
    )
    assert connected == handle
    await other.terminate(connected)
    await other.terminate(connected)
    assert json.loads((provider._directory(handle) / "finished.json").read_text())["reason"] == "not_started"


@pytest.mark.asyncio
async def test_forged_and_symlink_handles_cannot_touch_other_directories(tmp_path):
    provider = LocalProcessSandboxProvider(tmp_path / "state")
    with pytest.raises(ValueError, match="Invalid"):
        await provider.terminate(SandboxHandle(provider="local-process", template="browser", sandbox_id="../elsewhere"))
    handle = await provider.create(spec(tmp_path))
    directory = provider._directory(handle)
    saved = directory.with_name("saved")
    directory.rename(saved)
    directory.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        await provider.terminate(handle)
    assert not (saved / "stop").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"desktop": True},
        {"platform": "linux"},
        {"template": "private"},
        {"resources": {"cpu_count": 2}},
        {"metadata": {}},
    ],
)
async def test_unsupported_allocations_fail_before_create(tmp_path, changes):
    provider = LocalProcessSandboxProvider(tmp_path / "state")
    invalid = SandboxProviderSpec.model_validate({**spec(tmp_path).model_dump(), **changes})
    with pytest.raises(ValueError):
        await provider.create(invalid)
    assert list(provider.root.iterdir()) == []


def test_manifest_rejects_guest_commands_before_any_work():
    with pytest.raises(ValueError, match="guest commands"):
        BatchManifest.model_validate(
            {
                "defaults": {
                    "sandbox": {
                        "provider": "local-process",
                        "template": "browser",
                        "desktop": False,
                        "platform": "browser",
                    },
                    "startup_commands": [{"command": "echo not-supported"}],
                },
                "tasks": [{"task_id": "one", "agent_kind": "web", "query": "test"}],
            }
        )


def wait_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"Missing {path}"
        time.sleep(0.05)


def make_supervisor_settings(directory, lifetime):
    directory.mkdir(mode=0o700)
    (directory / "deadline").write_text(str(time.time() + lifetime))
    (directory / "worker.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", "import time; time.sleep(120)"],
                "environment": {"PATH": os.environ["PATH"]},
            }
        )
    )


@pytest.mark.parametrize("trigger", ["stop", "expiry", "signal"])
def test_supervisor_stops_and_reaps_its_worker(tmp_path, trigger):
    directory = tmp_path / "worker"
    make_supervisor_settings(directory, 0.5 if trigger == "expiry" else 120)
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "northcua.sandboxes.provider.local_supervisor", str(directory), str(os.getpid())]
    )
    try:
        wait_file(directory / "started.json")
        pid = json.loads((directory / "started.json").read_text())["worker_pid"]
        if trigger == "stop":
            (directory / "stop").touch()
        elif trigger == "signal":
            supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=10) == 0
        receipt = json.loads((directory / "finished.json").read_text())
        assert receipt["reason"] == ("lease_expired" if trigger == "expiry" else "stopped")
        assert not (directory / "worker.json").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=10)


def test_supervisor_cleans_worker_after_owner_is_killed(tmp_path):
    directory = tmp_path / "worker"
    make_supervisor_settings(directory, 120)
    code = """import os,subprocess,sys,time
subprocess.Popen([sys.executable, '-m', 'northcua.sandboxes.provider.local_supervisor', sys.argv[1], str(os.getpid())])
time.sleep(120)
"""
    owner = subprocess.Popen([sys.executable, "-c", code, str(directory)])
    try:
        wait_file(directory / "started.json")
        pid = json.loads((directory / "started.json").read_text())["worker_pid"]
        owner.kill()
        owner.wait(timeout=5)
        wait_file(directory / "finished.json")
        assert json.loads((directory / "finished.json").read_text())["reason"] == "owner_exit"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


@pytest.mark.parametrize("early_exit", [False, True])
def test_supervisor_reaps_detached_descendant(tmp_path, early_exit):
    import psutil

    directory = tmp_path / "worker"
    make_supervisor_settings(directory, 120)
    settings = json.loads((directory / "worker.json").read_text())
    code = """import os,subprocess,sys,time,pathlib
time.sleep(0.25)
p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
pathlib.Path('descendant.pid').write_text(str(p.pid))
"""
    code += "os._exit(0)" if early_exit else "time.sleep(120)"
    settings["argv"] = [sys.executable, "-c", code]
    (directory / "worker.json").write_text(json.dumps(settings))
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "northcua.sandboxes.provider.local_supervisor", str(directory), str(os.getpid())]
    )
    child = None
    unrelated = None
    try:
        wait_file(directory / "started.json")
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        wait_file(directory / "descendant.pid")
        child = psutil.Process(int((directory / "descendant.pid").read_text()))
        if not early_exit:
            time.sleep(0.2)
            (directory / "stop").touch()
        assert supervisor.wait(timeout=15) == 0
        receipt = json.loads((directory / "finished.json").read_text())
        assert receipt["cleanup_confirmed"] is (not early_exit), receipt
        assert receipt["remaining_pids"] == [], receipt
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=15)
        if child is not None and child.is_running():
            child.kill()
        if unrelated is not None and unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=5)


def test_marker_discovery_preserves_unreadable_new_processes_and_ignores_baseline(monkeypatch):
    from types import SimpleNamespace
    import psutil
    from northcua.sandboxes.provider.local_supervisor import _remember_marked

    class Process:
        def __init__(self, pid, created):
            self.pid, self.created = pid, created

        def uids(self):
            return SimpleNamespace(real=os.getuid())

        def create_time(self):
            return self.created

        def is_running(self):
            return True

        def status(self):
            return psutil.STATUS_RUNNING

        def environ(self):
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil, "process_iter", lambda: [Process(1, 100), Process(2, 200), Process(3, 301)])
    owned = {}
    assert _remember_marked("test-marker", {(1, 100), (3, 300)}, owned) == [2, 3]
    assert owned == {}


@pytest.mark.asyncio
async def test_retired_allocation_cannot_start_a_worker(tmp_path):
    provider = LocalProcessSandboxProvider(tmp_path / "state")
    handle = await provider.create(spec(tmp_path))
    await provider.terminate(handle)
    binding = SandboxDeviceBinding(
        device_id="demo",
        display_name="demo",
        server_url="ws://127.0.0.1:1",
        platform="browser",
        capabilities={Capability.BROWSER_PAGE},
        agentkit_agent_kinds={"web"},
    )
    with pytest.raises(RuntimeError, match="retired"):
        await provider.start_device_worker(handle, binding)
    assert not (provider._directory(handle) / "worker.json").exists()


@pytest.mark.asyncio
async def test_dead_supervisor_cannot_fabricate_cleanup_success(tmp_path):
    provider = LocalProcessSandboxProvider(tmp_path / "state")
    handle = await provider.create(spec(tmp_path))
    directory = provider._directory(handle)
    (directory / "worker.json").write_text("{}")
    (directory / "started.json").write_text('{"worker_pid": 999999}')
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=5)
    provider._processes[handle.sandbox_id] = exited
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await provider.terminate(handle)
    assert not (directory / "finished.json").exists()


@pytest.mark.asyncio
async def test_launch_failure_removes_credentials_and_confirms_no_worker(tmp_path, monkeypatch):
    provider = LocalProcessSandboxProvider(tmp_path / "state")
    handle = await provider.create(spec(tmp_path))
    binding = SandboxDeviceBinding(
        device_id="demo",
        display_name="demo",
        server_url="ws://127.0.0.1:1",
        token="synthetic-test-secret",
        platform="browser",
        capabilities={Capability.BROWSER_PAGE},
        agentkit_agent_kinds={"web"},
    )

    def fail(*_args, **_kwargs):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr("northcua.sandboxes.provider.local_process.subprocess.Popen", fail)
    with pytest.raises(OSError):
        await provider.start_device_worker(handle, binding)
    await provider.terminate(handle)
    directory = provider._directory(handle)
    assert not (directory / "worker.json").exists()
    assert json.loads((directory / "finished.json").read_text())["cleanup_confirmed"] is True


@pytest.mark.asyncio
async def test_local_maintenance_scopes_equal_batch_ids_by_output_directory(tmp_path):
    from northcua.batch.maintenance.operations import BatchMaintenanceOperations
    from northcua.sandboxes.capacity_contracts import SandboxInventoryItem

    local = SandboxInventoryItem(
        sandbox_id="mine", provider="local-process", metadata={"local_batch_output_dir": str(tmp_path / "mine")}
    )
    other = local.model_copy(
        update={"sandbox_id": "other", "metadata": {"local_batch_output_dir": str(tmp_path / "other")}}
    )

    class Inventory:
        async def inventory(self, **_kwargs):
            return [local, other]

    operations = BatchMaintenanceOperations(sandboxes=Inventory())
    assert await operations._inventory(batch_dir=tmp_path / "mine", provider_id="local-process", batch_id="same") == [
        local
    ]
