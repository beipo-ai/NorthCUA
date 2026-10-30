"""E2B provider fallback behavior tests."""

from __future__ import annotations

import asyncio
import base64
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from northcua.sandboxes.contracts import (
    SandboxCapacityProbeRequest,
    SandboxCommandSpec,
    SandboxHandle,
    SandboxLifecycleState,
    SandboxProviderSpec,
)
from northcua.sandboxes.provider import e2b_runtime as e2b_runtime_module
from northcua.sandboxes.provider.e2b import E2BSandboxProvider
from northcua.sandboxes.provider.e2b_executor import reaper_telemetry
from northcua.sandboxes.provider.e2b_runtime import _run_sdk_command


class _FallbackCommands:
    """Command runner double that exercises fallback branches."""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        exception: Exception | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.exception = exception
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, command: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append((command, kwargs))
        if kwargs.get("background") is True:
            raise TypeError("background is unsupported")
        if self.exception is not None:
            raise self.exception
        return SimpleNamespace(exit_code=self.exit_code, stdout=self.stdout, stderr=self.stderr)


def test_late_orphan_campaign_claims_one_reconcile_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    expected = {
        "northcua": "true",
        "batch_id": "batch-1",
        "task_id": "task-1",
        "attempt": "1",
        "template": "template",
        "create_token": "allocation-1",
        "create_call_token": "atomic-campaign",
    }
    spec = SandboxProviderSpec(provider="e2b", template="template", metadata=expected)
    kill_calls: list[str] = []
    kill_started = threading.Event()
    kill_release = threading.Event()
    monkeypatch.setattr(e2b_runtime_module, "_ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda _expected: [SimpleNamespace(sandbox_id="one-orphan", metadata=expected)],
    )

    def kill(sandbox_id: str) -> None:
        kill_calls.append(sandbox_id)
        kill_started.set()
        assert kill_release.wait(timeout=5)

    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", kill)
    provider._queue_late_create_orphan_reconcile(  # noqa: SLF001
        RuntimeError("The write operation timed out"),
        spec=spec,
    )
    assert kill_started.wait(timeout=2)
    assert kill_calls == ["one-orphan"]
    kill_release.set()
    for _ in range(200):
        if not reaper_telemetry()["reaper_campaigns_active"]:
            break
        threading.Event().wait(0.01)
    assert kill_calls == ["one-orphan"]


def test_e2b_command_request_timeout_keeps_transport_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_REQUEST_TIMEOUT_SECONDS", "2400")

    _run_sdk_command(sandbox, "true", timeout=2400)

    assert commands.calls == [("true", {"timeout": 2400, "request_timeout": 2520})]


def test_e2b_command_request_timeout_preserves_larger_operator_value(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_REQUEST_TIMEOUT_SECONDS", "3000")
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_REQUEST_TIMEOUT_CAP_SECONDS", "3000")

    _run_sdk_command(sandbox, "true", timeout=2400)

    assert commands.calls == [("true", {"timeout": 2400, "request_timeout": 3000})]


def test_e2b_command_request_timeout_caps_legacy_floor_for_short_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)
    monkeypatch.setenv("NORTHCUA_E2B_COMMAND_REQUEST_TIMEOUT_SECONDS", "2400")

    _run_sdk_command(sandbox, "true", timeout=30)

    assert commands.calls == [("true", {"timeout": 30, "request_timeout": 300})]


def test_e2b_command_request_timeout_uses_per_command_margin_without_legacy_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)
    monkeypatch.delenv("NORTHCUA_E2B_COMMAND_REQUEST_TIMEOUT_SECONDS", raising=False)

    _run_sdk_command(sandbox, "true", timeout=30)

    assert commands.calls == [("true", {"timeout": 30, "request_timeout": 180})]


class _FallbackSandbox:
    """Small sandbox stand-in with command fallback support only."""

    sandbox_id = "sandbox-1"
    id = "sandbox-1"
    stream_url = ""

    def __init__(self, commands: _FallbackCommands) -> None:
        self.commands = commands
        self.files = object()


class _RetryFiles:
    """File API double that fails once before accepting an upload."""

    def __init__(self) -> None:
        self.calls = 0
        self.content: bytes | str = b""
        self.request_timeouts: list[float | None] = []

    def write(
        self,
        _path: str,
        content: bytes | str,
        *,
        request_timeout: float | None = None,
    ) -> None:
        self.calls += 1
        self.request_timeouts.append(request_timeout)
        if self.calls == 1:
            raise TimeoutError("transient upload timeout")
        self.content = content


class _LegacyFiles:
    """Pre-request-timeout SDK file API double."""

    def __init__(self) -> None:
        self.content: bytes | str = b""

    def write(self, _path: str, content: bytes | str) -> None:
        self.content = content


class _UnsupportedFiles:
    """File API double that rejects writes to exercise shell fallback."""

    def __init__(self) -> None:
        self.calls = 0

    def write(self, _path: str, _content: bytes | str) -> None:
        self.calls += 1
        raise NotImplementedError("files.write unsupported")


class _ReadFiles:
    """File API double that records binary read mode requests."""

    def __init__(self, content: bytes | bytearray) -> None:
        self.content = content
        self.formats: list[str] = []

    def read(self, _path: str, *, format: str = "text") -> bytes | bytearray:
        self.formats.append(format)
        return self.content

    def write(self, _path: str, _content: bytes | str) -> None:
        raise AssertionError("not used")


class _RetrySandbox:
    """Sandbox stand-in with a retryable file API."""

    sandbox_id = "sandbox-retry"
    id = "sandbox-retry"
    stream_url = ""

    def __init__(self, files: _RetryFiles) -> None:
        self.commands = _FallbackCommands()
        self.files = files


class _LegacyWriteSandbox:
    """Sandbox stand-in for an older E2B SDK writer signature."""

    sandbox_id = "sandbox-legacy-write"
    id = "sandbox-legacy-write"
    stream_url = ""

    def __init__(self, files: _LegacyFiles) -> None:
        self.commands = _FallbackCommands()
        self.files = files


class _UnsupportedWriteSandbox:
    """Sandbox stand-in whose file API forces command-based upload."""

    sandbox_id = "sandbox-unsupported-write"
    id = "sandbox-unsupported-write"
    stream_url = ""

    def __init__(self, files: _UnsupportedFiles) -> None:
        self.commands = _FallbackCommands()
        self.files = files


class _ReadSandbox:
    """Sandbox stand-in with a binary-capable file API."""

    sandbox_id = "sandbox-read"
    id = "sandbox-read"
    stream_url = ""

    def __init__(self, files: _ReadFiles) -> None:
        self.commands = _FallbackCommands(stdout=str(len(files.content)))
        self.files = files


class _TerminateRetrySandbox:
    """Sandbox stand-in whose terminate call fails once with a retryable error."""

    sandbox_id = "sandbox-terminate-retry"
    id = "sandbox-terminate-retry"
    stream_url = ""

    def __init__(self) -> None:
        self.commands = _FallbackCommands()
        self.files = object()
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_calls == 1:
            raise RuntimeError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")


class _TerminateAlwaysBrokenPipeSandbox:
    """Sandbox stand-in whose terminate call always raises BrokenPipeError."""

    sandbox_id = "sandbox-terminate-broken-pipe"
    id = "sandbox-terminate-broken-pipe"
    stream_url = ""

    def __init__(self) -> None:
        self.commands = _FallbackCommands()
        self.files = object()
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        raise BrokenPipeError(32, "Broken pipe")


class _KillRecordingSandbox:
    """Sandbox stand-in used by orphan-cleanup tests."""

    sandbox_id = "orphan-1"
    id = "orphan-1"
    stream_url = ""

    def __init__(self) -> None:
        self.commands = _FallbackCommands()
        self.files = object()
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1


class _ControlRecordingSandbox(_KillRecordingSandbox):
    """Sandbox double that records control-plane request timeout kwargs."""

    def __init__(self) -> None:
        super().__init__()
        self.kill_kwargs: list[dict[str, object]] = []
        self.set_timeout_calls: list[tuple[int, dict[str, object]]] = []

    def kill(self, **kwargs: object) -> None:
        self.kill_kwargs.append(kwargs)
        super().kill()

    def set_timeout(self, timeout: int, **kwargs: object) -> None:
        self.set_timeout_calls.append((timeout, kwargs))


class _ControlRecordingSandboxClass:
    """SDK class double for create/connect request timeout tests."""

    create_calls: list[dict[str, object]] = []
    connect_calls: list[tuple[str, dict[str, object]]] = []
    sandbox = _ControlRecordingSandbox()

    @classmethod
    def create(cls, **kwargs: object) -> _ControlRecordingSandbox:
        cls.create_calls.append(kwargs)
        return cls.sandbox

    @classmethod
    def connect(cls, sandbox_id: str, **kwargs: object) -> _ControlRecordingSandbox:
        cls.connect_calls.append((sandbox_id, kwargs))
        return cls.sandbox


class CommandExitException(Exception):
    """E2B-shaped command exit exception double."""

    def __init__(self, *, exit_code: int, stdout: str, stderr: str, error: str = "") -> None:
        super().__init__(stderr)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error


def test_e2b_start_process_fallback_writes_separate_stderr_file() -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)

    result = E2BSandboxProvider()._start_process_sync(  # type: ignore[arg-type]
        sandbox,
        SandboxCommandSpec(command="python worker.py"),
    )

    assert result.ok
    fallback_command = commands.calls[-1][0]
    all_commands = "\n".join(command for command, _ in commands.calls)
    assert any(kwargs.get("background") is True for _, kwargs in commands.calls)
    assert "mkdir -p /tmp/northcua-device-logs" in all_commands
    assert ": >/tmp/northcua-device-worker.out" in all_commands
    assert ": >/tmp/northcua-device-worker.err" in all_commands
    assert ">/tmp/northcua-device-worker.out" in fallback_command
    assert "2>/tmp/northcua-device-worker.err" in fallback_command


def test_e2b_command_line_applies_cwd_and_env() -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)

    result = E2BSandboxProvider()._run_command_sync(  # type: ignore[arg-type]
        sandbox,
        SandboxCommandSpec(
            command="python worker.py",
            cwd="/tmp/work dir",
            env={"NORTHCUA_MODE": "golden demo"},
        ),
    )

    assert result.ok
    command = commands.calls[-1][0]
    script = E2BSandboxProvider()._command_script(  # type: ignore[attr-defined]
        SandboxCommandSpec(
            command="python worker.py",
            cwd="/tmp/work dir",
            env={"NORTHCUA_MODE": "golden demo"},
        )
    )
    assert command.startswith("bash -lc ")
    assert "cd '/tmp/work dir'" in script
    assert "export NORTHCUA_MODE='golden demo'" in script
    assert "python worker.py" in script


def test_e2b_run_command_normalizes_sdk_command_exit_exception() -> None:
    commands = _FallbackCommands(
        exception=CommandExitException(exit_code=7, stdout="visible-secret", stderr="failed-secret")
    )
    sandbox = _FallbackSandbox(commands)

    result = E2BSandboxProvider()._run_command_sync(  # type: ignore[arg-type]
        sandbox,
        SandboxCommandSpec(command="exit 7", redact=["secret"]),
    )

    assert result.exit_code == 7
    assert result.stdout == "visible-[REDACTED]"
    assert result.stderr == "failed-[REDACTED]"


def test_e2b_run_command_preserves_sdk_command_error_when_stderr_is_empty() -> None:
    commands = _FallbackCommands(exception=CommandExitException(exit_code=-1, stdout="", stderr="", error="boom"))
    sandbox = _FallbackSandbox(commands)

    result = E2BSandboxProvider()._run_command_sync(  # type: ignore[arg-type]
        sandbox,
        SandboxCommandSpec(command="echo boom"),
    )

    assert result.exit_code == -1
    assert result.stderr == "boom"


def test_e2b_start_process_applies_cwd_and_env_inside_background_wrapper() -> None:
    commands = _FallbackCommands()
    sandbox = _FallbackSandbox(commands)

    result = E2BSandboxProvider()._start_process_sync(  # type: ignore[arg-type]
        sandbox,
        SandboxCommandSpec(
            command="python worker.py",
            cwd="/tmp/work dir",
            env={"NORTHCUA_MODE": "golden demo"},
        ),
    )

    assert result.ok
    command = commands.calls[-1][0]
    all_commands = "\n".join(command for command, _ in commands.calls)
    script = E2BSandboxProvider()._command_script(  # type: ignore[attr-defined]
        SandboxCommandSpec(
            command="python worker.py",
            cwd="/tmp/work dir",
            env={"NORTHCUA_MODE": "golden demo"},
        )
    )
    assert "setsid bash -lc " in command
    assert "</dev/null" in command
    assert any(kwargs.get("background") is True for _, kwargs in commands.calls)
    assert "mkdir -p /tmp/northcua-device-logs" in all_commands
    assert ">/tmp/northcua-device-worker.out" in command
    assert "2>/tmp/northcua-device-worker.err" in command
    assert "cd '/tmp/work dir'" in script
    assert "export NORTHCUA_MODE='golden demo'" in script
    assert "python worker.py" in script


def test_e2b_upload_fallback_raises_on_remote_failure(tmp_path: Path) -> None:
    commands = _FallbackCommands(exit_code=1, stderr="permission denied")
    sandbox = _FallbackSandbox(commands)
    local = tmp_path / "input.txt"
    local.write_text("payload", encoding="utf-8")

    with pytest.raises(RuntimeError, match="permission denied"):
        E2BSandboxProvider()._upload_sync(sandbox, local, "/root/input.txt")  # type: ignore[arg-type]


def test_e2b_upload_falls_back_when_sdk_write_is_unsupported(tmp_path: Path) -> None:
    files = _UnsupportedFiles()
    sandbox = _UnsupportedWriteSandbox(files)
    local = tmp_path / "input.txt"
    local.write_text("payload", encoding="utf-8")

    E2BSandboxProvider()._upload_sync(sandbox, local, "/tmp/input.txt")  # type: ignore[arg-type]

    commands = [command for command, _ in sandbox.commands.calls]
    assert files.calls == 1
    assert len(commands) == 1
    assert commands[0].startswith("mkdir -p /tmp && base64 -d > /tmp/input.txt <<'NORTHCUA_UPLOAD_B64'")
    assert "cGF5bG9hZA==" in commands[0]


def test_e2b_upload_fallback_chunks_large_payload_when_sdk_write_is_unsupported(tmp_path: Path) -> None:
    files = _UnsupportedFiles()
    sandbox = _UnsupportedWriteSandbox(files)
    local = tmp_path / "input.bin"
    local.write_bytes(b"x" * (96 * 1024 + 1))

    E2BSandboxProvider()._upload_sync(sandbox, local, "/tmp/input.bin")  # type: ignore[arg-type]

    commands = [command for command, _ in sandbox.commands.calls]
    assert files.calls == 1
    assert commands[0] == "mkdir -p /tmp && : > /tmp/input.bin.northcua-upload.b64"
    assert any(command.startswith("printf %s ") for command in commands)
    assert any("base64 -d /tmp/input.bin.northcua-upload.b64 > /tmp/input.bin" in command for command in commands)


@pytest.mark.asyncio
async def test_e2b_upload_retries_transient_sdk_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("northcua.sandboxes.provider.e2b._UPLOAD_RETRY_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setenv("NORTHCUA_E2B_UPLOAD_REQUEST_TIMEOUT_SECONDS", "123")
    files = _RetryFiles()
    provider = E2BSandboxProvider()
    provider._sandboxes["sandbox-retry"] = _RetrySandbox(files)  # type: ignore[assignment]
    local = tmp_path / "input.txt"
    local.write_text("payload", encoding="utf-8")
    handle = SandboxHandle(
        sandbox_id="sandbox-retry",
        provider="e2b",
        template="template",
        state=SandboxLifecycleState.CREATED,
    )

    await provider.upload(handle, local, "/tmp/input.txt")

    assert files.calls == 2
    assert files.content == b"payload"
    assert files.request_timeouts == [123, 123]


def test_e2b_upload_supports_legacy_sdk_writer_signature(tmp_path: Path) -> None:
    files = _LegacyFiles()
    sandbox = _LegacyWriteSandbox(files)
    local = tmp_path / "input.txt"
    local.write_text("payload", encoding="utf-8")

    E2BSandboxProvider()._upload_sync(sandbox, local, "/tmp/input.txt")  # type: ignore[arg-type]

    assert files.content == b"payload"


def test_e2b_download_uses_binary_file_read_mode(tmp_path: Path) -> None:
    files = _ReadFiles(bytearray(b"\x89PNG\r\n\x1a\n\xff"))
    sandbox = _ReadSandbox(files)
    local = tmp_path / "screenshot.png"

    E2BSandboxProvider()._download_sync(sandbox, "/tmp/screenshot.png", local)  # type: ignore[arg-type]

    assert sandbox.commands.calls == []
    assert files.formats == ["bytes"]
    assert local.read_bytes() == b"\x89PNG\r\n\x1a\n\xff"


def test_e2b_control_sdk_calls_receive_specific_request_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    sandbox_cls = _ControlRecordingSandboxClass
    sandbox_cls.create_calls = []
    sandbox_cls.connect_calls = []
    sandbox_cls.sandbox = _ControlRecordingSandbox()
    monkeypatch.setenv("NORTHCUA_E2B_REQUEST_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_REQUEST_TIMEOUT_SECONDS", "61")
    monkeypatch.setenv("NORTHCUA_E2B_CONNECT_REQUEST_TIMEOUT_SECONDS", "62")
    monkeypatch.setenv("NORTHCUA_E2B_TERMINATE_REQUEST_TIMEOUT_SECONDS", "63")
    monkeypatch.setenv("NORTHCUA_E2B_SET_TIMEOUT_REQUEST_TIMEOUT_SECONDS", "64")
    monkeypatch.setenv("NORTHCUA_E2B_PROXY_URL", "http://proxy.test:3128")
    monkeypatch.setattr(provider, "_load_sdk", lambda *, desktop: sandbox_cls)
    spec = SandboxProviderSpec(provider="e2b", template="template", sandbox_id="sandbox-1")

    created = provider._create_sync(spec)
    connected = provider._connect_sync(spec)
    provider._kill_sync(created)
    provider._set_timeout_sync(created, 2400)

    assert connected is created
    assert sandbox_cls.create_calls[0]["request_timeout"] == 61
    assert sandbox_cls.create_calls[0]["proxy"] == "http://proxy.test:3128"
    assert sandbox_cls.connect_calls == [("sandbox-1", {"request_timeout": 62, "proxy": "http://proxy.test:3128"})]
    assert created.kill_kwargs == [{"request_timeout": 63}]
    assert created.set_timeout_calls == [(2400, {"request_timeout": 64})]


def test_e2b_exact_orphan_inventory_uses_call_token_and_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    list_calls: list[dict[str, object]] = []

    class FakePaginator:
        has_next = True

        def next_items(self) -> list[SimpleNamespace]:
            self.has_next = False
            return [SimpleNamespace(sandbox_id="orphan-1", metadata={})]

    class FakeSandbox:
        @classmethod
        def list(cls, **kwargs: object) -> FakePaginator:
            list_calls.append(kwargs)
            return FakePaginator()

    class FakeSandboxQuery:
        def __init__(self, **kwargs: object) -> None:
            self.metadata = kwargs["metadata"]
            self.state = kwargs["state"]

    fake_e2b = types.SimpleNamespace(
        Sandbox=FakeSandbox,
        SandboxQuery=FakeSandboxQuery,
        SandboxState=types.SimpleNamespace(RUNNING="running", PAUSED="paused"),
    )
    monkeypatch.setenv("NORTHCUA_E2B_LIST_REQUEST_TIMEOUT_SECONDS", "300")
    monkeypatch.setattr(
        "northcua.sandboxes.provider.e2b_runtime.importlib.import_module",
        lambda name: fake_e2b if name == "e2b" else __import__(name),
    )

    rows = provider._list_matching_orphan_sandboxes_sync(
        {
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": "call-1",
        }
    )

    assert [row.sandbox_id for row in rows] == ["orphan-1"]
    assert len(list_calls) == 1
    query = list_calls[0]["query"]
    assert isinstance(query, FakeSandboxQuery)
    assert query.metadata == {"create_call_token": "call-1"}
    assert query.state == ["running", "paused"]
    assert list_calls[0]["limit"] == 20
    assert list_calls[0]["request_timeout"] == 30


def test_e2b_exact_orphan_kill_treats_already_absent_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    kill_calls: list[tuple[str, dict[str, object]]] = []

    class FakeSandbox:
        @classmethod
        def kill(cls, sandbox_id: str, **kwargs: object) -> bool:
            kill_calls.append((sandbox_id, kwargs))
            return False

    monkeypatch.setattr(
        "northcua.sandboxes.provider.e2b_runtime.importlib.import_module",
        lambda name: types.SimpleNamespace(Sandbox=FakeSandbox) if name == "e2b" else __import__(name),
    )
    monkeypatch.setenv("NORTHCUA_E2B_TERMINATE_REQUEST_TIMEOUT_SECONDS", "63")

    provider._kill_sandbox_id_sync("already-absent")

    assert kill_calls == [("already-absent", {"request_timeout": 63})]


@pytest.mark.asyncio
async def test_e2b_create_hard_timeout_terminates_late_result(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    release = threading.Event()
    started = threading.Event()
    sandbox = _KillRecordingSandbox()
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")

    def delayed_create(_spec: SandboxProviderSpec) -> _KillRecordingSandbox:
        started.set()
        assert release.wait(timeout=5)
        return sandbox

    monkeypatch.setattr(provider, "_create_sync", delayed_create)

    with pytest.raises(TimeoutError, match="hard timeout of 1s"):
        await provider.create(SandboxProviderSpec(provider="e2b", template="template"))

    assert started.is_set()
    release.set()
    for _ in range(50):
        if sandbox.kill_calls:
            break
        await asyncio.sleep(0.02)
    assert sandbox.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_cancelled_create_terminates_late_result(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    release = threading.Event()
    started = threading.Event()
    sandbox = _KillRecordingSandbox()
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")

    def delayed_create(_spec: SandboxProviderSpec) -> _KillRecordingSandbox:
        started.set()
        assert release.wait(timeout=5)
        return sandbox

    monkeypatch.setattr(provider, "_create_sync", delayed_create)
    create_task = asyncio.create_task(provider.create(SandboxProviderSpec(provider="e2b", template="template")))
    await asyncio.to_thread(started.wait, 2)
    create_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await create_task

    release.set()
    for _ in range(50):
        if sandbox.kill_calls:
            break
        await asyncio.sleep(0.02)
    assert sandbox.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_cancelled_create_reconciles_late_ambiguous_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    release = threading.Event()
    started = threading.Event()
    seen_call_token = ""
    killed: list[str] = []
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_HARD_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")
    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0,))

    def delayed_create(create_spec: SandboxProviderSpec) -> SimpleNamespace:
        nonlocal seen_call_token
        seen_call_token = str(create_spec.metadata["create_call_token"])
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("The write operation timed out")

    def list_matching(expected: dict[str, str]) -> list[SimpleNamespace]:
        assert expected["create_call_token"] == seen_call_token
        return [SimpleNamespace(sandbox_id="ambiguous-orphan", metadata=dict(expected))]

    monkeypatch.setattr(provider, "_create_sync", delayed_create)
    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
    create_task = asyncio.create_task(
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
    create_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await create_task

    release.set()
    for _ in range(100):
        if killed:
            break
        await asyncio.sleep(0.01)

    assert killed == ["ambiguous-orphan"]


@pytest.mark.asyncio
async def test_e2b_completed_create_result_has_one_late_cleanup_owner() -> None:
    provider = E2BSandboxProvider()
    sandbox = _KillRecordingSandbox()
    spec = SandboxProviderSpec(provider="e2b", template="template")
    create_task = asyncio.create_task(asyncio.sleep(0, result=sandbox))
    await create_task
    assert create_task.done()

    provider._track_late_create_cleanup(create_task, spec=spec)
    provider._track_late_create_cleanup(create_task, spec=spec)

    for _ in range(50):
        if sandbox.kill_calls:
            break
        await asyncio.sleep(0.02)
    assert sandbox.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_completed_create_exception_is_reconciled_when_caller_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    killed: list[str] = []
    spec = SandboxProviderSpec(
        provider="e2b",
        template="template",
        metadata={
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": "call-1",
        },
    )

    async def fail() -> SimpleNamespace:
        raise RuntimeError("The write operation timed out")

    create_task = asyncio.create_task(fail())
    with pytest.raises(RuntimeError):
        await create_task
    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda expected: [SimpleNamespace(sandbox_id="cancel-race-orphan", metadata=dict(expected))],
    )
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0,))

    provider._track_late_create_cleanup(create_task, spec=spec)
    for _ in range(100):
        if killed:
            break
        await asyncio.sleep(0.01)
    assert killed == ["cancel-race-orphan"]


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("httpx", "WriteTimeout"),
        ("httpx", "ReadTimeout"),
        ("httpx", "PoolTimeout"),
        ("httpcore", "ConnectTimeout"),
        ("httpcore", "PoolTimeout"),
        ("httpcore", "RemoteProtocolError"),
    ],
)
def test_e2b_retryable_classifier_recognizes_typed_transport_errors(module: str, name: str) -> None:
    error_type = type(name, (Exception,), {"__module__": module})
    assert e2b_runtime_module._is_retryable_e2b_error(error_type("timed out"))


@pytest.mark.parametrize(
    "message",
    ("", "timed out", "The write operation timed out", "context deadline exceeded"),
)
def test_e2b_retryable_classifier_never_infers_transport_from_builtin_timeout(message: str) -> None:
    assert not e2b_runtime_module._is_retryable_e2b_error(TimeoutError(message))


@pytest.mark.asyncio
async def test_cancelled_async_orphan_reconcile_hands_remaining_work_to_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    inventory_calls = 0
    first_inventory_done = asyncio.Event()
    killed: list[str] = []
    spec = SandboxProviderSpec(
        provider="e2b",
        template="template",
        metadata={
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": "call-1",
        },
    )
    monkeypatch.setattr(e2b_runtime_module, "_ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0, 0.03, 0.03))

    def list_matching(expected: dict[str, str]) -> list[SimpleNamespace]:
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 1:
            first_inventory_done.set()
            return []
        return [SimpleNamespace(sandbox_id="cancelled-reconcile-orphan", metadata=dict(expected))]

    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
    cleanup = asyncio.create_task(
        provider._cleanup_possible_orphaned_create(
            spec,
            delays=e2b_runtime_module._ORPHAN_RECONCILE_DELAYS_SECONDS,
            ambiguous_error=RuntimeError("The write operation timed out"),
        )
    )
    await asyncio.wait_for(first_inventory_done.wait(), timeout=1)
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    for _ in range(200):
        if killed:
            break
        await asyncio.sleep(0.01)
    assert inventory_calls == 2
    assert killed == ["cancelled-reconcile-orphan"]


@pytest.mark.asyncio
async def test_cancelled_orphan_inventory_inflight_hands_late_error_to_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    inventory_started = threading.Event()
    inventory_release = threading.Event()
    inventory_calls = 0
    killed: list[str] = []
    spec = SandboxProviderSpec(
        provider="e2b",
        template="template",
        metadata={
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": "call-inventory",
        },
    )
    monkeypatch.setattr(e2b_runtime_module, "_ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0,))

    def list_matching(expected: dict[str, str]) -> list[SimpleNamespace]:
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 1:
            inventory_started.set()
            assert inventory_release.wait(timeout=5)
            raise RuntimeError("The write operation timed out")
        return [SimpleNamespace(sandbox_id="inventory-cancel-orphan", metadata=dict(expected))]

    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
    cleanup = asyncio.create_task(
        provider._cleanup_possible_orphaned_create(
            spec,
            ambiguous_error=RuntimeError("The write operation timed out"),
        )
    )
    await asyncio.to_thread(inventory_started.wait, 1)
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    inventory_release.set()

    for _ in range(200):
        if killed:
            break
        await asyncio.sleep(0.01)
    assert inventory_calls == 2
    assert killed == ["inventory-cancel-orphan"]


@pytest.mark.asyncio
async def test_cancelled_orphan_terminate_inflight_keeps_one_cleanup_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    terminate_started = threading.Event()
    terminate_release = threading.Event()
    kill_calls: list[str] = []
    expected = {
        "northcua": "true",
        "batch_id": "batch-1",
        "task_id": "task-1",
        "attempt": "1",
        "template": "template",
        "create_token": "allocation-1",
        "create_call_token": "call-terminate",
    }
    spec = SandboxProviderSpec(provider="e2b", template="template", metadata=expected)
    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda _expected: [SimpleNamespace(sandbox_id="terminate-cancel-orphan", metadata=expected)],
    )

    def kill(sandbox_id: str) -> None:
        kill_calls.append(sandbox_id)
        terminate_started.set()
        assert terminate_release.wait(timeout=5)

    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", kill)
    cleanup = asyncio.create_task(
        provider._cleanup_possible_orphaned_create(
            spec,
            ambiguous_error=RuntimeError("The write operation timed out"),
        )
    )
    await asyncio.to_thread(terminate_started.wait, 1)
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    terminate_release.set()

    for _ in range(100):
        if len(kill_calls) > 1:
            break
        await asyncio.sleep(0.01)
    assert kill_calls == ["terminate-cancel-orphan"]


@pytest.mark.asyncio
async def test_cancelled_orphan_terminate_late_failure_reconciles_from_create_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    terminate_started = threading.Event()
    terminate_release = threading.Event()
    inventory_calls = 0
    kill_calls = 0
    expected = {
        "northcua": "true",
        "batch_id": "batch-1",
        "task_id": "task-1",
        "attempt": "1",
        "template": "template",
        "create_token": "allocation-1",
        "create_call_token": "call-terminate-failure",
    }
    spec = SandboxProviderSpec(provider="e2b", template="template", metadata=expected)
    monkeypatch.setattr(e2b_runtime_module, "_ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0,))

    def list_matching(_expected: dict[str, str]) -> list[SimpleNamespace]:
        nonlocal inventory_calls
        inventory_calls += 1
        return [SimpleNamespace(sandbox_id="terminate-failure-orphan", metadata=expected)]

    def kill(_sandbox_id: str) -> None:
        nonlocal kill_calls
        kill_calls += 1
        if kill_calls == 1:
            terminate_started.set()
            assert terminate_release.wait(timeout=5)
            raise RuntimeError("permission denied")

    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", kill)
    cleanup = asyncio.create_task(
        provider._cleanup_possible_orphaned_create(
            spec,
            ambiguous_error=RuntimeError("The write operation timed out"),
        )
    )
    await asyncio.to_thread(terminate_started.wait, 1)
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    terminate_release.set()

    for _ in range(200):
        if kill_calls >= 2:
            break
        await asyncio.sleep(0.01)
    assert inventory_calls == 2
    assert kill_calls == 2


@pytest.mark.asyncio
async def test_e2b_late_create_cleanup_retries_retryable_terminate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    sandbox = _TerminateRetrySandbox()
    spec = SandboxProviderSpec(provider="e2b", template="template")
    create_task = asyncio.create_task(asyncio.sleep(0, result=sandbox))
    monkeypatch.setattr(
        "northcua.sandboxes.provider.e2b_runtime._LATE_CREATE_TERMINATE_RETRY_BACKOFF_SECONDS",
        (0.0,),
    )

    provider._track_late_create_cleanup(create_task, spec=spec)

    for _ in range(50):
        if sandbox.kill_calls >= 2:
            break
        await asyncio.sleep(0.02)
    assert sandbox.kill_calls == 2


@pytest.mark.asyncio
async def test_e2b_create_does_not_retry_potentially_non_idempotent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    calls = 0

    def create_once(_spec: SandboxProviderSpec) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        raise TimeoutError("ambiguous create failure")

    monkeypatch.setattr(provider, "_create_sync", create_once)

    with pytest.raises(TimeoutError, match="ambiguous create failure"):
        await provider.create(SandboxProviderSpec(provider="e2b", template="template"))

    assert calls == 1


@pytest.mark.asyncio
async def test_e2b_create_retries_retryable_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    calls = 0

    def create_once(_spec: SandboxProviderSpec) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
        return SimpleNamespace(sandbox_id="sandbox-create-retry", id="sandbox-create-retry", stream_url="")

    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._CREATE_RETRY_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0,))
    monkeypatch.setattr(provider, "_create_sync", create_once)

    handle = await provider.create(SandboxProviderSpec(provider="e2b", template="template"))

    assert calls == 2
    assert handle.sandbox_id == "sandbox-create-retry"


@pytest.mark.asyncio
async def test_e2b_create_deadline_retry_cleans_matching_create_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    calls = 0
    create_call_tokens: list[str] = []
    inventory_call_tokens: list[str] = []
    killed: list[str] = []

    def create_once(create_spec: SandboxProviderSpec) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        assert create_spec.metadata["create_token"] == "create-1"
        create_call_tokens.append(str(create_spec.metadata["create_call_token"]))
        if calls == 1:
            raise RuntimeError("context deadline exceeded: deadline_exceeded")
        return SimpleNamespace(
            sandbox_id="sandbox-after-deadline-retry",
            id="sandbox-after-deadline-retry",
            stream_url="",
        )

    def list_matching(expected: dict[str, str]) -> list[SimpleNamespace]:
        inventory_call_tokens.append(expected["create_call_token"])
        return [
            SimpleNamespace(
                sandbox_id="orphan-1",
                metadata={
                    **expected,
                },
            )
        ]

    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._CREATE_RETRY_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr(provider, "_create_sync", create_once)
    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)

    handle = await provider.create(
        SandboxProviderSpec(
            provider="e2b",
            template="template",
            metadata={
                "northcua": "true",
                "batch_id": "batch-1",
                "task_id": "task-1",
                "attempt": "1",
                "template": "template",
                "create_token": "create-1",
            },
        )
    )

    assert calls == 2
    assert handle.sandbox_id == "sandbox-after-deadline-retry"
    assert len(set(create_call_tokens)) == 2
    assert inventory_call_tokens == [create_call_tokens[0]]
    assert killed == ["orphan-1"]


@pytest.mark.asyncio
async def test_e2b_create_deadline_retry_skips_orphan_inventory_without_create_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    calls = 0

    def create_once(_spec: SandboxProviderSpec) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("context deadline exceeded: deadline_exceeded")
        return SimpleNamespace(sandbox_id="sandbox-without-cleanup", id="sandbox-without-cleanup", stream_url="")

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("orphan inventory cleanup must be skipped without create_token")

    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._CREATE_RETRY_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr(provider, "_create_sync", create_once)
    monkeypatch.setattr(provider, "_list_sandboxes_sync", fail_if_called)
    monkeypatch.setattr(provider, "_connect_sync", fail_if_called)

    handle = await provider.create(
        SandboxProviderSpec(
            provider="e2b",
            template="template",
            metadata={
                "northcua": "true",
                "batch_id": "batch-1",
                "task_id": "task-1",
                "attempt": "1",
                "template": "template",
            },
        )
    )

    assert calls == 2
    assert handle.sandbox_id == "sandbox-without-cleanup"


@pytest.mark.asyncio
async def test_e2b_create_final_retryable_failure_cleans_matching_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    create_call_tokens: list[str] = []
    inventory_calls = 0
    killed: list[str] = []
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")

    def create_once(create_spec: SandboxProviderSpec) -> SimpleNamespace:
        assert create_spec.metadata["create_token"] == "create-1"
        create_call_tokens.append(str(create_spec.metadata["create_call_token"]))
        raise RuntimeError("SSL eof")

    def list_matching(expected: dict[str, str]) -> list[SimpleNamespace]:
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 1:
            return []
        return [SimpleNamespace(sandbox_id="orphan-1", metadata=dict(expected))]

    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(provider, "_create_sync", create_once)
    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)

    with pytest.raises(RuntimeError, match="SSL eof"):
        await provider.create(
            SandboxProviderSpec(
                provider="e2b",
                template="template",
                metadata={
                    "northcua": "true",
                    "batch_id": "batch-1",
                    "task_id": "task-1",
                    "attempt": "1",
                    "template": "template",
                    "create_token": "create-1",
                },
            )
        )

    assert inventory_calls == 2
    assert killed == ["orphan-1"]
    assert len(create_call_tokens) == 1
    assert create_call_tokens[0] != "create-1"


@pytest.mark.asyncio
async def test_e2b_create_orphan_cleanup_requires_unique_create_token(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    killed: list[str] = []
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(provider, "_create_sync", lambda _spec: (_ for _ in ()).throw(RuntimeError("SSL eof")))
    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda expected: [
            SimpleNamespace(
                sandbox_id="other-run",
                metadata={key: value for key, value in expected.items() if key != "create_call_token"},
            )
        ],
    )
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)
    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0,))

    with pytest.raises(RuntimeError, match="SSL eof"):
        await provider.create(
            SandboxProviderSpec(
                provider="e2b",
                template="template",
                metadata={
                    "northcua": "true",
                    "batch_id": "batch-1",
                    "task_id": "task-1",
                    "attempt": "1",
                    "template": "template",
                    "create_token": "create-1",
                },
            )
        )

    assert killed == []


@pytest.mark.asyncio
async def test_e2b_create_orphan_cleanup_does_not_kill_later_successful_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    killed: list[str] = []
    spec = SandboxProviderSpec(
        provider="e2b",
        template="template",
        metadata={
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": "failed-call",
        },
    )

    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda _expected: [
            SimpleNamespace(sandbox_id="old-orphan", metadata=dict(spec.metadata)),
            SimpleNamespace(
                sandbox_id="later-success",
                metadata={**spec.metadata, "create_call_token": "successful-call"},
            ),
        ],
    )
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)

    await provider._cleanup_possible_orphaned_create(spec)

    assert killed == ["old-orphan"]


@pytest.mark.asyncio
async def test_e2b_create_orphan_cleanup_rechecks_after_terminate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    spec = SandboxProviderSpec(
        provider="e2b",
        template="template",
        metadata={
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": "failed-call",
        },
    )
    kill_calls = 0

    def kill_once(_sandbox_id: str) -> None:
        nonlocal kill_calls
        kill_calls += 1
        if kill_calls == 1:
            raise RuntimeError("temporary terminate failure")

    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda _expected: [SimpleNamespace(sandbox_id="old-orphan", metadata=dict(spec.metadata))],
    )
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", kill_once)

    await provider._cleanup_possible_orphaned_create(spec, delays=(0.0, 0.0))

    assert kill_calls == 2


@pytest.mark.asyncio
async def test_e2b_create_orphan_inventory_error_does_not_log_token_value(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = E2BSandboxProvider()
    secret_token = "sensitive-create-call-token"
    spec = SandboxProviderSpec(
        provider="e2b",
        template="template",
        metadata={
            "northcua": "true",
            "batch_id": "batch-1",
            "task_id": "task-1",
            "attempt": "1",
            "template": "template",
            "create_token": "allocation-1",
            "create_call_token": secret_token,
        },
    )
    monkeypatch.setattr(
        provider,
        "_list_matching_orphan_sandboxes_sync",
        lambda _expected: (_ for _ in ()).throw(RuntimeError("inventory unavailable")),
    )

    with caplog.at_level("WARNING"):
        await provider._cleanup_possible_orphaned_create(spec)

    assert secret_token not in caplog.text


@pytest.mark.asyncio
async def test_e2b_concurrent_ambiguous_creates_cleanup_only_failed_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    call_counts: dict[str, int] = {}
    orphan_by_call_token: dict[str, SimpleNamespace] = {}
    successful_ids: set[str] = set()
    killed: list[str] = []
    state_lock = threading.Lock()
    monkeypatch.setenv("NORTHCUA_E2B_CREATE_MAX_ATTEMPTS", "2")
    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._CREATE_RETRY_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr("northcua.sandboxes.provider.e2b_runtime._ORPHAN_RECONCILE_DELAYS_SECONDS", (0.0, 0.0))

    def create_once(create_spec: SandboxProviderSpec) -> SimpleNamespace:
        task_id = str(create_spec.metadata["task_id"])
        call_token = str(create_spec.metadata["create_call_token"])
        with state_lock:
            call_count = call_counts.get(task_id, 0) + 1
            call_counts[task_id] = call_count
            if call_count == 1:
                orphan_by_call_token[call_token] = SimpleNamespace(
                    sandbox_id=f"orphan-{task_id}",
                    metadata=dict(create_spec.metadata),
                )
            else:
                successful_ids.add(f"success-{task_id}")
        if call_count == 1:
            raise RuntimeError("The write operation timed out")
        return SimpleNamespace(
            sandbox_id=f"success-{task_id}",
            id=f"success-{task_id}",
            stream_url="",
        )

    inventory_calls: dict[str, int] = {}

    def list_matching(expected: dict[str, str]) -> list[SimpleNamespace]:
        call_token = expected["create_call_token"]
        with state_lock:
            count = inventory_calls.get(call_token, 0) + 1
            inventory_calls[call_token] = count
            orphan = orphan_by_call_token.get(call_token)
        return [] if count == 1 or orphan is None else [orphan]

    monkeypatch.setattr(provider, "_create_sync", create_once)
    monkeypatch.setattr(provider, "_list_matching_orphan_sandboxes_sync", list_matching)
    monkeypatch.setattr(provider, "_kill_sandbox_id_sync", killed.append)

    handles = await asyncio.gather(
        *(
            provider.create(
                SandboxProviderSpec(
                    provider="e2b",
                    template="template",
                    metadata={
                        "northcua": "true",
                        "batch_id": "stress-batch",
                        "task_id": f"task-{index}",
                        "attempt": "1",
                        "template": "template",
                        "create_token": f"allocation-{index}",
                    },
                )
            )
            for index in range(32)
        )
    )

    assert len({handle.sandbox_id for handle in handles}) == 32
    assert set(killed) == {f"orphan-task-{index}" for index in range(32)}
    assert len(killed) == 32
    assert successful_ids.isdisjoint(killed)
    assert set(call_counts.values()) == {2}


@pytest.mark.asyncio
async def test_e2b_terminate_retries_retryable_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    sandbox = _TerminateRetrySandbox()
    provider._sandboxes[sandbox.sandbox_id] = sandbox  # type: ignore[assignment]
    monkeypatch.setattr("northcua.sandboxes.provider.e2b._TERMINATE_RETRY_BACKOFF_SECONDS", (0.0,))

    await provider.terminate(
        SandboxHandle(
            sandbox_id=sandbox.sandbox_id,
            provider="e2b",
            template="template",
            state=SandboxLifecycleState.CREATED,
        )
    )

    assert sandbox.kill_calls == 2
    assert sandbox.sandbox_id not in provider._sandboxes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2b_terminate_keeps_cached_handle_after_exhausted_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = E2BSandboxProvider()
    sandbox = _TerminateAlwaysBrokenPipeSandbox()
    provider._sandboxes[sandbox.sandbox_id] = sandbox  # type: ignore[assignment]
    monkeypatch.setattr("northcua.sandboxes.provider.e2b._TERMINATE_RETRY_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setenv("NORTHCUA_E2B_TERMINATE_MAX_ATTEMPTS", "2")

    with pytest.raises(BrokenPipeError):
        await provider.terminate(
            SandboxHandle(
                sandbox_id=sandbox.sandbox_id,
                provider="e2b",
                template="template",
                state=SandboxLifecycleState.CREATED,
            )
        )

    assert sandbox.kill_calls == 2
    assert provider._sandboxes[sandbox.sandbox_id] is sandbox  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2b_capacity_probe_uses_passive_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BSandboxProvider()
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.setenv("NORTHCUA_E2B_MAX_ACTIVE_SANDBOXES", "5")
    monkeypatch.setattr(provider, "_load_inventory_sdk", lambda: None)
    monkeypatch.setattr(
        provider,
        "_list_sandboxes_sync",
        lambda: [
            {
                "sandboxID": "s1",
                "templateID": "osworld_desktop",
                "state": "running",
                "metadata": {"northcua": "true", "batch_id": "b1"},
            },
            {
                "sandboxID": "s2",
                "templateID": "other",
                "state": "paused",
                "metadata": {"owner": "manual"},
            },
        ],
    )

    result = await provider.capacity_probe(
        SandboxCapacityProbeRequest(provider="e2b", template="osworld_desktop", requested_concurrency=4)
    )

    assert result.total_active == 2
    assert result.northcua_active == 1
    assert result.matching_template_active == 1
    assert result.max_active_sandboxes == 5
    assert result.available_sandboxes == 3
    assert result.recommended_concurrency == 3
