"""Sandbox lifecycle file upload coverage."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from northcua.sandboxes.contracts import (
    SandboxCommandResult,
    SandboxCommandSpec,
    SandboxFileDeliveryMode,
    SandboxFileUploadSpec,
    SandboxHandle,
    SandboxProviderSpec,
)
from northcua.sandboxes.lifecycle.manager import SandboxLifecycleManager
from northcua.sandboxes.provider.fake import FakeSandboxProvider


class _SlowCreateFakeProvider(FakeSandboxProvider):
    """Fake provider whose create call can be completed after cancellation."""

    def __init__(self) -> None:
        super().__init__()
        self.create_started = asyncio.Event()
        self.finish_create = asyncio.Event()

    async def create(self, spec: SandboxProviderSpec) -> SandboxHandle:
        self.create_started.set()
        await self.finish_create.wait()
        return await super().create(spec)


class _WindowReadinessFailureProvider(FakeSandboxProvider):
    async def run_command(self, handle: SandboxHandle, command: SandboxCommandSpec) -> SandboxCommandResult:
        self.commands.append((handle.sandbox_id, command))
        return SandboxCommandResult(
            exit_code=1,
            stderr="window readiness timed out after process appeared: app never mapped a window",
        )


class _LongOutputFailureProvider(FakeSandboxProvider):
    async def run_command(self, handle: SandboxHandle, command: SandboxCommandSpec) -> SandboxCommandResult:
        self.commands.append((handle.sandbox_id, command))
        return SandboxCommandResult(
            exit_code=1,
            stderr="compose-root-cause\n" + ("middle\n" * 300) + "traceback-tail",
        )


class _SymlinkParentProvider(FakeSandboxProvider):
    async def run_command(self, handle: SandboxHandle, command: SandboxCommandSpec) -> SandboxCommandResult:
        if command.label == "sandbox-upload-symlink-guard":
            self.commands.append((handle.sandbox_id, command))
            return SandboxCommandResult(exit_code=1, stderr="sandbox upload path contains symlink: /tmp/northcua")
        return await super().run_command(handle, command)


class _AlwaysSuccessfulCommandProvider(FakeSandboxProvider):
    async def run_command(self, handle: SandboxHandle, command: SandboxCommandSpec) -> SandboxCommandResult:
        self.commands.append((handle.sandbox_id, command))
        return SandboxCommandResult(stdout="ok")


@pytest.mark.asyncio
async def test_lifecycle_releases_acquired_sandbox_when_custody_callback_fails() -> None:
    provider = FakeSandboxProvider()

    def fail_custody_write(handle: SandboxHandle) -> None:
        assert handle.sandbox_id == "fake-sandbox-1"
        raise OSError("synthetic custody write failure")

    with pytest.raises(OSError, match="custody write failure"):
        await SandboxLifecycleManager(provider).acquire(
            SandboxProviderSpec(provider="fake", template="ubuntu"),
            startup_commands=[],
            on_handle_acquired=fail_custody_write,
        )

    assert provider.terminated == ["fake-sandbox-1"]


@pytest.mark.asyncio
async def test_lifecycle_uploads_files_before_startup_commands(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    source = tmp_path / "fixture.txt"
    source.write_text("payload", encoding="utf-8")
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[SandboxFileUploadSpec(local_path=source, remote_path="/tmp/northcua/fixture.txt", mode="0644")],
    )

    handle = await SandboxLifecycleManager(provider).acquire(
        spec,
        startup_commands=[SandboxCommandSpec(command="cat /tmp/northcua/fixture.txt")],
    )

    assert provider.uploads == [(handle.sandbox_id, source.resolve(), "/tmp/northcua/fixture.txt")]
    commands = [command for _, command in provider.commands]
    assert commands[0].label == "sandbox-upload-symlink-guard"
    assert 'if [ -L "$northcua_path" ]' in commands[0].command
    assert commands[1].command == "mkdir -p -- /tmp/northcua"
    assert commands[2].label == "sandbox-upload-destination-symlink-guard"
    assert commands[3].command.endswith("chmod 0644 /tmp/northcua/fixture.txt")
    assert commands[-1].command == "cat /tmp/northcua/fixture.txt"


@pytest.mark.asyncio
async def test_lifecycle_verifies_upload_checksum_before_startup(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    source = tmp_path / "fixture.txt"
    source.write_text("payload", encoding="utf-8")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(
                local_path=source,
                remote_path="/tmp/northcua/fixture.txt",
                sha256=checksum,
            )
        ],
    )

    await SandboxLifecycleManager(provider).acquire(
        spec,
        startup_commands=[SandboxCommandSpec(command="cat /tmp/northcua/fixture.txt")],
    )

    command_texts = [command.command for _, command in provider.commands]
    checksum_command = next(command for command in command_texts if "sha256sum -c -" in command)
    assert checksum in checksum_command
    assert command_texts.index(checksum_command) < command_texts.index("cat /tmp/northcua/fixture.txt")


@pytest.mark.asyncio
async def test_lifecycle_applies_parent_and_file_ownership_around_upload(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    source = tmp_path / "fixture.txt"
    source.write_text("payload", encoding="utf-8")
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(
                local_path=source,
                remote_path="/home/user/project/fixture.txt",
                mode="0640",
                owner="user",
                group="users",
                parent_mode="0750",
                parent_owner="user",
                parent_group="users",
            )
        ],
    )

    handle = await SandboxLifecycleManager(provider).acquire(spec, startup_commands=[])

    assert provider.uploads == [(handle.sandbox_id, source.resolve(), "/home/user/project/fixture.txt")]
    commands = [command for _, command in provider.commands]
    assert commands[0].label == "sandbox-upload-symlink-guard"
    assert commands[1].command == "mkdir -p -- /home/user/project"
    assert commands[4].label == "sandbox-upload-destination-symlink-guard"
    mutation_commands = [command for command in commands[2:] if not command.label]
    assert [command.command.rsplit("; ", maxsplit=1)[-1] for command in mutation_commands] == [
        "chown user:users /home/user/project",
        "chmod 0750 /home/user/project",
        "chown user:users /home/user/project/fixture.txt",
        "chmod 0640 /home/user/project/fixture.txt",
    ]
    assert all('if [ -L "$northcua_path" ]' in command.command for command in mutation_commands)


@pytest.mark.asyncio
async def test_lifecycle_never_uploads_non_guest_visibility(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(
                local_path=tmp_path / "missing-evaluator.txt",
                remote_path="/tmp/northcua/evaluator.txt",
                visibility="evaluator",
            ),
            SandboxFileUploadSpec(
                local_path=tmp_path / "missing-host.txt",
                remote_path="/tmp/northcua/host.txt",
                visibility="host",
            ),
        ],
    )

    await SandboxLifecycleManager(provider).acquire(spec, startup_commands=[])

    assert provider.uploads == []
    assert provider.commands == []
    assert provider.terminated == []


@pytest.mark.asyncio
async def test_lifecycle_fails_closed_before_upload_when_parent_component_is_symlink(tmp_path: Path) -> None:
    provider = _SymlinkParentProvider()
    source = tmp_path / "fixture.txt"
    source.write_text("payload", encoding="utf-8")
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[SandboxFileUploadSpec(local_path=source, remote_path="/tmp/northcua/fixture.txt")],
    )

    with pytest.raises(RuntimeError, match="symlink guard failed"):
        await SandboxLifecycleManager(provider).acquire(spec, startup_commands=[])

    assert provider.uploads == []
    assert provider.terminated == ["fake-sandbox-1"]


@pytest.mark.asyncio
async def test_lifecycle_stages_content_addressed_guest_download_without_host_upload_or_proxy() -> None:
    provider = _AlwaysSuccessfulCommandProvider()
    stages: list[tuple[str, dict[str, object]]] = []
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(
                remote_path="/tmp/northcua/large.bin",
                delivery=SandboxFileDeliveryMode.GUEST_DOWNLOAD,
                source_url="https://artifacts.example.test/blobs/large",
                source_host="artifacts.example.test",
                source_port=443,
                source_address="8.8.8.8",
                sha256="a" * 64,
                size_bytes=100 * 1024 * 1024,
                intended_app="media.vlc",
                host_verification="endpoint_and_size_verified",
            )
        ],
    )

    await SandboxLifecycleManager(provider).acquire(
        spec,
        startup_commands=[SandboxCommandSpec(command="echo init", label="case-init")],
        stage_sink=lambda stage, payload: stages.append((stage, payload)),
    )

    assert provider.uploads == []
    staging = next(command for _, command in provider.commands if command.label == "sandbox-large-artifact-staging")
    download = next(command for _, command in provider.commands if command.label == "sandbox-guest-download")
    finalize = next(command for _, command in provider.commands if command.label == "sandbox-large-artifact-finalize")
    assert "chown root:root" in staging.command
    assert "chmod 0700" in staging.command
    assert "sandbox artifact ancestry is not root-owned" in staging.command
    assert "curl --disable --fail" in download.command
    assert "--noproxy '*'" in download.command
    assert str(100 * 1024 * 1024) in download.command
    assert "--resolve artifacts.example.test:443:8.8.8.8" in download.command
    assert "--write-out '%{http_code}'" in download.command
    assert "--location" not in download.command
    assert "--proto-redir" not in download.command
    assert "--max-redirs" not in download.command
    assert download.env == {
        "ALL_PROXY": "",
        "FTP_PROXY": "",
        "HTTPS_PROXY": "",
        "HTTP_PROXY": "",
        "NO_PROXY": "",
        "all_proxy": "",
        "ftp_proxy": "",
        "https_proxy": "",
        "http_proxy": "",
        "no_proxy": "",
    }
    assert 'unset "$northcua_proxy_var"' in download.command
    assert "sha256sum -c -" not in download.command
    assert "sha256sum -c -" in finalize.command
    assert 'test "$(stat -c %h -- "$northcua_artifact")" -eq 1' in finalize.command
    assert 'mv -fT -- "$northcua_artifact" "$northcua_final"' in finalize.command
    assert download.redact == ["https://artifacts.example.test/blobs/large"]
    commands = [command for _, command in provider.commands]
    assert (
        commands.index(staging)
        < commands.index(download)
        < commands.index(finalize)
        < next(index for index, command in enumerate(commands) if command.label == "case-init")
    )
    evidence = next(payload for stage, payload in stages if stage == "large_artifact_delivery")
    assert evidence["status"] == "succeeded"
    assert evidence["app_profile"] == "media.vlc"
    assert evidence["fixture_remote_path"] == "/tmp/northcua/large.bin"
    assert evidence["host_verification"] == "endpoint_and_size_verified"
    assert evidence["guest_verification"] == "sha256_and_size_verified"
    assert evidence["verified_before_init"] is True


@pytest.mark.asyncio
async def test_lifecycle_stages_content_addressed_chunked_upload(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    source = tmp_path / "large.bin"
    source.write_bytes(b"0123456789")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(
                local_path=source,
                remote_path="/tmp/northcua/large.bin",
                delivery=SandboxFileDeliveryMode.CHUNKED_UPLOAD,
                sha256=checksum,
                size_bytes=source.stat().st_size,
                chunk_size_bytes=4,
                intended_app="media.vlc",
                host_verification="sha256_and_size_verified",
            )
        ],
    )

    await SandboxLifecycleManager(provider).acquire(spec, startup_commands=[])

    destination_key = hashlib.sha256(b"/tmp/northcua/large.bin").hexdigest()[:12]
    staging_dir = f"/tmp/northcua/.northcua-stage-{checksum[:16]}-{destination_key}"
    remote_chunks = [remote for _, _, remote in provider.uploads]
    assert remote_chunks == [
        f"{staging_dir}/chunks/chunk-000000000000",
        f"{staging_dir}/chunks/chunk-000000000001",
        f"{staging_dir}/chunks/chunk-000000000002",
    ]
    assert all(remote != "/tmp/northcua/large.bin" for remote in remote_chunks)
    staging = next(command for _, command in provider.commands if command.label == "sandbox-large-artifact-staging")
    assemble = next(command for _, command in provider.commands if command.label == "sandbox-upload-chunk-assemble")
    finalize = next(command for _, command in provider.commands if command.label == "sandbox-large-artifact-finalize")
    assert staging_dir in staging.command
    assert "chown root:root" in staging.command
    assert "chmod 0700" in staging.command
    assert staging_dir in assemble.command
    assert checksum not in assemble.command
    assert checksum in finalize.command
    assert "wc -c" in finalize.command
    assert 'mv -fT -- "$northcua_artifact" "$northcua_final"' in finalize.command


@pytest.mark.asyncio
async def test_large_artifact_finalize_defers_parent_and_file_ownership() -> None:
    provider = _AlwaysSuccessfulCommandProvider()
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(
                remote_path="/tmp/northcua/project/large.bin",
                mode="0640",
                owner="user",
                group="users",
                parent_mode="0750",
                parent_owner="user",
                parent_group="users",
                delivery=SandboxFileDeliveryMode.GUEST_DOWNLOAD,
                source_url="https://artifacts.example.test/blobs/large",
                source_host="artifacts.example.test",
                source_port=443,
                source_address="8.8.8.8",
                sha256="a" * 64,
                size_bytes=1024,
                intended_app="media.vlc",
                host_verification="endpoint_and_size_verified",
            )
        ],
    )

    await SandboxLifecycleManager(provider).acquire(spec, startup_commands=[])

    commands = [command for _, command in provider.commands]
    finalize_index = next(
        index for index, command in enumerate(commands) if command.label == "sandbox-large-artifact-finalize"
    )
    assert all("chown user" not in command.command for command in commands[:finalize_index])
    finalize = commands[finalize_index]
    assert 'chown user:users -- "$northcua_artifact"' in finalize.command
    assert 'chmod 0640 -- "$northcua_artifact"' in finalize.command
    assert 'chown user:users -- "$northcua_parent"' in finalize.command
    assert 'chmod 0750 -- "$northcua_parent"' in finalize.command
    rename_index = finalize.command.index('mv -fT -- "$northcua_artifact" "$northcua_final"')
    assert finalize.command.index('chown user:users -- "$northcua_artifact"') < rename_index
    assert rename_index < finalize.command.index('chown user:users -- "$northcua_parent"')
    assert finalize.command.count("sandbox artifact ancestry is not root-owned") == 2


@pytest.mark.asyncio
async def test_lifecycle_startup_failure_message_is_bucketable() -> None:
    provider = _WindowReadinessFailureProvider()

    with pytest.raises(RuntimeError) as exc_info:
        await SandboxLifecycleManager(provider).acquire(
            SandboxProviderSpec(provider="fake", template="ubuntu"),
            startup_commands=[SandboxCommandSpec(command="open app", label="catalog-open-file")],
        )

    message = str(exc_info.value)
    assert "label='catalog-open-file'" in message
    assert "failure_kind=window_readiness_timeout_after_process_appeared" in message
    assert "stderr_tail='window readiness timed out after process appeared" in message
    assert provider.terminated == ["fake-sandbox-1"]


@pytest.mark.asyncio
async def test_lifecycle_long_failure_preserves_output_head_and_tail() -> None:
    provider = _LongOutputFailureProvider()

    with pytest.raises(RuntimeError) as exc_info:
        await SandboxLifecycleManager(provider).acquire(
            SandboxProviderSpec(provider="fake", template="ubuntu"),
            startup_commands=[SandboxCommandSpec(command="start services", label="start-osworld2-services")],
        )

    message = str(exc_info.value)
    assert "stderr_head='compose-root-cause" in message
    assert "traceback-tail'" in message
    assert provider.terminated == ["fake-sandbox-1"]


@pytest.mark.asyncio
async def test_lifecycle_releases_created_sandbox_when_upload_source_is_missing(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    spec = SandboxProviderSpec(
        provider="fake",
        template="ubuntu",
        file_uploads=[
            SandboxFileUploadSpec(local_path=tmp_path / "missing.txt", remote_path="/tmp/northcua/missing.txt")
        ],
    )

    with pytest.raises(RuntimeError, match="upload source"):
        await SandboxLifecycleManager(provider).acquire(spec, startup_commands=[])

    assert provider.terminated == ["fake-sandbox-1"]


@pytest.mark.asyncio
async def test_lifecycle_releases_sandbox_when_create_finishes_after_cancellation() -> None:
    provider = _SlowCreateFakeProvider()
    task = asyncio.create_task(
        SandboxLifecycleManager(provider).acquire(
            SandboxProviderSpec(provider="fake", template="ubuntu"),
            startup_commands=[],
        )
    )
    await provider.create_started.wait()

    task.cancel()
    provider.finish_create.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert provider.terminated == ["fake-sandbox-1"]


@pytest.mark.asyncio
async def test_lifecycle_cancellation_before_create_semaphore_does_not_create_sandbox() -> None:
    provider = _SlowCreateFakeProvider()
    create_semaphore = asyncio.Semaphore(0)
    task = asyncio.create_task(
        SandboxLifecycleManager(provider).acquire(
            SandboxProviderSpec(provider="fake", template="ubuntu"),
            startup_commands=[],
            create_semaphore=create_semaphore,
        )
    )
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert not provider.create_started.is_set()
    assert provider.terminated == []


def test_sandbox_file_upload_mode_requires_octal_digits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="octal"):
        SandboxFileUploadSpec(local_path=tmp_path / "fixture.txt", remote_path="/tmp/fixture.txt", mode="9999")


def test_sandbox_file_upload_normalizes_destination_and_rejects_traversal(tmp_path: Path) -> None:
    upload = SandboxFileUploadSpec(
        local_path=tmp_path / "fixture.txt",
        remote_path="/tmp//northcua/./fixture.txt",
    )

    assert upload.remote_path == "/tmp/northcua/fixture.txt"
    with pytest.raises(ValueError, match="traversal"):
        SandboxFileUploadSpec(
            local_path=tmp_path / "fixture.txt",
            remote_path="/tmp/northcua/../escape.txt",
        )
    with pytest.raises(ValueError, match="unsafe characters"):
        SandboxFileUploadSpec(
            local_path=tmp_path / "fixture.txt",
            remote_path="/tmp/northcua\\escape.txt",
        )


def test_content_addressed_delivery_contract_fails_closed_without_required_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source endpoint"):
        SandboxFileUploadSpec(
            remote_path="/tmp/northcua/large.bin",
            delivery="guest_download",
            source_url="https://artifacts.example.test/large.bin",
        )
    with pytest.raises(ValueError, match="sha256 and positive size_bytes"):
        SandboxFileUploadSpec(
            remote_path="/tmp/northcua/large.bin",
            delivery="guest_download",
            source_url="https://artifacts.example.test/large.bin",
            source_host="artifacts.example.test",
            source_port=443,
            source_address="8.8.8.8",
        )
    with pytest.raises(ValueError, match="require local_path"):
        SandboxFileUploadSpec(
            remote_path="/tmp/northcua/large.bin",
            delivery="chunked_upload",
            sha256="a" * 64,
            size_bytes=1,
        )
