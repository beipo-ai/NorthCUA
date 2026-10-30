from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from northcua.sandboxes.contracts import SandboxProviderSpec, SandboxTemplateSpec
from northcua.sandboxes.operations import SandboxOperations
from northcua.sandboxes.provider.e2b_template_runtime import E2BTemplateRuntimeMixin
from northcua.sandboxes.template.doctor import SandboxTemplateDoctor
from northcua.sandboxes.template.loader import (
    RUNTIME_ARCHIVE_SHA256_ENV,
    RUNTIME_ARCHIVE_URL_ENV,
    _apply_runtime_source_overrides,
    _resolve_copy_sources,
    load_template_spec,
)


def test_template_spec_supports_base_template() -> None:
    spec = load_template_spec("examples/templates/desktop.yaml")
    assert spec.base_template == "desktop"
    assert spec.alias == "northcua-desktop"


def test_safe_hk_tos_upload_accepts_current_sdk_request_timeout() -> None:
    parameters = list(inspect.signature(E2BTemplateRuntimeMixin._safe_hk_tos_upload_file).parameters)

    assert parameters[-3:] == ["gzip", "stack_trace", "request_timeout"]


def test_safe_hk_tos_upload_streams_and_rewinds_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    e2b_template_utils = pytest.importorskip("e2b.template.utils")
    import httpx

    class TrackingArchive(io.BytesIO):
        def __init__(self, payload: bytes) -> None:
            super().__init__(payload)
            self.read_sizes: list[int] = []
            self.was_closed = False

        def read(self, size: int = -1) -> bytes:
            assert size > 0, "archive must never be materialized with an unbounded read"
            self.read_sizes.append(size)
            return super().read(size)

        def close(self) -> None:
            self.was_closed = True
            super().close()

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class FakeAsyncClient:
        def __init__(self, **_: object) -> None:
            self.statuses = iter((503, 204))
            self.payloads: list[bytes] = []
            clients.append(self)

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def put(self, _: str, *, headers: dict[str, str], content: object) -> FakeResponse:
            chunks = [chunk async for chunk in content]  # type: ignore[attr-defined]
            payload = b"".join(chunks)
            assert headers["Content-Length"] == str(len(payload))
            self.payloads.append(payload)
            return FakeResponse(next(self.statuses))

    payload = b"x" * (2 * 1024 * 1024 + 17)
    archive = TrackingArchive(payload)
    clients: list[FakeAsyncClient] = []
    monkeypatch.setattr(e2b_template_utils, "tar_file_stream", lambda *_args: archive)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    original_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _delay: original_sleep(0))
    asyncio.run(
        E2BTemplateRuntimeMixin._safe_hk_tos_upload_file(
            object(),
            ".",
            ".",
            "https://bucket.tos-s3-cn-hongkong.ivolces.com/archive?signature=redacted",
            [],
            False,
            True,
            None,
            1800,
        )
    )

    assert clients[0].payloads == [payload, payload]
    assert archive.read_sizes
    assert set(archive.read_sizes) == {1024 * 1024}
    assert archive.was_closed is True


def test_e2b_template_alias_rejects_name_tag_separator() -> None:
    with pytest.raises(ValueError, match="must not contain ':'"):
        SandboxTemplateSpec.model_validate({"provider": "e2b", "alias": "northcua_e2b:office-web"})

    spec = SandboxTemplateSpec.model_validate(
        {"provider": "e2b", "alias": "northcua-e2b-office-web", "base_template": "e2b_desktop:base"}
    )

    assert spec.base_template == "e2b_desktop:base"


def test_e2b_task_provider_rejects_privileged_osworld_v2_seed_alias() -> None:
    for template in (
        "northcua-e2b-osworld-v2-overlay-seed-20260715-r36",
        "5ao1na7b5pzo3tqbuodh",
        "lx6n2crezt7lbh7oyr45",
        "t2t60gk116nvkvk5yl6r",
        "uy5kojf3yra068mcdtjd",
        "mk9tqumjfsiotzy9px5s",
        "a3cefoqnhstm5aow54ae",
        "95yyo17n9fvhlk7hcd01",
        "vc5q8fiqsosz7zeug40w",
        "wss1wgae7w03ma1i2d40",
        "484i686qsy8o6coy8vu1",
        "3ha35i28tsxnn4mdkuve",
        "r9pi1wn4va7tcgh4hjgl",
        "nxrfbnqnlrg8fu5w1u1g",
        "nrjzyxy9w6k5u0fchmr6",
        "02wekmavogtgzxyyubhw",
        "aq88a3jsg4e8o4m6fb5v",
        "feycztfh8lxqk4if5gmh",
        "tm0k7s6qkqegxa41ipap",
        "df0sm826re26p1t09bjk",
        "nqe0bnmz4mvbho03w1rt",
        "me3vazd8lwx9ftfnr3a4",
        "impbjctir9smn48v8i7s",
        "57zq325jpuf0txsok7up",
        "b2y4ehnqo4t6h3359afj",
    ):
        with pytest.raises(ValueError, match="build seed templates cannot be routed"):
            SandboxProviderSpec.model_validate({"provider": "e2b", "template": template})


def test_base_runtime_template_declares_doctor_dependencies() -> None:
    spec = load_template_spec("examples/templates/desktop.yaml")
    packages = {package for layer in spec.layers for package in layer.packages}
    assert {"curl", "python3-tk", "xdotool", "scrot", "wmctrl"} <= packages


def test_public_template_installs_the_checked_release_wheels() -> None:
    for kind in ("desktop", "web"):
        spec = load_template_spec(f"examples/templates/{kind}.yaml")
        assert any(layer.files == {"wheelhouse": "/opt/northcua/wheels"} for layer in spec.layers)
        assert not any(layer.kind == "git" for layer in spec.layers)
        assert any("/opt/northcua/wheels/*.whl" in layer.command for layer in spec.layers)


def test_base_runtime_build_doctor_does_not_require_desktop_display() -> None:
    spec = load_template_spec("examples/templates/desktop.yaml")
    assert any("doctor --metadata-only" in layer.command for layer in spec.layers)


def test_template_git_layer_requires_remote_source_and_absolute_destination() -> None:
    with pytest.raises(ValueError, match="http"):
        SandboxTemplateSpec.model_validate(
            {"provider": "e2b", "alias": "bad", "layers": [{"kind": "git", "source": "git@host:repo"}]}
        )

    with pytest.raises(ValueError, match="credentials"):
        SandboxTemplateSpec.model_validate(
            {
                "provider": "e2b",
                "alias": "bad",
                "layers": [{"kind": "git", "source": "https://user:token@example.com/repo.git", "destination": "/src"}],
            }
        )

    with pytest.raises(ValueError, match="absolute"):
        SandboxTemplateSpec.model_validate(
            {
                "provider": "e2b",
                "alias": "bad",
                "layers": [{"kind": "git", "source": "https://example.com/repo.git", "destination": "src"}],
            }
        )


def test_runtime_source_override_applies_only_to_runtime_git_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "source-fixture",
            "layers": [
                {
                    "kind": "git",
                    "source": "https://example.com/runtime.git",
                    "ref": "main",
                    "destination": "/opt/northcua/source",
                },
            ],
        }
    )
    monkeypatch.setenv("NORTHCUA_TEMPLATE_RUNTIME_SOURCE", "https://example.com/NorthCUA.git")
    monkeypatch.setenv("NORTHCUA_TEMPLATE_RUNTIME_REF", "feature/e2b-batch-runner")

    overridden = _apply_runtime_source_overrides(spec)
    runtime_sources = [
        layer for layer in overridden.layers if layer.kind == "git" and layer.destination == "/opt/northcua/source"
    ]

    assert len(runtime_sources) == 1
    assert runtime_sources[0].source == "https://example.com/NorthCUA.git"
    assert runtime_sources[0].ref == "feature/e2b-batch-runner"


def test_runtime_archive_override_converts_runtime_source_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "source-fixture",
            "layers": [
                {
                    "kind": "git",
                    "source": "https://example.com/runtime.git",
                    "ref": "main",
                    "destination": "/opt/northcua/source",
                },
            ],
        }
    )
    checksum = "a" * 64
    monkeypatch.setenv(RUNTIME_ARCHIVE_URL_ENV, "https://example.com/runtime.tar.gz")
    monkeypatch.setenv(RUNTIME_ARCHIVE_SHA256_ENV, checksum)

    overridden = _apply_runtime_source_overrides(spec)
    runtime_sources = [
        layer for layer in overridden.layers if layer.kind == "archive" and layer.destination == "/opt/northcua/source"
    ]

    assert len(runtime_sources) == 1
    assert runtime_sources[0].source_env == RUNTIME_ARCHIVE_URL_ENV
    assert runtime_sources[0].sha256 == checksum
    assert runtime_sources[0].strip_components == 1


def test_archive_source_command_uses_env_reference_without_url_value() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "archive-test",
            "layers": [
                {
                    "kind": "archive",
                    "source_env": RUNTIME_ARCHIVE_URL_ENV,
                    "destination": "/opt/northcua/source",
                    "sha256": "b" * 64,
                    "strip_components": 1,
                }
            ],
        }
    )

    command = E2BTemplateRuntimeMixin._archive_source_command(spec.layers[0])

    assert RUNTIME_ARCHIVE_URL_ENV in command
    assert "https://example.com/runtime.tar.gz" not in command
    assert "curl --http1.1" in command
    assert "--retry-all-errors" in command
    assert "--max-time 3600" in command
    assert "mkdir -p /opt/northcua" in command
    assert "sha256sum -c -" in command


def test_hk_tos_upload_rewrite_preserves_signed_host_and_query() -> None:
    source = (
        "https://e2b-build-hk.tos-s3-cn-hongkong.ivolces.com/object/key"
        "?X-Tos-Credential=credential-value&X-Tos-Signature=ultra-secret-value"
    )
    rewritten, signed_host = E2BTemplateRuntimeMixin._rewrite_hk_tos_upload_url(source)
    assert signed_host == "e2b-build-hk.tos-s3-cn-hongkong.ivolces.com"
    assert rewritten == (
        "https://e2b-build-hk.tos-s3-cn-hongkong.volces.com/object/key"
        "?X-Tos-Credential=credential-value&X-Tos-Signature=ultra-secret-value"
    )
    redacted = E2BTemplateRuntimeMixin._redact(source)
    assert "credential-value" not in redacted
    assert "ultra-secret-value" not in redacted


def test_e2b_template_build_cache_policy_defaults_to_full_rebuild() -> None:
    default_spec = SandboxTemplateSpec.model_validate({"provider": "e2b", "alias": "default-cache-policy"})
    cached_spec = SandboxTemplateSpec.model_validate(
        {"provider": "e2b", "alias": "cached-build", "metadata": {"e2b_skip_cache": False}}
    )

    assert E2BTemplateRuntimeMixin._skip_build_cache(default_spec) is True
    assert E2BTemplateRuntimeMixin._skip_build_cache(cached_spec) is False


def test_e2b_template_build_cache_policy_rejects_non_boolean() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {"provider": "e2b", "alias": "bad-cache-policy", "metadata": {"e2b_skip_cache": "false"}}
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        E2BTemplateRuntimeMixin._skip_build_cache(spec)


def test_e2b_explicit_root_run_user_avoids_a_user_layer_contract() -> None:
    explicit = SandboxTemplateSpec.model_validate(
        {"provider": "e2b", "alias": "explicit-root", "metadata": {"e2b_explicit_run_user": "root"}}
    )
    default = SandboxTemplateSpec.model_validate({"provider": "e2b", "alias": "default-user-layer"})

    assert E2BTemplateRuntimeMixin._explicit_run_user(explicit) == "root"
    assert E2BTemplateRuntimeMixin._explicit_run_user(default) is None

    invalid = SandboxTemplateSpec.model_validate(
        {"provider": "e2b", "alias": "bad-explicit-user", "metadata": {"e2b_explicit_run_user": "user"}}
    )
    with pytest.raises(ValueError, match="supports only root"):
        E2BTemplateRuntimeMixin._explicit_run_user(invalid)


def test_e2b_run_batch_sizes_include_image_layers() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "image-run-layer",
            "metadata": {"e2b_run_batch_sizes": [1]},
            "layers": [{"kind": "run", "command": "true"}],
            "image": {"layers": [{"kind": "run", "command": "touch /bypass"}]},
        }
    )

    with pytest.raises(ValueError, match="account for every run layer"):
        E2BTemplateRuntimeMixin._run_batch_sizes(spec)


def test_e2b_template_run_batches_account_for_every_run_layer() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "cache-sensitive-run-batches",
            "metadata": {"e2b_run_batch_sizes": [1, 2, 1]},
            "layers": [
                {"kind": "run", "command": "true"},
                {"kind": "copy", "files": {"source": "/tmp/source"}},
                {"kind": "run", "command": "true"},
                {"kind": "run", "command": "true"},
                {"kind": "run", "command": "true"},
            ],
        }
    )

    assert E2BTemplateRuntimeMixin._run_batch_sizes(spec) == (1, 2, 1)


@pytest.mark.parametrize("value", [[], [0], [1, True], [1, 1]])
def test_e2b_template_run_batches_reject_invalid_boundaries(value: list[object]) -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "bad-run-batches",
            "metadata": {"e2b_run_batch_sizes": value},
            "layers": [{"kind": "run", "command": "true"}],
        }
    )

    with pytest.raises(ValueError, match="e2b_run_batch_sizes"):
        E2BTemplateRuntimeMixin._run_batch_sizes(spec)


def test_e2b_template_run_batches_reject_command_kinds_with_implicit_batching() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "implicit-command-batch",
            "metadata": {"e2b_run_batch_sizes": [1]},
            "layers": [
                {"kind": "apt", "packages": ["curl"]},
                {"kind": "run", "command": "true"},
            ],
        }
    )

    with pytest.raises(ValueError, match="only supports run/copy/expose"):
        E2BTemplateRuntimeMixin._run_batch_sizes(spec)


def test_e2b_template_startup_commands_are_explicitly_paired() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "startup-contract",
            "metadata": {"e2b_start_cmd": "systemctl start app", "e2b_ready_cmd": "systemctl is-active app"},
        }
    )

    assert E2BTemplateRuntimeMixin._startup_commands(spec) == (
        "systemctl start app",
        "systemctl is-active app",
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"e2b_start_cmd": "true"},
        {"e2b_ready_cmd": "true"},
        {"e2b_start_cmd": "", "e2b_ready_cmd": "true"},
        {"e2b_start_cmd": "true", "e2b_ready_cmd": "bad\0command"},
    ],
)
def test_e2b_template_startup_commands_reject_partial_or_unsafe_values(metadata: dict[str, str]) -> None:
    spec = SandboxTemplateSpec.model_validate({"provider": "e2b", "alias": "bad-startup", "metadata": metadata})

    with pytest.raises(ValueError, match="must be paired non-empty strings"):
        E2BTemplateRuntimeMixin._startup_commands(spec)


@pytest.mark.parametrize(
    "source",
    [
        "http://e2b-build-hk.tos-s3-cn-hongkong.ivolces.com/key?signature=value",
        "https://e2b-build-hk.tos-s3-cn-hongkong.volces.com/key?signature=value",
        "https://e2b-build-hk.tos-s3-cn-hongkong.ivolces.com/key",
        "https://user:password@e2b-build-hk.tos-s3-cn-hongkong.ivolces.com/key?signature=value",
    ],
)
def test_hk_tos_upload_rewrite_rejects_non_allowlisted_urls(source: str) -> None:
    with pytest.raises(RuntimeError, match="allowlisted"):
        E2BTemplateRuntimeMixin._rewrite_hk_tos_upload_url(source)


def test_archive_source_command_allows_large_artifact_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "archive-test",
            "layers": [
                {
                    "kind": "archive",
                    "source_env": RUNTIME_ARCHIVE_URL_ENV,
                    "destination": "/opt/northcua/source",
                    "sha256": "b" * 64,
                    "strip_components": 1,
                }
            ],
        }
    )
    monkeypatch.setenv("NORTHCUA_TEMPLATE_ARCHIVE_MAX_TIME_SECONDS", "7200")
    monkeypatch.setenv("NORTHCUA_TEMPLATE_ARCHIVE_CONNECT_TIMEOUT_SECONDS", "30")

    command = E2BTemplateRuntimeMixin._archive_source_command(spec.layers[0])

    assert "--retry-max-time 7200" in command
    assert "--max-time 7200" in command
    assert "--connect-timeout 30" in command


def test_e2b_managed_python_base_selector_uses_sdk_python_image() -> None:
    class FakeTemplateRoot:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def from_python_image(self, version: str) -> object:
            self.calls.append(("python", version))
            return object()

    root = FakeTemplateRoot()

    E2BTemplateRuntimeMixin._template_from_base_selector(
        root,
        base_template="",
        base_image="e2b-python:3.12",
    )

    assert root.calls == [("python", "3.12")]


def test_e2b_managed_base_kind_is_reported_without_registry_pull() -> None:
    assert E2BTemplateRuntimeMixin._base_kind(base_template="", base_image="e2b-python:3.12") == "e2b-python"
    assert E2BTemplateRuntimeMixin._base_kind(base_template="", base_image="ubuntu:22.04") == "registry-image"


@pytest.mark.asyncio
async def test_e2b_build_combines_managed_base_context_batches_user_and_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBuilder:
        def __init__(self) -> None:
            self._template = SimpleNamespace(_file_context_path="")
            self.run_calls: list[tuple[list[str], str | None]] = []
            self.startup: tuple[str, str] | None = None
            self.set_users: list[str] = []

        def copy(self, _src: str, _dest: str) -> FakeBuilder:
            return self

        def run_cmd(self, command: str | list[str], user: str | None = None) -> FakeBuilder:
            self.run_calls.append(([command] if isinstance(command, str) else command, user))
            return self

        def set_envs(self, _envs: dict[str, str]) -> FakeBuilder:
            return self

        def set_user(self, user: str) -> FakeBuilder:
            self.set_users.append(user)
            return self

        def set_start_cmd(self, start_cmd: str, ready_cmd: str) -> FakeBuilder:
            self.startup = (start_cmd, ready_cmd)
            return self

    class FakeAsyncTemplate:
        builder = FakeBuilder()
        selector_calls: list[tuple[str, str]] = []
        build_kwargs: dict[str, object] = {}

        def from_python_image(self, version: str) -> FakeBuilder:
            self.selector_calls.append(("python", version))
            return self.builder

        @classmethod
        async def build(cls, _builder: FakeBuilder, **kwargs: object) -> None:
            cls.build_kwargs = kwargs

    real_import_module = importlib.import_module

    def import_module(name: str) -> object:
        if name == "e2b":
            return SimpleNamespace(AsyncTemplate=FakeAsyncTemplate)
        return real_import_module(name)

    monkeypatch.setattr("northcua.sandboxes.provider.e2b_template_runtime.importlib.import_module", import_module)
    monkeypatch.setenv("NORTHCUA_TEST_MANAGED_BASE", "e2b-python:3.12")
    monkeypatch.setenv("NORTHCUA_TEST_BUILD_CONTEXT", str(tmp_path))
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "managed-integration",
            "base_image_env": "NORTHCUA_TEST_MANAGED_BASE",
            "build_context_env": "NORTHCUA_TEST_BUILD_CONTEXT",
            "layers": [
                {"kind": "run", "command": "echo first"},
                {"kind": "run", "command": "echo second"},
            ],
            "metadata": {
                "e2b_run_batch_sizes": [1, 1],
                "e2b_explicit_run_user": "root",
                "e2b_start_cmd": "start-gnome",
                "e2b_ready_cmd": "probe-gnome",
                "e2b_skip_cache": False,
            },
        }
    )

    alias = await E2BTemplateRuntimeMixin()._build_template_async(spec)

    assert alias == "managed-integration"
    assert FakeAsyncTemplate.selector_calls == [("python", "3.12")]
    assert FakeAsyncTemplate.builder._template._file_context_path == str(tmp_path.resolve())
    assert [user for _commands, user in FakeAsyncTemplate.builder.run_calls] == ["root", "root"]
    assert [len(commands) for commands, _user in FakeAsyncTemplate.builder.run_calls] == [1, 1]
    assert FakeAsyncTemplate.builder.set_users == []
    assert FakeAsyncTemplate.builder.startup == ("start-gnome", "probe-gnome")
    assert FakeAsyncTemplate.build_kwargs["skip_cache"] is False


def test_template_doctor_requires_accessibility_import() -> None:
    command = SandboxTemplateDoctor()._doctor_command()

    assert '"pyatspi"' in command
    assert 'name not in {"pyatspi"}' not in command


def test_template_doctor_does_not_echo_raw_binary_values() -> None:
    command = SandboxTemplateDoctor(checked_binaries=["bad'binary"])._doctor_command()

    assert "echo checked bad'binary" not in command
    assert "bad'binary" in command


def test_template_doctor_merges_core_and_catalog_binaries() -> None:
    doctor = SandboxTemplateDoctor(checked_binaries=["gimp", "python3.12"])

    assert doctor._checked_binaries[:6] == [  # noqa: SLF001 - contract regression for doctor merge behavior.
        "python3",
        "python3.12",
        "northcua-device",
        "xdotool",
        "wmctrl",
        "scrot",
    ]
    assert doctor._checked_binaries.count("python3.12") == 1  # noqa: SLF001
    assert "gimp" in doctor._checked_binaries  # noqa: SLF001


def test_template_copy_sources_resolve_relative_to_spec_or_repo(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("ok", encoding="utf-8")
    spec_path = tmp_path / "template.yaml"
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "fake",
            "alias": "copy-test",
            "layers": [{"kind": "copy", "files": {"payload.txt": "/tmp/payload.txt"}}],
        }
    )

    resolved = _resolve_copy_sources(spec, spec_path=spec_path)

    assert resolved.layers[0].files == {str(source.resolve()): "/tmp/payload.txt"}


def test_e2b_template_copy_sources_are_repo_relative() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "copy-test",
            "layers": [{"kind": "copy", "files": {"examples/templates/web.yaml": "/tmp/spec.yaml"}}],
        }
    )

    resolved = _resolve_copy_sources(spec, spec_path=Path("examples/templates/desktop.yaml").resolve())

    assert resolved.layers[0].files == {"examples/templates/web.yaml": "/tmp/spec.yaml"}


def test_e2b_image_copy_sources_are_repo_relative() -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "copy-test",
            "image": {"layers": [{"kind": "copy", "files": {"examples/templates/web.yaml": "/tmp/spec.yaml"}}]},
        }
    )

    resolved = _resolve_copy_sources(spec, spec_path=Path("examples/templates/desktop.yaml").resolve())

    assert resolved.image is not None
    assert resolved.image.layers[0].files == {"examples/templates/web.yaml": "/tmp/spec.yaml"}


def test_e2b_template_copy_sources_reject_outside_build_context(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("ok", encoding="utf-8")
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "copy-test",
            "layers": [{"kind": "copy", "files": {str(source): "/tmp/payload.txt"}}],
        }
    )

    with pytest.raises(ValueError, match="configured build context"):
        _resolve_copy_sources(spec, spec_path=tmp_path / "template.yaml")


def test_e2b_template_copy_sources_use_external_build_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_context = tmp_path / "context"
    build_context.mkdir()
    (build_context / "payload.bin").write_bytes(b"official")
    monkeypatch.setenv("OSWORLD_V2_BUILD_CONTEXT", str(build_context))
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "copy-test",
            "build_context_env": "OSWORLD_V2_BUILD_CONTEXT",
            "layers": [{"kind": "copy", "files": {"payload.bin": "/tmp/payload.bin"}}],
        }
    )

    resolved = _resolve_copy_sources(spec, spec_path=tmp_path / "template.yaml")

    assert resolved.layers[0].files == {"payload.bin": "/tmp/payload.bin"}


def test_e2b_template_copy_sources_require_external_build_context_env(tmp_path: Path) -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "copy-test",
            "build_context_env": "OSWORLD_V2_BUILD_CONTEXT",
        }
    )

    with pytest.raises(RuntimeError, match="OSWORLD_V2_BUILD_CONTEXT"):
        _resolve_copy_sources(spec, spec_path=tmp_path / "template.yaml")


def test_e2b_template_copy_sources_reject_escape_from_external_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_context = tmp_path / "context"
    build_context.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    monkeypatch.setenv("OSWORLD_V2_BUILD_CONTEXT", str(build_context))
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "copy-test",
            "build_context_env": "OSWORLD_V2_BUILD_CONTEXT",
            "layers": [{"kind": "copy", "files": {str(outside): "/tmp/outside.bin"}}],
        }
    )

    with pytest.raises(ValueError, match="configured build context"):
        _resolve_copy_sources(spec, spec_path=tmp_path / "template.yaml")


def test_e2b_base_image_env_requires_digest_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SandboxTemplateSpec.model_validate(
        {
            "provider": "e2b",
            "alias": "rootfs-test",
            "base_image_env": "OSWORLD_V2_BASE_IMAGE",
            "metadata": {"require_digest_pinned_base_image": True},
        }
    )
    monkeypatch.setenv("OSWORLD_V2_BASE_IMAGE", "registry.example/osworld:v2")

    with pytest.raises(RuntimeError, match="sha256 digest"):
        E2BTemplateRuntimeMixin._resolve_base_image(spec)

    digest = "registry.example/osworld@sha256:" + "a" * 64
    monkeypatch.setenv("OSWORLD_V2_BASE_IMAGE", digest)
    assert E2BTemplateRuntimeMixin._resolve_base_image(spec) == digest


def test_e2b_template_log_redaction_preserves_template_aliases() -> None:
    assert E2BTemplateRuntimeMixin._redact("base_template=e2b_desktop:base") == "base_template=e2b_desktop:base"
    synthetic_key = "e2b_" + "A" * 20
    assert E2BTemplateRuntimeMixin._redact(f"api_key={synthetic_key}") == "api_key=<redacted>"
    assert E2BTemplateRuntimeMixin._redact("Authorization: Bearer topsecret") == ("Authorization: Bearer <redacted>")
    assert E2BTemplateRuntimeMixin._redact('Authorization: Bearer "two word secret"') == (
        "Authorization: Bearer <redacted>"
    )
    assert "topsecret" not in E2BTemplateRuntimeMixin._redact('{"api_key": "topsecret"}')
    assert "two word secret" not in E2BTemplateRuntimeMixin._redact('{"api_key": "two word secret"}')
    assert "topsecret" not in E2BTemplateRuntimeMixin._redact("password = topsecret")
    assert (
        E2BTemplateRuntimeMixin._redact("https://token@example.com/source.tar.gz?signature=abc&x=1")
        == "https://<redacted>@example.com/source.tar.gz?signature=<redacted>&x=1"
    )


@pytest.mark.asyncio
async def test_template_doctor_requires_live_gate_for_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORTHCUA_RUN_E2B_BATCH", raising=False)

    with pytest.raises(RuntimeError, match="NORTHCUA_RUN_E2B_BATCH"):
        await SandboxOperations().doctor(provider_id="e2b", template="template", desktop=True)
