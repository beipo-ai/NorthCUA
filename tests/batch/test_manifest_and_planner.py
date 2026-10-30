"""Batch manifest and planner tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchTaskSpec
from northcua.batch.manifest import load_batch_manifest
from northcua.batch.manifest import loader as manifest_loader
from northcua.batch.planner import BatchTaskPlanner
from northcua.batch.request_adapters import BatchRequestAdapterRegistry
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.capabilities import Capability


def test_yaml_manifest_loads_defaults_and_tasks(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
batch_id: demo
defaults:
  controller_url: ws://controller
  sandbox:
    provider: fake
    template: ubuntu
tasks:
  - task_id: t1
    agent_kind: web
    query: Open the page
    url: https://example.com
""",
        encoding="utf-8",
    )

    manifest = load_batch_manifest(manifest_path)

    assert manifest.batch_id == "demo"
    assert manifest.defaults.sandbox.provider == "fake"
    assert manifest.tasks[0].request == "Open the page"


def test_jsonl_manifest_loads_tasks(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tasks.jsonl"
    manifest_path.write_text('{"task_id":"t1","agent_kind":"desktop","query":"Calculate 1+1"}\n', encoding="utf-8")

    manifest = load_batch_manifest(manifest_path)

    assert manifest.tasks[0].agent_kind == "desktop"
    assert manifest.tasks[0].request == "Calculate 1+1"


def test_json_manifest_uses_json_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "tasks.json"
    manifest_path.write_text(
        '{"batch_id":"json-demo","tasks":[{"task_id":"t1","agent_kind":"desktop","query":"Open"}]}',
        encoding="utf-8",
    )

    def reject_yaml(_: str) -> object:
        raise AssertionError("JSON manifests must not use the YAML parser")

    monkeypatch.setattr(manifest_loader.yaml, "safe_load", reject_yaml)

    manifest = load_batch_manifest(manifest_path)

    assert manifest.batch_id == "json-demo"
    assert manifest.tasks[0].request == "Open"


def test_manifest_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
batch_id: duplicate-demo
tasks:
  - task_id: same
    agent_kind: web
    query: First
  - task_id: same
    agent_kind: web
    query: Second
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task_id values must be unique"):
        load_batch_manifest(manifest_path)


def test_planner_maps_builtin_web_task_to_controller_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
batch_id: demo
tasks:
  - task_id: t1
    agent_kind: web
    query: Search docs
    url: https://example.com
""",
        encoding="utf-8",
    )
    manifest = load_batch_manifest(manifest_path)

    planned = BatchTaskPlanner().plan(manifest, manifest.tasks[0], device_id="device-1", sandbox_id="sandbox-1")

    assert planned.agent_kind == "web"
    assert planned.target_device_id == "device-1"
    assert Capability.BROWSER_PAGE in planned.required_capabilities
    assert planned.metadata["batch_id"] == "demo"
    assert planned.metadata["url"] == "https://example.com"


def test_planner_does_not_duplicate_custom_agent_artifacts() -> None:
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="input",
        kind=ArtifactKind.TEXT,
        data=b"hello",
        mime_type="text/plain",
    )
    batch = BatchManifest(
        batch_id="custom",
        tasks=[
            BatchTaskSpec(
                task_id="custom-1",
                agent_kind="android",
                query="Inspect app",
                input_artifacts=[artifact],
            )
        ],
    )

    planned = BatchTaskPlanner().plan(batch, batch.tasks[0], device_id="device-1", sandbox_id="sandbox-1")

    assert planned.input_artifacts == [artifact]


def test_planner_uses_injected_agent_request_adapter() -> None:
    """Custom agents can project request aliases without editing planner code."""

    class AndroidAdapter:
        agent_kinds = frozenset({"android"})

        def can_project(self, task: BatchTaskSpec) -> bool:
            return task.agent_kind == "android"

        def project(self, task: BatchTaskSpec):
            return {
                "agent_kind": "android",
                "request": f"android::{task.request}",
                "target_device_id": "",
                "required_capabilities": set(task.required_capabilities),
                "input_artifacts": [],
                "metadata": {"app_package": task.metadata.get("app_package", "")},
            }

    registry = BatchRequestAdapterRegistry([AndroidAdapter()])
    manifest = BatchManifest(
        batch_id="custom-adapter",
        tasks=[
            BatchTaskSpec(
                task_id="android-1",
                agent_kind="android",
                query="Inspect app",
                metadata={"app_package": "com.example"},
            )
        ],
    )

    planned = BatchTaskPlanner(request_adapters=registry).plan(
        manifest,
        manifest.tasks[0],
        device_id="device-1",
        sandbox_id="sandbox-1",
    )

    assert planned.request == "android::Inspect app"
    assert planned.metadata["app_package"] == "com.example"


def test_planner_generates_deterministic_controller_ids() -> None:
    """Controller ids stay stable for idempotent replay of one attempt."""

    manifest = BatchManifest(
        batch_id="deterministic",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="Open")],
    )
    planner = BatchTaskPlanner()

    first = planner.plan(manifest, manifest.tasks[0], device_id="device", sandbox_id="sandbox", attempt=2)
    second = planner.plan(manifest, manifest.tasks[0], device_id="device", sandbox_id="sandbox", attempt=2)
    third = planner.plan(manifest, manifest.tasks[0], device_id="device", sandbox_id="sandbox", attempt=3)

    assert first.controller_task_id == second.controller_task_id
    assert first.trace_id == second.trace_id
    assert first.controller_task_id != third.controller_task_id
    assert first.metadata["batch_attempt"] == 2
