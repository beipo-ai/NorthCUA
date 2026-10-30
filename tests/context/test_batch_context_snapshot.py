from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.prompt_builder import PromptBuilder

from northcua.batch.context_snapshot import freeze_batch_context_configs, snapshot_controller_app_config
from northcua.batch.contracts import BatchManifest, BatchTaskSpec


def test_batch_context_config_snapshot_is_immutable_and_resume_safe(tmp_path) -> None:
    agent = tmp_path / "agent.yaml"
    prompt = tmp_path / "prompt.j2"
    prompt.write_text("Frozen prompt", encoding="utf-8")
    agent.write_text(
        "name: primary\nsystem_prompt_type: jinja\nsystem_prompt: prompt.j2\nllm_config:\n  model: fake\n"
        "  base_url: https://example.invalid/v1\n  api_key: test-key\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.yaml"
    summary.write_text("name: summary\nllm_config:\n  model: fake\n  max_context_tokens: 8192\n", encoding="utf-8")
    app = tmp_path / "app.yaml"
    app.write_text(
        """agent_config_path: agent.yaml
context_management:
  mode: enforce
  summary:
    model:
      source: agent_config
      agent_config_path: summary.yaml
desktop:
  backend: native
  primary_screen_only: false
""",
        encoding="utf-8",
    )
    manifest = BatchManifest(
        batch_id="batch-snapshot",
        tasks=[
            BatchTaskSpec(
                task_id="case",
                agent_kind="desktop",
                request="do work",
                metadata={"controller_app_config_path": str(app)},
            )
        ],
    )
    batch_dir = tmp_path / "output" / manifest.batch_id

    first = freeze_batch_context_configs(manifest, batch_output_dir=batch_dir)
    frozen_path = first.tasks[0].metadata["controller_app_config_path"]
    frozen_text = Path(str(frozen_path)).read_text(encoding="utf-8")
    app.write_text("context_management:\n  mode: off\n", encoding="utf-8")
    resumed = freeze_batch_context_configs(first, batch_output_dir=batch_dir)

    assert resumed.tasks[0].metadata["controller_app_config_path"] == frozen_path
    assert Path(str(frozen_path)).read_text(encoding="utf-8") == frozen_text
    index = json.loads((batch_dir / "config_snapshot" / "index.json").read_text(encoding="utf-8"))
    assert index["version"] == 1
    assert len(index["configs"]) == 1
    frozen_app = yaml.safe_load(frozen_text)
    assert frozen_app["desktop"]["backend"] == "native"
    assert frozen_app["desktop"]["primary_screen_only"] is False
    frozen_agent_path = Path(frozen_app["agent_config_path"])
    frozen_agent = yaml.safe_load(frozen_agent_path.read_text(encoding="utf-8"))
    assert Path(frozen_agent["system_prompt"]).is_file()
    assert AgentConfig.from_yaml(frozen_agent_path).name == "primary"


def test_controller_snapshot_requires_explicit_rotation_after_source_mutation(tmp_path, monkeypatch) -> None:
    source = tmp_path / "app.yaml"
    source.write_text("agent_config_path: first.yaml\n", encoding="utf-8")
    root = tmp_path / "controller-artifacts" / ".controller_config_snapshot"

    first = snapshot_controller_app_config(source, snapshot_root=root)
    first_text = first.read_text(encoding="utf-8")
    source.write_text("agent_config_path: changed.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit rollout/rollback"):
        snapshot_controller_app_config(source, snapshot_root=root)
    monkeypatch.setenv("NORTHCUA_REFRESH_CONTROLLER_CONFIG_SNAPSHOT", "1")
    after_rotation = snapshot_controller_app_config(source, snapshot_root=root)

    assert after_rotation != first
    assert after_rotation.read_text(encoding="utf-8") != first_text


def test_controller_snapshot_detects_dependency_only_change(tmp_path, monkeypatch) -> None:
    prompt = tmp_path / "prompt.j2"
    prompt.write_text("first prompt", encoding="utf-8")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: agent\nsystem_prompt_type: jinja\nsystem_prompt: prompt.j2\n",
        encoding="utf-8",
    )
    source = tmp_path / "app.yaml"
    source.write_text("agent_config_path: agent.yaml\n", encoding="utf-8")
    root = tmp_path / "controller" / ".controller_config_snapshot"

    first = snapshot_controller_app_config(source, snapshot_root=root)
    prompt.write_text("changed prompt", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit rollout/rollback"):
        snapshot_controller_app_config(source, snapshot_root=root)
    monkeypatch.setenv("NORTHCUA_REFRESH_CONTROLLER_CONFIG_SNAPSHOT", "1")

    rotated = snapshot_controller_app_config(source, snapshot_root=root)

    assert rotated != first


def test_controller_snapshot_rejects_tampered_frozen_dependency(tmp_path) -> None:
    prompt = tmp_path / "prompt.j2"
    prompt.write_text("first prompt", encoding="utf-8")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: agent\nsystem_prompt_type: jinja\nsystem_prompt: prompt.j2\n",
        encoding="utf-8",
    )
    source = tmp_path / "app.yaml"
    source.write_text("agent_config_path: agent.yaml\n", encoding="utf-8")
    root = tmp_path / "controller" / ".controller_config_snapshot"
    frozen = snapshot_controller_app_config(source, snapshot_root=root)
    frozen_agent = Path(yaml.safe_load(frozen.read_text(encoding="utf-8"))["agent_config_path"])
    frozen_agent.write_text("name: tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity validation"):
        snapshot_controller_app_config(source, snapshot_root=root)


@pytest.mark.parametrize(
    ("prompt_type", "prompt_value"),
    [
        ("string", "prompt.j2"),
        ("jinja", "prompt.j2"),
        ("file", ["prompt.j2", "prompt-2.j2"]),
        ("jinja", [{"content": "prompt.j2", "cache": False}, {"content": "prompt-2.j2"}]),
    ],
)
def test_snapshot_preserves_all_nexau_system_prompt_representations(tmp_path, prompt_type, prompt_value) -> None:
    (tmp_path / "prompt.j2").write_text("First prompt", encoding="utf-8")
    (tmp_path / "prompt-2.j2").write_text("Second prompt", encoding="utf-8")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        yaml.safe_dump(
            {
                "name": "agent",
                "system_prompt_type": prompt_type,
                "system_prompt": prompt_value,
                "llm_config": {
                    "model": "fake",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "test-key",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        "agent_config_path: agent.yaml\ncontext_management:\n  mode: shadow\n",
        encoding="utf-8",
    )
    manifest = BatchManifest(
        batch_id="prompt-codec",
        tasks=[
            BatchTaskSpec(
                task_id="case",
                agent_kind="generic",
                request="work",
                metadata={"controller_app_config_path": str(app)},
            )
        ],
    )

    frozen = freeze_batch_context_configs(manifest, batch_output_dir=tmp_path / "output")
    frozen_app = yaml.safe_load(
        Path(str(frozen.tasks[0].metadata["controller_app_config_path"])).read_text(encoding="utf-8")
    )
    frozen_agent_path = Path(frozen_app["agent_config_path"])
    original = AgentConfig.from_yaml(agent)
    copied = AgentConfig.from_yaml(frozen_agent_path)
    builder = PromptBuilder()
    original_parts = builder.build_system_prompt(original, tools=[], include_tool_instructions=False)
    copied_parts = builder.build_system_prompt(copied, tools=[], include_tool_instructions=False)

    assert [(part.text, part.cache) for part in copied_parts] == [(part.text, part.cache) for part in original_parts]
    if prompt_type == "string":
        frozen_raw = yaml.safe_load(frozen_agent_path.read_text(encoding="utf-8"))
        assert frozen_raw["system_prompt"] == "prompt.j2"


def test_batch_snapshot_resume_rejects_tampered_app_or_dependency(tmp_path) -> None:
    prompt = tmp_path / "prompt.j2"
    prompt.write_text("original", encoding="utf-8")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: agent\nsystem_prompt_type: jinja\nsystem_prompt: prompt.j2\n",
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        "agent_config_path: agent.yaml\ncontext_management:\n  mode: enforce\n",
        encoding="utf-8",
    )
    manifest = BatchManifest(
        batch_id="integrity",
        tasks=[
            BatchTaskSpec(
                task_id="case",
                agent_kind="desktop",
                request="work",
                metadata={"controller_app_config_path": str(app)},
            )
        ],
    )
    output = tmp_path / "output"
    frozen = freeze_batch_context_configs(manifest, batch_output_dir=output)
    frozen_app_path = Path(str(frozen.tasks[0].metadata["controller_app_config_path"]))
    frozen_app_path.write_text("context_management:\n  mode: off\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity validation"):
        freeze_batch_context_configs(frozen, batch_output_dir=output)


def test_batch_snapshot_is_exact_noop_when_context_management_is_off(tmp_path) -> None:
    app = tmp_path / "app.yaml"
    app.write_text("agent_config_path: mutable.yaml\ncontext_management:\n  mode: off\n", encoding="utf-8")
    manifest = BatchManifest(
        batch_id="off",
        tasks=[
            BatchTaskSpec(
                task_id="case",
                agent_kind="desktop",
                request="work",
                metadata={"controller_app_config_path": str(app)},
            )
        ],
    )

    frozen = freeze_batch_context_configs(manifest, batch_output_dir=tmp_path / "output")

    assert frozen == manifest
    assert not (tmp_path / "output" / "config_snapshot").exists()


def test_batch_snapshot_gates_each_agent_override_independently(tmp_path) -> None:
    app = tmp_path / "app.yaml"
    app.write_text(
        """agent_config_path: agent.yaml
context_management:
  mode: off
  agent_overrides:
    web:
      mode: enforce
""",
        encoding="utf-8",
    )
    manifest = BatchManifest(
        batch_id="mixed",
        tasks=[
            BatchTaskSpec(
                task_id="desktop",
                agent_kind="desktop",
                request="work",
                metadata={"controller_app_config_path": str(app)},
            ),
            BatchTaskSpec(
                task_id="web",
                agent_kind="web",
                request="work",
                metadata={"controller_app_config_path": str(app)},
            ),
        ],
    )

    frozen = freeze_batch_context_configs(manifest, batch_output_dir=tmp_path / "output")

    assert frozen.tasks[0].metadata == manifest.tasks[0].metadata
    assert frozen.tasks[1].metadata["controller_app_config_path"] != str(app)
