"""AgentKit skill resolver and asset-tool tests."""

from pathlib import Path

import pytest

pytest.importorskip("nexau.archs.main_sub.skill", reason="AgentKit skill tests require NexAU.")

from nexau.archs.main_sub.skill import Skill

from northcua.agentkit import AgentCliSpec, AgentRuntimeSpec, AgentSkillSpec, AgentSpec
from northcua.agentkit.skill_assets import AgentKitSkillAssetToolFactory
from northcua.agentkit.skills import AgentKitSkillResolver, ResolvedAgentSkillSettings
from northcua.core.config import AgentKitSkillConfig, NorthCUAAppConfig
from northcua_protocol.capabilities import PlatformKind


def test_agentkit_skill_resolver_merges_raw_spec_and_app_paths(tmp_path: Path) -> None:
    """Spec/config skills become absolute NexAU folder paths without duplicates."""

    agent_dir = tmp_path / "agent"
    app_dir = tmp_path / "app"
    raw_skill = agent_dir / "raw-skill"
    spec_skill = agent_dir / "spec-skill"
    app_skill = app_dir / "app-skill"
    for folder in (raw_skill, spec_skill, app_skill):
        folder.mkdir(parents=True)
    raw_config = {"skills": ["raw-skill"]}
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android"),
        skills=AgentSkillSpec(paths=("spec-skill",), asset_tools_enabled=True),
    )
    app = NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "agent.yaml",
            "agents": {
                "android": {
                    "skills": {
                        "paths": ["app-skill"],
                        "asset_read_max_bytes": 1234,
                        "allowed_file_extensions": [".md"],
                    }
                }
            },
        }
    )

    settings = AgentKitSkillResolver.apply_to_raw_agent_config(
        raw_config,
        spec=spec,
        app_agent_config=app.agents["android"],
        app_config_path=app_dir / "app.yaml",
        agent_config_path=agent_dir / "agent.yaml",
    )

    assert raw_config["skills"] == [str(raw_skill.resolve()), str(spec_skill.resolve()), str(app_skill.resolve())]
    assert settings.asset_tools_enabled is True
    assert settings.asset_read_max_bytes == 1234
    assert settings.allowed_file_extensions == frozenset({".md"})


def test_agentkit_skill_resolver_can_skip_missing_optional_paths(tmp_path: Path) -> None:
    """Explicit allow_missing is useful for portable example configs."""

    raw_config: dict[str, object] = {}
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android"),
        skills=AgentSkillSpec(paths=("missing-skill",), allow_missing=True),
    )

    settings = AgentKitSkillResolver.apply_to_raw_agent_config(
        raw_config,
        spec=spec,
        app_agent_config=None,
        app_config_path=tmp_path / "app.yaml",
        agent_config_path=tmp_path / "agent.yaml",
    )

    assert settings.skill_paths == ()
    assert raw_config == {}


def test_agentkit_skill_resolver_respects_disabled_config(tmp_path: Path) -> None:
    """A disabled app skill block suppresses spec defaults and asset tools."""

    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android"),
        skills=AgentSkillSpec(paths=("missing-skill",), asset_tools_enabled=True, allow_missing=True),
    )

    settings = AgentKitSkillResolver.resolve_settings(
        spec_skills=spec.skills,
        app_skills=AgentKitSkillConfig(enabled=False),
        app_config_path=tmp_path / "app.yaml",
        agent_config_path=tmp_path / "agent.yaml",
    )

    assert settings.skill_paths == ()
    assert settings.asset_tools_enabled is False


def test_skill_asset_tools_read_text_and_summarize_binary(tmp_path: Path) -> None:
    """Asset tools only expose files from registered skill folders."""

    skill_dir = tmp_path / "spreadsheet-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: spreadsheet-skill\ndescription: Spreadsheet examples\n---\nUse examples.",
        encoding="utf-8",
    )
    (skill_dir / "examples").mkdir()
    (skill_dir / "examples" / "invoice.csv").write_text("invoice,total\nA-1,42\n", encoding="utf-8")
    for index in range(3):
        (skill_dir / "examples" / f"extra-{index}.csv").write_text("id\n1\n", encoding="utf-8")
    (skill_dir / "image.bin").write_bytes(b"\x00\x01")
    skill = Skill.from_folder(skill_dir)
    factory = AgentKitSkillAssetToolFactory(
        skills=[skill],
        settings=ResolvedAgentSkillSettings(
            skill_paths=(skill_dir,),
            asset_tools_enabled=True,
            asset_read_max_bytes=8,
            allowed_file_extensions=frozenset({".csv", ".md"}),
            allow_binary_summary=True,
            allow_missing=False,
        ),
    )
    tools = {tool.name: tool for tool in factory.build_tools()}

    listed = tools["list_skill_assets"].implementation(skill_name="spreadsheet-skill", path="examples", max_entries=2)
    read = tools["read_skill_asset"].implementation(
        skill_name="spreadsheet-skill",
        relative_path="examples/invoice.csv",
    )
    read_with_large_request = tools["read_skill_asset"].implementation(
        skill_name="spreadsheet-skill",
        relative_path="examples/invoice.csv",
        max_bytes=100,
    )
    binary = tools["read_skill_asset"].implementation(skill_name="spreadsheet-skill", relative_path="image.bin")
    traversal = tools["read_skill_asset"].implementation(skill_name="spreadsheet-skill", relative_path="../x")

    assert listed["status"] == "success"
    assert len(listed["result"]["entries"]) == 2
    assert read["result"]["content"] == "invoice,"
    assert read["result"]["truncated"] is True
    assert read_with_large_request["result"]["content"] == "invoice,"
    assert read_with_large_request["result"]["truncated"] is True
    assert binary["result"]["binary_summary"] is True
    assert traversal["status"] == "failure"
