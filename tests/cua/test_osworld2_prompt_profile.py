import hashlib
from pathlib import Path


from northcua.agents.desktop.prompt.builders import PromptEnvelopeBuilder
from northcua.core.config import NorthCUAAppConfig
from northcua.core.coordinates import CoordinatePolicy
from northcua.core.enums import CoordinateSpace
from northcua.agents.desktop.prompt.profiles.osworld2.prompt import SYSTEM_TEMPLATE_SHA256, system_template_path


def profile_config():
    return NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "fixture.yaml",
            "prompt": {"system_prompt_profile": "osworld2"},
            "execution": {"no_tool_call_attempts": 3},
            "evaluation": {"enabled": False},
            "desktop": {"coordinate_policy": {"input_space": "normalized_1000"}},
        }
    )


def test_single_frozen_prompt_and_runtime_query():
    assert hashlib.sha256(system_template_path().read_bytes()).hexdigest() == SYSTEM_TEMPLATE_SHA256
    config = profile_config()
    root = Path(__file__).resolve().parents[2] / "northcua/agents/desktop/prompt/templates"
    builder = PromptEnvelopeBuilder(
        root, config.prompt, coordinate_policy=CoordinatePolicy(CoordinateSpace.NORMALIZED_1000)
    )
    system = builder.system_builder.render(
        1920,
        1080,
        desktop_environment="osworld",
        home_directory="/home/user",
        coordinate_policy=CoordinatePolicy(CoordinateSpace.NORMALIZED_1000),
    )
    assert "normalized 0-1000" in system and "ask_user" not in system
    assert "{{" not in system
    assert builder.observation_builder.render_persistent("  original query\n") == "  original query\n"
    assert "[INFEASIBLE]" not in builder.observation_builder.render_persistent("task")
    assert len(list(system_template_path().parent.glob("*.j2"))) == 1


def test_portable_example_and_auxiliary_binding(monkeypatch, tmp_path):
    import shutil
    from northcua.core.config_loader import NorthCUAConfigLoader

    root = Path(__file__).resolve().parents[2]
    config_dir = tmp_path / "local-config"
    shutil.copytree(root / "examples/desktop-settings/osworld2", config_dir)
    monkeypatch.chdir(tmp_path)
    for name in ("NORTHCUA_MODEL", "OPENAI_API_KEY", "NORTHCUA_SUMMARY_MODEL", "NORTHCUA_SUMMARY_API_KEY"):
        monkeypatch.setenv(name, "fixture")
    for name in ("NORTHCUA_BASE_URL", "NORTHCUA_SUMMARY_BASE_URL"):
        monkeypatch.setenv(name, "https://fixture.invalid/v1")
    bootstrap = NorthCUAConfigLoader.from_app_config(config_dir / "app.yaml")
    assert bootstrap.app.context_management.summary.model.agent_config_path == str(config_dir / "summary-agent.yaml")
    assert bootstrap.agent.llm_config.get_param("extra_body") == {
        "presence_penalty": 1.0,
        "top_k": 20,
        "min_p": 0,
        "repetition_penalty": 1.0,
    }
    assert bootstrap.agent.llm_config.temperature == 0.8
    assert bootstrap.agent.llm_config.timeout == 600
    assert bootstrap.agent.llm_config.max_tokens == 65536
    assert bootstrap.app.execution.no_tool_call_attempts == 3
