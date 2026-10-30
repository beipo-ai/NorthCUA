from __future__ import annotations

from pathlib import Path

from northcua.core.config import NorthCUAAppConfig
from northcua.core.effective_config import ensure_evaluation_config


def test_effective_config_adds_evaluation_defaults_when_missing(tmp_path: Path) -> None:
    source = tmp_path / "app.yaml"
    source.write_text(
        """
agent_config_path: agent.yaml
artifacts:
  root_dir: logs
""".strip(),
        encoding="utf-8",
    )
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        """
agent_config_path: default-agent.yaml
evaluation:
  enabled: true
  schedule: after_round_async
  outputs:
    round_log_name: round_evaluations.jsonl
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "effective.yaml"

    config = ensure_evaluation_config(source_path=source, default_path=defaults, output_path=output)
    reloaded = NorthCUAAppConfig.from_yaml(output)

    assert config.evaluation.enabled is True
    assert reloaded.evaluation.enabled is True
    assert reloaded.artifacts.round_evaluation_log_name == "round_evaluations.jsonl"


def test_effective_config_respects_existing_evaluation_block(tmp_path: Path) -> None:
    source = tmp_path / "app.yaml"
    source.write_text(
        """
agent_config_path: agent.yaml
evaluation:
  enabled: false
  schedule: disabled
""".strip(),
        encoding="utf-8",
    )
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        """
agent_config_path: default-agent.yaml
evaluation:
  enabled: true
  schedule: after_round_async
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "effective.yaml"

    config = ensure_evaluation_config(source_path=source, default_path=defaults, output_path=output)

    assert config.evaluation.enabled is False
    assert config.evaluation.schedule.value == "disabled"
