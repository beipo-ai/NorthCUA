"""Copied Desktop settings recipes start, and `config validate` checks what startup checks."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from northcua.agents.desktop.observation import ObservationCoordinator
from northcua.agents.desktop.observation.adapters import BUILT_IN_PROBE_ADAPTER_NAMES, ProbeAdapterRegistry
from northcua.cli.config_commands import run_config_command
from northcua.cli.parser import build_parser
from northcua.core.config import AccessibilityObservationConfig, ArtifactConfig
from northcua.core.config_loader import NorthCUAConfigLoader
from northcua.core.errors import ConfigValidationError
from northcua.runtime.artifacts.store import NorthCUAArtifactStore

ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = ROOT / "examples" / "desktop-settings"
RECIPES = sorted(path.name for path in RECIPE_ROOT.iterdir() if (path / "app.yaml").is_file())


@pytest.fixture
def model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NORTHCUA_MODEL", "OPENAI_API_KEY", "NORTHCUA_SUMMARY_MODEL", "NORTHCUA_SUMMARY_API_KEY"):
        monkeypatch.setenv(name, "fixture")
    for name in ("NORTHCUA_BASE_URL", "NORTHCUA_SUMMARY_BASE_URL"):
        monkeypatch.setenv(name, "https://fixture.invalid/v1")


def _copy_recipe(recipe: str, tmp_path: Path) -> Path:
    target = tmp_path / recipe
    shutil.copytree(RECIPE_ROOT / recipe, target)
    return target / "app.yaml"


def _config_command(*argv: str) -> int:
    return run_config_command(build_parser().parse_args(["config", *argv]))


def test_desktop_settings_recipes_are_discovered() -> None:
    assert {"long-context", "response-recovery"} <= set(RECIPES)


@pytest.mark.parametrize("recipe", RECIPES)
def test_copied_recipe_validates_and_builds_local_observation(
    recipe: str, tmp_path: Path, model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_config = _copy_recipe(recipe, tmp_path)
    monkeypatch.chdir(tmp_path)

    assert _config_command("validate", "--app-config", str(app_config)) == 0
    bootstrap = NorthCUAConfigLoader.from_app_config(app_config, agent_kind="desktop")
    coordinator = ObservationCoordinator.for_local(
        app_config=bootstrap.app,
        agent_config=bootstrap.agent,
        artifact_store=NorthCUAArtifactStore.for_session(tmp_path / "artifacts", "recipe", ArtifactConfig()),
    )

    assert set(coordinator.probe_chains) == {"desktop_snapshot_default"}
    assert coordinator.accessibility_augmentor is None


@pytest.mark.parametrize("recipe", RECIPES)
def test_config_validate_reports_enabled_a11y_without_its_server(
    recipe: str, tmp_path: Path, model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_config = _copy_recipe(recipe, tmp_path)
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load(app_config.read_text(encoding="utf-8"))
    data["observation"] = {"a11y": {"enabled": True}}
    app_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="unknown MCP server `DesktopNativeA11yExecutor`"):
        _config_command("validate", "--app-config", str(app_config))


def test_config_validate_accepts_initialized_templates(
    tmp_path: Path, model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _config_command("init", "--directory", str(tmp_path / "config")) == 0

    assert _config_command("validate", "--app-config", str(tmp_path / "config" / "app.example.yaml")) == 0


def test_accessibility_observation_is_off_by_default() -> None:
    assert AccessibilityObservationConfig().enabled is False


def test_built_in_probe_adapter_names_match_registry(tmp_path: Path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "direct", ArtifactConfig())

    assert ProbeAdapterRegistry(store).names == BUILT_IN_PROBE_ADAPTER_NAMES
