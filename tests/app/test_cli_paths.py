from __future__ import annotations

from tests.app.test_cli import default_app_config_path, default_session_id, resolve_app_config_path, resolve_log_root


def test_default_session_id_has_prefix() -> None:
    session_id = default_session_id("task")

    assert session_id.startswith("task-")


def test_default_app_config_path_prefers_local() -> None:
    path = default_app_config_path()

    assert path.name in {"app.local.yaml", "app.example.yaml"}


def test_resolve_app_config_path_uses_default_when_missing() -> None:
    resolved = resolve_app_config_path(None)

    assert resolved == default_app_config_path()


def test_resolve_log_root_places_plain_name_under_repo_logs() -> None:
    resolved = resolve_log_root("calculator")

    assert resolved is not None
    assert resolved.name == "calculator"
    assert resolved.parent.name == "logs"


def test_resolve_log_root_preserves_path_like_values(tmp_path) -> None:
    resolved = resolve_log_root(str(tmp_path / "northcua-runs"))

    assert resolved == (tmp_path / "northcua-runs").resolve()
