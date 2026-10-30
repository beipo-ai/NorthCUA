"""Controller configuration loading tests."""

import argparse

from northcua.cli.controller_commands import register_controller_commands
from northcua.cli.parallelism_args import apply_controller_parallelism_defaults
from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig


def test_controller_timeout_env_overrides_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "controller.yaml"
    config_path.write_text(
        "\n".join(
            [
                "command_timeout_seconds: 60",
                "offline_after_seconds: 30",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_OFFLINE_AFTER_SECONDS", "90")

    config = ControllerConfig.from_sources(config_path=str(config_path))

    assert config.command_timeout_seconds == 300
    assert config.offline_after_seconds == 90


def test_controller_explicit_timeout_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_SECONDS", "300")

    config = ControllerConfig.from_sources(overrides={"command_timeout_seconds": 45})

    assert config.command_timeout_seconds == 45


def test_controller_websocket_open_timeout_zero_disables(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_OPEN_TIMEOUT_SECONDS", "0")

    config = ControllerConfig.from_sources()

    assert config.websocket_open_timeout_seconds is None


def test_controller_websocket_ping_defaults_are_disabled_for_tunnel_stability() -> None:
    config = ControllerConfig.from_sources()

    assert config.websocket_ping_interval_seconds is None
    assert config.websocket_ping_timeout_seconds is None


def test_controller_websocket_ping_settings_can_be_enabled(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_PING_INTERVAL_SECONDS", "20")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_PING_TIMEOUT_SECONDS", "60")

    config = ControllerConfig.from_sources()

    assert config.websocket_ping_interval_seconds == 20.0
    assert config.websocket_ping_timeout_seconds == 60.0


def test_controller_websocket_ping_settings_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_PING_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_PING_TIMEOUT_SECONDS", "0")

    config = ControllerConfig.from_sources()

    assert config.websocket_ping_interval_seconds is None
    assert config.websocket_ping_timeout_seconds is None


def test_controller_websocket_backlog_non_positive_uses_default(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_BACKLOG", "0")

    config = ControllerConfig.from_sources()

    assert config.websocket_backlog == 4096


def test_controller_websocket_config_overrides_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "controller.yaml"
    config_path.write_text(
        "\n".join(
            [
                "websocket_open_timeout_seconds: 5",
                "websocket_backlog: 128",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_OPEN_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WEBSOCKET_BACKLOG", "1024")

    config = ControllerConfig.from_sources(config_path=str(config_path))

    assert config.websocket_open_timeout_seconds == 15
    assert config.websocket_backlog == 1024


def test_controller_runtime_telemetry_interval_env_override(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_RUNTIME_TELEMETRY_INTERVAL_SECONDS", "2.5")

    config = ControllerConfig.from_sources()

    assert config.runtime_telemetry_interval_seconds == 2.5


def test_controller_runtime_telemetry_interval_zero_disables(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_RUNTIME_TELEMETRY_INTERVAL_SECONDS", "0")

    config = ControllerConfig.from_sources()

    assert config.runtime_telemetry_interval_seconds == 0
    assert config.device_telemetry_persist_mode == "off"


def test_controller_device_telemetry_persistence_defaults_to_aggregate(monkeypatch):
    monkeypatch.delenv("NORTHCUA_CONTROLLER_DEVICE_TELEMETRY_PERSIST_MODE", raising=False)

    config = ControllerConfig.from_sources()

    assert config.device_telemetry_persist_mode == "aggregate"


def test_controller_high_pressure_defaults_match_batch_resume_needs(monkeypatch):
    monkeypatch.delenv("NORTHCUA_CONTROLLER_OFFLINE_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("NORTHCUA_CONTROLLER_WEBSOCKET_OPEN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("NORTHCUA_CONTROLLER_WEBSOCKET_BACKLOG", raising=False)

    config = ControllerConfig.from_sources()

    assert config.offline_after_seconds == 240
    assert config.websocket_open_timeout_seconds == 120
    assert config.websocket_backlog == 4096


def test_controller_device_telemetry_persistence_env_overrides(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_DEVICE_TELEMETRY_PERSIST_MODE", "sample")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_DEVICE_TELEMETRY_SAMPLE_RATE", "0.25")

    config = ControllerConfig.from_sources()

    assert config.device_telemetry_persist_mode == "sample"
    assert config.device_telemetry_sample_rate == 0.25


def test_controller_device_telemetry_sample_rate_is_clamped(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_DEVICE_TELEMETRY_SAMPLE_RATE", "2")

    config = ControllerConfig.from_sources()

    assert config.device_telemetry_sample_rate == 1.0


def test_controller_device_telemetry_aggregate_requires_runtime_telemetry(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_DEVICE_TELEMETRY_PERSIST_MODE", "aggregate")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_RUNTIME_TELEMETRY_INTERVAL_SECONDS", "0")

    try:
        ControllerConfig.from_sources()
    except ValueError as exc:
        assert "requires runtime_telemetry_interval_seconds" in str(exc)
    else:
        raise AssertionError("aggregate telemetry must have a persistence sink")


def test_controller_heartbeat_persist_interval_env_override(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_HEARTBEAT_PERSIST_INTERVAL_SECONDS", "30")

    config = ControllerConfig.from_sources()

    assert config.heartbeat_persist_interval_seconds == 30


def test_controller_event_pressure_defaults(monkeypatch):
    monkeypatch.delenv("NORTHCUA_CONTROLLER_COMMAND_STAGE_EVENT_MODE", raising=False)
    monkeypatch.delenv("NORTHCUA_CONTROLLER_COMMAND_STAGE_EVENT_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("NORTHCUA_CONTROLLER_EVENT_WRITE_BATCH_SIZE", raising=False)
    monkeypatch.delenv("NORTHCUA_CONTROLLER_EVENT_WRITE_FLUSH_INTERVAL_SECONDS", raising=False)

    config = ControllerConfig.from_sources()

    assert config.command_stage_event_mode == "aggregate"
    assert config.command_stage_event_sample_rate == 0.02
    assert config.event_write_batch_size == 128
    assert config.event_write_flush_interval_seconds == 0.25
    assert config.heartbeat_persist_interval_seconds == 30


def test_controller_event_pressure_env_overrides(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_STAGE_EVENT_MODE", "sample")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_STAGE_EVENT_SAMPLE_RATE", "0.1")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_EVENT_WRITE_BATCH_SIZE", "512")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_EVENT_WRITE_FLUSH_INTERVAL_SECONDS", "1.5")

    config = ControllerConfig.from_sources()

    assert config.command_stage_event_mode == "sample"
    assert config.command_stage_event_sample_rate == 0.1
    assert config.event_write_batch_size == 512
    assert config.event_write_flush_interval_seconds == 1.5


def test_controller_process_executor_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_EXECUTOR_MODE", "process")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_RUNNER_WORKERS", "8")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WORKER_QUEUE_MAX", "64")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WORKER_START_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WORKER_HEARTBEAT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_PROCESS_WORKER_MODULE", "northcua.test.fake_worker")

    config = ControllerConfig.from_sources(overrides={"runner_workers": 4})

    assert config.executor_mode == "process"
    assert config.runner_workers == 4
    assert config.worker_queue_max == 64
    assert config.worker_start_timeout_seconds == 30
    assert config.worker_heartbeat_timeout_seconds == 45
    assert config.process_worker_module == "northcua.test.fake_worker"


def test_controller_server_passes_timeout_floor_to_command_bus(tmp_path) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            blob_dir=str(tmp_path / "blobs"),
            command_timeout_floor_seconds=123,
            security={"insecure_dev_no_auth": True},
        )
    )
    try:
        assert server.command_bus.command_timeout_floor_seconds == 123
    finally:
        server.store.close()


def test_controller_legacy_action_timeout_env_populates_timeout_floor(monkeypatch) -> None:
    monkeypatch.delenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", raising=False)
    monkeypatch.setenv("NORTHCUA_ACTION_COMMAND_TIMEOUT_SECONDS", "180")

    config = ControllerConfig.from_sources()

    assert config.command_timeout_floor_seconds == 180


def test_controller_worker_queue_max_cli_zero_overrides_env(monkeypatch):
    monkeypatch.setenv("NORTHCUA_CONTROLLER_WORKER_QUEUE_MAX", "64")

    config = ControllerConfig.from_sources(overrides={"worker_queue_max": 0})

    assert config.worker_queue_max == 0


def test_controller_worker_queue_max_cli_preserves_unset_and_explicit_zero() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_controller_commands(subparsers)

    default_args = parser.parse_args(["controller", "serve"])
    zero_args = parser.parse_args(["controller", "serve", "--worker-queue-max", "0"])

    assert default_args.worker_queue_max is None
    assert zero_args.worker_queue_max == 0


def test_controller_parallelism_fills_unset_process_executor_caps() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_controller_commands(subparsers)

    args = parser.parse_args(["controller", "serve", "--parallelism", "120"])
    apply_controller_parallelism_defaults(args)

    assert args.runner_workers == 120
    assert args.worker_queue_max == 120


def test_controller_parallelism_preserves_explicit_queue_zero() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_controller_commands(subparsers)

    args = parser.parse_args(["controller", "serve", "--parallelism", "120", "--worker-queue-max", "0"])
    apply_controller_parallelism_defaults(args)

    assert args.runner_workers == 120
    assert args.worker_queue_max == 0
