from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northcua.integrations.osworld import runner


class Environment:
    vm_ip = "127.0.0.1"
    server_port = 5000

    def __init__(self):
        self.actions = []
        self.closed = False
        self.evaluated = False

    def reset(self, *, task_config):
        self.task = task_config

    def step(self, action, pause=0):
        self.actions.append(action)

    def evaluate(self):
        self.evaluated = True
        return float(self.actions == ["FAIL"])

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "status,reason,limit,actions",
    [
        ("fail", "max_steps", True, []),
        ("fail", "model_reported_failure", False, []),
        ("fail", "infeasible", False, ["FAIL"]),
        ("finish", "completed", False, ["DONE"]),
        ("error", "runtime_error", False, []),
    ],
)
def test_oracle_only_receives_explicit_terminal_claim(monkeypatch, tmp_path, status, reason, limit, actions):
    env = Environment()
    monkeypatch.setattr(
        runner,
        "_run_actor",
        AsyncMock(
            return_value=SimpleNamespace(
                terminal_status=status, hit_step_limit=limit, terminal_decision={"reason": reason}
            )
        ),
    )
    report = runner.run_osworld_task(
        task={"instruction": "Task"},
        environment_factory=lambda: env,
        app_config=Path("config.yaml"),
        output_dir=tmp_path,
    )
    assert env.closed and env.actions == actions
    assert env.evaluated == (status != "error")
    assert report["score"] == (None if status == "error" else float(actions == ["FAIL"]))


@pytest.mark.parametrize("failure", ["reset", "actor", "persist"])
def test_environment_released_after_failure(monkeypatch, tmp_path, failure):
    env = Environment()
    actor = AsyncMock(
        return_value=SimpleNamespace(
            terminal_status="finish", hit_step_limit=False, terminal_decision={"reason": "completed"}
        )
    )
    monkeypatch.setattr(runner, "_run_actor", actor)

    def fail(*args, **kwargs):
        raise OSError("disk or runtime failure")

    if failure == "reset":
        monkeypatch.setattr(env, "reset", fail)
    if failure == "actor":
        actor.side_effect = RuntimeError("actor failed")
    if failure == "persist":
        monkeypatch.setattr(Path, "write_text", fail)
    try:
        runner.run_osworld_task(
            task={"instruction": "Task"},
            environment_factory=lambda: env,
            app_config=Path("config.yaml"),
            output_dir=tmp_path,
        )
    except OSError:
        assert failure == "persist"
    assert env.closed
