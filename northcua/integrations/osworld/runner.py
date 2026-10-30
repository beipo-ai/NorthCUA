"""Run an upstream task with a fresh DesktopEnv and its authoritative evaluator."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class DesktopEnvironment(Protocol):
    vm_ip: str
    server_port: int

    def reset(self, *, task_config: dict) -> object: ...
    def step(self, action: str, pause: float = 2) -> object: ...
    def evaluate(self) -> float: ...
    def close(self) -> None: ...


async def _run_actor(
    *, app_config: Path, environment: DesktopEnvironment, query: str, output_dir: Path, session_id: str, max_steps: int
):
    from northcua.agents.desktop.application import DesktopApplication
    from northcua.agentinfra.runtime.protocol import DesktopRunRequest
    from northcua.core.config_loader import NorthCUAConfigLoader

    bootstrap = NorthCUAConfigLoader.from_app_config(app_config)
    config = bootstrap.app.model_copy(deep=True)
    config.desktop.backend = "osworld"
    config.desktop.osworld.vm_ip = environment.vm_ip
    config.desktop.osworld.server_port = environment.server_port
    config.artifacts.root_dir = str(output_dir)
    app = DesktopApplication(config, bootstrap.agent, session_id=session_id)
    try:
        return await app.run_query(DesktopRunRequest(session_id=session_id, query=query, max_steps=max_steps))
    finally:
        await app.stop()


def run_osworld_task(
    *,
    task: dict,
    environment_factory: Callable[[], DesktopEnvironment],
    app_config: Path,
    output_dir: Path,
    max_steps: int = 50,
) -> dict:
    """Own one environment from creation through oracle evaluation and release.

    Runs synchronously, so upstream reset/evaluate operations cannot outlive a
    cancelled async waiter and race resource cleanup. Use separate processes
    for concurrent OSWorld tasks. Upstream task JSON is passed through intact.
    """
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("OSWorld task JSON requires a non-empty instruction")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    session_id = f"osworld-{uuid.uuid4().hex}"
    output_dir = Path(output_dir).resolve()
    result_path = output_dir / session_id / "osworld_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=False)
    report = {
        "session_id": session_id,
        "task_id": task.get("id", ""),
        "score": None,
        "actor_status": "not_started",
        "error": "",
        "cleanup_error": "",
    }
    environment = None
    try:
        environment = environment_factory()
        environment.reset(task_config=task)
        result = asyncio.run(
            _run_actor(
                app_config=app_config,
                environment=environment,
                query=instruction,
                output_dir=output_dir,
                session_id=session_id,
                max_steps=max_steps,
            )
        )
        status = str(result.terminal_status or "").upper()
        report.update(
            actor_status=status, hit_step_limit=result.hit_step_limit, terminal_decision=dict(result.terminal_decision)
        )
        reason = result.terminal_decision.get("reason", "")
        if reason == "completed" and not result.hit_step_limit:
            environment.step("DONE", pause=0)
        elif reason == "infeasible" and not result.hit_step_limit:
            environment.step("FAIL", pause=0)
        elif status == "ERROR":
            raise RuntimeError("Actor runtime failed; oracle evaluation was not performed")
        # Step limits do not imply an infeasibility claim. The oracle sees no FAIL action.
        report["score"] = float(environment.evaluate())
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        # Release even when report persistence fails (for example, a full disk).
        try:
            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            if environment is not None:
                try:
                    environment.close()
                except Exception as exc:
                    report["cleanup_error"] = str(exc)
                    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
