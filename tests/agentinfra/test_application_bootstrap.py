from __future__ import annotations

import asyncio

from nexau.archs.main_sub.execution.middleware.context_compaction import ContextCompactionMiddleware

from northcua.agents.desktop.application import NorthCUAApplication


def test_application_bootstrap_wires_session_manager_and_tracer(tmp_path, desktop_native_test_session) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    agent_yaml = config_dir / "agent.yaml"
    agent_yaml.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
system_prompt: You are NorthCUA.
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
mcp_servers:
  - name: DesktopNativeDataExecutor
    type: stdio
    model_visible: false
    command: python3
  - name: DesktopNativeA11yExecutor
    type: stdio
    model_visible: false
    command: python3
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.context_compaction:ContextCompactionMiddleware
    params:
      max_context_tokens: 200000
      auto_compact: true
      threshold: 0.75
      compaction_strategy: tool_result_compaction
      keep_iterations: 2
""".strip(),
        encoding="utf-8",
    )

    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text(
        """
agent_config_path: agent.yaml
artifacts:
  root_dir: test-logs
websocket:
  host: 127.0.0.1
  port: 9999
execution:
  user_id: bootstrap-user
  session_backend: memory
  enable_in_memory_tracer: true
""".strip(),
        encoding="utf-8",
    )

    app = NorthCUAApplication.from_app_config(app_yaml, session_id="bootstrap-session")

    assert app.session_manager is not None
    assert app.agent_config.resolved_tracer is not None
    assert app.facade.user_id == "bootstrap-user"
    assert app.websocket_server.runtime_mode.value == "local"
    assert app.agent_config.middlewares is not None
    assert any(isinstance(middleware, ContextCompactionMiddleware) for middleware in app.agent_config.middlewares)


def test_application_bootstrap_optionally_enables_langfuse_tracer(tmp_path, desktop_native_test_session) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    agent_yaml = config_dir / "agent.yaml"
    agent_yaml.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
mcp_servers:
  - name: DesktopNativeDataExecutor
    type: stdio
    model_visible: false
    command: python3
  - name: DesktopNativeA11yExecutor
    type: stdio
    model_visible: false
    command: python3
""".strip(),
        encoding="utf-8",
    )

    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text(
        """
agent_config_path: agent.yaml
execution:
  enable_in_memory_tracer: true
  enable_langfuse_tracer: true
  langfuse_public_key: pk-test
  langfuse_secret_key: sk-test
  langfuse_host: https://langfuse.example.com
  langfuse_tags:
    - integration-test
  experiment_id: exp-test
  experiment_tags:
    - nightly
""".strip(),
        encoding="utf-8",
    )

    app = NorthCUAApplication.from_app_config(app_yaml, session_id="langfuse-session")

    tracer_names = {tracer.__class__.__name__ for tracer in app.agent_config.tracers}
    assert "InMemoryTracer" in tracer_names
    assert "LangfuseTracer" in tracer_names
    langfuse_tracer = next(
        tracer for tracer in app.agent_config.tracers if tracer.__class__.__name__ == "LangfuseTracer"
    )
    assert langfuse_tracer._init_public_key == "pk-test"
    assert langfuse_tracer._init_secret_key == "sk-test"
    assert langfuse_tracer._init_host == "https://langfuse.example.com"
    assert "integration-test" in langfuse_tracer.tags
    assert "nightly" in langfuse_tracer.tags
    assert "experiment:exp-test" in langfuse_tracer.tags
    assert langfuse_tracer.metadata["experiment_id"] == "exp-test"
    assert app.agent_config.resolved_tracer is not None


def test_application_stop_drains_spawned_server_facades(tmp_path, monkeypatch, desktop_native_test_session) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    agent_yaml = config_dir / "agent.yaml"
    _agent_yaml = """
name: NorthCUAAgent
description: Desktop agent
system_prompt: You are NorthCUA.
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
mcp_servers:
  - name: DesktopNativeDataExecutor
    type: stdio
    model_visible: false
    command: python3
  - name: DesktopNativeA11yExecutor
    type: stdio
    model_visible: false
    command: python3
"""
    agent_yaml.write_text(_agent_yaml.strip(), encoding="utf-8")

    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text(
        """
agent_config_path: agent.yaml
artifacts:
  root_dir: test-logs
websocket:
  host: 127.0.0.1
  port: 9999
execution:
  user_id: bootstrap-user
  session_backend: memory
""".strip(),
        encoding="utf-8",
    )

    app = NorthCUAApplication.from_app_config(app_yaml, session_id="bootstrap-session")
    spawned = app.facade.spawn_session("spawned-session")
    app.server._facades[spawned.session_id] = spawned
    drained: list[str] = []

    async def _drain_root() -> None:
        drained.append(app.facade.session_id)

    async def _drain_spawned() -> None:
        drained.append(spawned.session_id)

    monkeypatch.setattr(app.facade, "drain_background_tasks", _drain_root)
    monkeypatch.setattr(spawned, "drain_background_tasks", _drain_spawned)

    asyncio.run(app.stop())

    assert set(drained) == {"bootstrap-session", "spawned-session"}
    assert app.server._facades == {}
