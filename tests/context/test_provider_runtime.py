from __future__ import annotations

from typing import cast

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent

import northcua.agentinfra.context.adapters.provider_runtime as runtime_module
from northcua.agentinfra.context.adapters.provider_runtime import resolve_summary_provider_runtime
from northcua.agentkernel.context.domain.config import ContextSummaryConfig


def _llm(model: str) -> LLMConfig:
    return LLMConfig(
        model=model,
        base_url="https://example.invalid/v1",
        api_key="test-key",
        max_tokens=512,
    )


def test_dedicated_summary_config_uses_only_resolved_llm_clients(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "summary-agent.yaml"
    config_path.write_text(
        """name: summary
llm_config:
  model: dedicated-summary
  base_url: https://example.invalid/v1
  api_key: test-key
  max_tokens: 512
  max_context_tokens: 32768
mcp_servers:
  - name: ignored-runtime-server
    model_visible: false
    type: in_memory
""",
        encoding="utf-8",
    )
    sync_client = object()
    async_client = object()
    monkeypatch.setattr(runtime_module, "_create_clients", lambda config: (sync_client, async_client))
    summary_config = ContextSummaryConfig.model_validate(
        {"model": {"source": "agent_config", "agent_config_path": str(config_path)}}
    )

    runtime = resolve_summary_provider_runtime(
        config=summary_config,
        primary_llm_config=_llm("primary"),
        prepared_agent=cast(Agent, object()),
    )

    assert runtime.llm_config.model == "dedicated-summary"
    assert runtime.sync_client is sync_client
    assert runtime.async_client is async_client
    assert runtime.context_limit_tokens == 32768
    assert runtime.owned_clients == (async_client, sync_client)
