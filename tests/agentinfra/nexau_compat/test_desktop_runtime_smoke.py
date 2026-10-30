from __future__ import annotations

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine

from northcua.agents.desktop.runtime import DesktopRuntimeOptions
from northcua.transport.ws.config import NorthCUAWebSocketTransportConfig
from northcua.transport.ws.transport import NorthCUAWebSocketTransport


@pytest.mark.asyncio
async def test_desktop_transport_constructs_finalized_public_nexau_agent() -> None:
    engine = InMemoryDatabaseEngine.get_shared_instance()
    agent_config = AgentConfig(
        name="macos-desktop-runtime-smoke",
        description="offline Desktop runtime boundary smoke",
        system_prompt="stable desktop system",
        llm_config=LLMConfig(
            model="fixture-model",
            base_url="https://example.invalid",
            api_key="fixture-key",
            api_type="openai_responses",
        ),
    )
    transport = NorthCUAWebSocketTransport(
        engine=engine,
        session_manager=SessionManager(engine=engine),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=agent_config,
    )

    agent = await transport.prepare_agent(
        user_id="macos-desktop-runtime-user",
        session_id="macos-desktop-runtime-session",
        agent_config=agent_config,
    )
    try:
        assert DesktopRuntimeOptions()
        assert getattr(agent, "__northcua_runtime_compat_finalized__", False) is True
        assert getattr(agent.executor.llm_caller, "__northcua_provider_compat_installed__", False) is True
    finally:
        agent.sync_cleanup()
