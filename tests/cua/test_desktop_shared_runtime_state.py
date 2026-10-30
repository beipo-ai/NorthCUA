from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from northcua.agentkit.runtime import AgentRuntimeCounterState, AgentRuntimeCursor
from northcua.agents.desktop.round_execution.step_persistence import RoundStepPersister
from northcua.agents.desktop.runtime import DesktopRuntimeStateStore
from northcua.agents.desktop.session import NorthCUARoundCoordinator
from northcua.core.models import RoundState


class _FakeArtifactStore:
    response_log_path = Path("responses.jsonl")
    evaluation_log_path = Path("evaluation.jsonl")


class _FakeRoundStateStore:
    def __init__(self, state: RoundState) -> None:
        self.state = state
        self.saved: list[RoundState] = []

    async def load(self, session_id: str, *, response_log_path: Path | None, evaluation_log_path: Path | None):
        assert session_id == self.state.session_id
        assert response_log_path == _FakeArtifactStore.response_log_path
        assert evaluation_log_path == _FakeArtifactStore.evaluation_log_path
        return self.state

    async def save(self, state: RoundState, storage: object | None = None) -> None:
        del storage
        self.saved.append(state)


def test_desktop_runtime_state_store_maps_round_state_and_shared_counters() -> None:
    state = RoundState(session_id="sess-runtime", session_step=7, round_num=3, round_step=4)
    counters = AgentRuntimeCounterState(session_id="stale")
    coordinator = NorthCUARoundCoordinator(session_id="sess-runtime")
    state_store = _FakeRoundStateStore(state)
    runtime_store = DesktopRuntimeStateStore(
        counters=counters,
        round_coordinator=coordinator,
        state_store=state_store,  # type: ignore[arg-type]
        artifact_store=_FakeArtifactStore(),  # type: ignore[arg-type]
    )

    asyncio.run(runtime_store.hydrate())

    assert coordinator.state is state
    assert counters.session_id == "sess-runtime"
    assert counters.next_session_step == 7
    assert counters.current_round_num == 3
    assert counters.next_round_step == 4

    cursor = counters.reserve_step()
    counters.complete_step(cursor)
    asyncio.run(runtime_store.save())

    assert state.session_step == 8
    assert state.round_num == 3
    assert state.round_step == 5
    assert state_store.saved == [state]


def test_round_step_persister_saves_advanced_state_after_post_step_failure() -> None:
    state = RoundState(session_id="sess-runtime", session_step=7, round_num=3, round_step=4)
    coordinator = NorthCUARoundCoordinator(session_id="sess-runtime")
    coordinator.replace_state(state)
    state_store = _FakeRoundStateStore(state)

    class _FakeSessionManager:
        async def get_session(self, *, user_id: str, session_id: str):
            assert user_id == "user-1"
            assert session_id == "sess-runtime"
            return SimpleNamespace(storage={"session": "storage"})

    persister = RoundStepPersister(
        artifact_store=object(),
        state_store=state_store,
        round_coordinator=coordinator,
        session_manager=_FakeSessionManager(),
        user_id="user-1",
        llm_trace_writer=object(),  # type: ignore[arg-type]
    )
    cursor = AgentRuntimeCursor(
        session_id="sess-runtime",
        round_num=3,
        round_step=4,
        session_step=7,
    )

    asyncio.run(persister.save_shared_runtime_failure_state(cursor))

    assert state.session_step == 8
    assert state.round_num == 3
    assert state.round_step == 5
    assert state_store.saved == [state]
