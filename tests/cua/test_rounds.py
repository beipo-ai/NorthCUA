"""Regression coverage for desktop round coordinator state transitions."""

from __future__ import annotations

from northcua.agents.desktop.session import NorthCUARoundCoordinator
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord


def test_round_coordinator_tracks_round_queue_state() -> None:
    coordinator = NorthCUARoundCoordinator("sess-1")
    coordinator.enqueue_initial_request("Open Calculator", metadata={"source": "test"})
    first = coordinator.next_round()
    assert first is not None
    assert first.metadata == {"source": "test"}
    assert coordinator.state.round_num == 1
    assert coordinator.state.session_step == 1
    assert coordinator.state.round_step == 1


def test_round_coordinator_emits_continuation_round_when_resume_results_exist() -> None:
    coordinator = NorthCUARoundCoordinator("sess-1")
    coordinator.enqueue_initial_request("Open Calculator")
    first = coordinator.next_round()
    assert first is not None
    coordinator.state.pending_resume_results = [
        CommandExecutionResultRecord(
            call_id="call-1",
            status=ResultStatus.SUCCESS,
            result={"text": "Clicked"},
        )
    ]

    continuation = coordinator.next_round()

    assert continuation is not None
    assert continuation.query == "Open Calculator"
    assert continuation.metadata == {"resume": True}
    assert coordinator.state.round_num == 1
