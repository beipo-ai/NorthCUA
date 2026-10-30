from __future__ import annotations

import pytest

from northcua.agentinfra.context.bootstrap import ContextSideEffectFence
from northcua.agentkernel.context.domain.enums import ContextManagementMode
from northcua.agentkernel.context.strategies import ContextSideEffectReplayError


def test_enforce_side_effect_fence_prevents_candidate_replay() -> None:
    fence = ContextSideEffectFence(mode=ContextManagementMode.ENFORCE)
    fence.mark_external_tool_dispatch()

    with pytest.raises(ContextSideEffectReplayError, match="fallback is disabled"):
        fence.raise_if_candidate_replay_unsafe(TimeoutError("provider timeout"))


def test_off_side_effect_fence_preserves_legacy_fallback() -> None:
    fence = ContextSideEffectFence(mode=ContextManagementMode.OFF)
    fence.mark_external_tool_dispatch()
    fence.raise_if_candidate_replay_unsafe(TimeoutError("provider timeout"))

    assert not fence.started
