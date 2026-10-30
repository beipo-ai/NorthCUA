from __future__ import annotations

from types import SimpleNamespace

from northcua.agentkit.runtime import (
    AgentRoundRuntime,
    AgentRuntimeCounterState,
    AgentRuntimeStepContext,
    AgentRuntimeStepOutcome,
    AgentSessionRuntime,
    AgentStepRuntime,
)
from northcua.core.enums import TaskStatus
from northcua.core.models import LLMTraceRecord, StepLogRecord, WindowState
from northcua.core.terminal import AgentTerminalDecision, AgentTerminalReason


class _DummyWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, payload: str) -> None:
        self.messages.append(payload)


class _DummyRounds:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            session_id="sess-1",
            pending_round_inputs=[],
            step_logs=[],
            pending_command_results={},
        )
        self._queued = []

    def enqueue_initial_request(self, query: str, metadata=None) -> None:
        self._queued.append(SimpleNamespace(query=query, metadata=metadata))

    def enqueue_followup(self, query: str, metadata=None) -> None:
        self._queued.append(SimpleNamespace(query=query, metadata=metadata))

    def next_round(self):
        if not self._queued:
            return None
        return self._queued.pop(0)


class _DummyFacade:
    def __init__(self, statuses: list[TaskStatus] | None = None, session_id: str = "sess-1") -> None:
        self.rounds = _DummyRounds()
        self.rounds.state.session_id = session_id
        self._statuses = list(statuses or [TaskStatus.FINISH])
        self.drain_calls = 0
        self.run_inputs = []
        self._active_round_input = None
        self.runtime_factory = _DummyRuntimeFactory(self)

    @property
    def session_id(self) -> str:
        return self.rounds.state.session_id

    async def run_round(
        self, round_input, snapshot=None, observation_context=None, remote_tool_dispatcher=None
    ) -> StepLogRecord:
        self.run_inputs.append(round_input)
        status = self._statuses.pop(0) if self._statuses else TaskStatus.FINISH
        return StepLogRecord(
            session_id=self.rounds.state.session_id,
            session_step=0,
            round_step=0,
            round_num=1,
            status=status,
            request=round_input.query,
            result="done" if status != TaskStatus.CONTINUE else "still working",
            llm_trace=LLMTraceRecord(),
            assistant_text="done",
            current_window=WindowState(),
        )

    async def hydrate_state(self) -> None:
        return None

    async def enqueue_initial_request(self, query: str, metadata=None) -> None:
        self.rounds.enqueue_initial_request(query, metadata=metadata)

    async def enqueue_followup(self, query: str, metadata=None) -> None:
        self.rounds.enqueue_followup(query, metadata=metadata)

    async def next_round(self):
        self._active_round_input = self.rounds.next_round()
        return self._active_round_input

    async def record_command_results(self, results) -> None:
        for result in results:
            self.rounds.state.pending_command_results[result.call_id] = result

    async def consume_command_results(self, call_ids):
        if not all(call_id in self.rounds.state.pending_command_results for call_id in call_ids):
            return None
        return [self.rounds.state.pending_command_results.pop(call_id) for call_id in call_ids]

    async def drain_background_tasks(self) -> None:
        self.drain_calls += 1

    def spawn_session(self, session_id: str) -> "_DummyFacade":
        return _DummyFacade(statuses=self._statuses or [TaskStatus.FINISH], session_id=session_id)


class _DummyRuntimeFactory:
    def __init__(self, facade: _DummyFacade) -> None:
        self.facade = facade

    def build_round_runtime(self, *, options):
        counters = AgentRuntimeCounterState(session_id=self.facade.session_id)
        return AgentRoundRuntime(
            session_runtime=AgentSessionRuntime(counters=counters),
            step_runtime=AgentStepRuntime(step_executor=_DummyRuntimeStepExecutor(self.facade, options)),
        )


class _DummyRuntimeStepExecutor:
    def __init__(self, facade: _DummyFacade, options) -> None:
        self.facade = facade
        self.options = options

    async def __call__(
        self,
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[StepLogRecord]:
        round_input = self.facade._active_round_input or SimpleNamespace(query=context.query, metadata={})
        observation_context = self.options.observation_context
        if self.options.observation_provider is not None:
            observation_context = await self.options.observation_provider.collect(
                session_step=context.cursor.session_step,
                previous_steps=list(context.history),
            )
        step = await self.facade.run_round(
            round_input,
            observation_context=observation_context,
            remote_tool_dispatcher=self.options.remote_tool_dispatcher,
        )
        step.session_step = context.cursor.session_step
        step.round_step = context.cursor.round_step
        terminal_decision = AgentTerminalDecision.continue_()
        if step.status == TaskStatus.FINISH:
            terminal_decision = AgentTerminalDecision.finish(final_answer=step.result)
        elif step.status != TaskStatus.CONTINUE:
            terminal_decision = AgentTerminalDecision.fail(
                reason=AgentTerminalReason.RUNTIME_ERROR,
                final_answer=step.result,
            )
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=step,
            terminal_decision=terminal_decision,
        )


class _DummyTransport:
    pass
