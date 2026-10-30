from __future__ import annotations

import asyncio
from pathlib import Path

from northcua.agentkit import (
    AgentArtifactPaths,
    AgentCliSpec,
    AgentPresentationSpec,
    AgentRenderContext,
    AgentRunStatus,
    AgentRuntimeSpec,
    AgentSessionSpec,
    AgentSpec,
    ArtifactLayoutSpec,
    PromptProjector,
    SessionAdapter,
    StepwiseToolAgentTemplate,
    StepwiseToolStepContext,
    StepwiseToolStepResult,
    TemplateTerminalRenderer,
    resolve_panel_projector,
    resolve_terminal_renderer,
    run_with_terminal_renderer,
)
from northcua.agentkit.agent_config import clone_agent_config_with_hooks
from northcua.agentkit.registry import get_agentkit_registry, reset_agentkit_registry_for_tests
from northcua_protocol.capabilities import PlatformKind
from tests.agentkit.test_agentkit_lowcode import (
    AndroidObservationProvider,
    AndroidPanelProjector,
    AndroidPromptProjector,
)


def test_presentation_renderer_resolves_from_agent_spec() -> None:
    reset_agentkit_registry_for_tests()

    web_renderer = resolve_terminal_renderer("web")
    web_projector = resolve_panel_projector("web")

    assert web_renderer.__class__.__name__ == "WebAgentTerminalRenderer"
    assert web_projector.__class__.__name__ == "IdentityPanelProjector"


def test_custom_panel_projector_is_applied_before_render_step(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            presentation=AgentPresentationSpec(
                renderer_cls=TemplateTerminalRenderer,
                panel_projector_cls=AndroidPanelProjector,
            ),
        )
    )
    emitted: list[str] = []

    async def _run() -> None:
        async def _agent(callbacks: object) -> dict[str, object]:
            callbacks.on_before_step(1)  # type: ignore[attr-defined]
            callbacks.on_step("raw step")  # type: ignore[attr-defined]
            return {"status": "success"}

        await run_with_terminal_renderer(
            context=AgentRenderContext(
                agent_kind="android",
                display_name="Android CUA",
                session_id="sess-android",
                log_dir=tmp_path,
                query="Project a panel",
            ),
            renderer=resolve_terminal_renderer("android"),
            projector=resolve_panel_projector("android"),
            run=_agent,
            emit=emitted.append,
        )

    asyncio.run(_run())

    assert "projected raw step" in "\n".join(emitted)
    assert "Step 7" in "\n".join(emitted)


def test_template_terminal_renderer_and_run_helper_support_lowcode_agents(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
        presentation=AgentPresentationSpec(renderer_cls=TemplateTerminalRenderer),
    )
    get_agentkit_registry().register(spec)
    emitted: list[str] = []

    async def _run() -> None:
        renderer = resolve_terminal_renderer("android")
        context = AgentRenderContext(
            agent_kind="android",
            display_name="Android CUA",
            session_id="sess-android",
            log_dir=tmp_path,
            query="Open settings",
        )

        async def _agent(callbacks: object) -> dict[str, object]:
            callbacks.on_before_step(1)  # type: ignore[attr-defined]
            callbacks.on_step({"step": 1, "status": "ok", "summary": "opened settings"})  # type: ignore[attr-defined]
            return {"status": "success", "step_count": 1}

        result = await run_with_terminal_renderer(
            context=context,
            renderer=renderer,
            run=_agent,
            emit=emitted.append,
        )

        assert result["status"] == "success"

    asyncio.run(_run())

    output = "\n".join(emitted)
    assert "Android CUA Run" in output
    assert "Step 1" in output
    assert "opened settings" in output
    assert "Android CUA Summary" in output


def test_clone_agent_config_with_hooks_preserves_runtime_objects() -> None:
    tracer = object()
    middleware = object()
    before_hook = object()
    after_hook = object()
    tool_hook = object()
    from nexau.archs.llm.llm_config import LLMConfig
    from nexau.archs.main_sub.config import AgentConfig

    base = AgentConfig(
        name="agentkit-test",
        description="test",
        system_prompt="test",
        llm_config=LLMConfig(
            model="gpt-5.4",
            base_url="http://localhost:3001/v1",
            api_key="key",
            api_type="openai_responses",
        ),
    )
    base.tracers = [tracer]  # type: ignore[list-item]
    base.middlewares = [middleware]  # type: ignore[list-item]

    cloned = clone_agent_config_with_hooks(
        base,
        before_model_hooks=[before_hook],
        after_model_hooks=[after_hook],
        after_tool_hooks=[tool_hook],
    )

    assert cloned is not base
    assert cloned.tracers == [tracer]
    assert cloned.middlewares == [middleware]
    assert cloned.before_model_hooks[-1] is before_hook
    assert cloned.after_model_hooks[-1] is after_hook
    assert cloned.after_tool_hooks[-1] is tool_hook
    assert base.before_model_hooks in (None, [])


def test_stepwise_template_runs_android_fake_agent_without_core_edits(tmp_path: Path) -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
        session=AgentSessionSpec(),
        artifacts=ArtifactLayoutSpec(screenshot_dir="shots", dom_dir="trees"),
    )

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        assert context.observation["screen"] == "settings-1"
        assert context.messages[0]["role"] == "system"  # type: ignore[index]
        return StepwiseToolStepResult(status="ok", summary="settings inspected", done=True)

    async def _run() -> None:
        template = StepwiseToolAgentTemplate(
            spec=spec,
            observation_provider=AndroidObservationProvider(),
            prompt_projector=AndroidPromptProjector(),
            step_executor=_executor,
        )
        renderer = TemplateTerminalRenderer(spec)
        emitted: list[str] = []
        context = AgentRenderContext(
            agent_kind="android",
            display_name="Android CUA",
            session_id="sess-android",
            log_dir=tmp_path,
            query="Inspect Settings",
        )

        async def _agent(callbacks: object) -> object:
            return await template.run(
                query="Inspect Settings",
                max_steps=5,
                on_before_step=callbacks.on_before_step,  # type: ignore[attr-defined]
                on_step=callbacks.on_step,  # type: ignore[attr-defined]
            )

        result = await run_with_terminal_renderer(
            context=context,
            renderer=renderer,
            run=_agent,
            emit=emitted.append,
        )
        assert result.status == "success"
        assert len(result.steps) == 1
        assert result.steps[0].step_index == 1
        assert result.steps[0].summary == "settings inspected"
        assert "Step 1" in "\n".join(emitted)

    asyncio.run(_run())

    paths = AgentArtifactPaths.from_layout(tmp_path / "sess-android", spec.artifacts)
    assert paths.screenshot_dir == tmp_path / "sess-android" / "shots"
    assert paths.dom_dir == tmp_path / "sess-android" / "trees"


def test_stepwise_template_respects_max_steps_history_and_closes_on_error() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
    )
    closed: list[bool] = []
    seen_history: list[int] = []

    class _Session(SessionAdapter):
        async def close(self) -> None:
            closed.append(True)

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        seen_history.append(len(context.messages[1]["content"]["history"]))  # type: ignore[index]
        return StepwiseToolStepResult(status="ok", summary=f"step {context.step_index}", done=False)

    class _Projector(PromptProjector):
        def project(
            self, *, query: str, observation: object, history: tuple[object, ...] = ()
        ) -> list[dict[str, object]]:
            return [
                {"role": "system", "content": query},
                {"role": "user", "content": {"observation": observation, "history": list(history)}},
            ]

    async def _run_max_steps() -> None:
        result = await StepwiseToolAgentTemplate(
            spec=spec,
            session=_Session(agent_kind="android"),
            observation_provider=AndroidObservationProvider(),
            prompt_projector=_Projector(),
            step_executor=_executor,
        ).run(query="Inspect Settings", max_steps=2)
        assert result.status == AgentRunStatus.MAX_STEPS
        assert [step.step_index for step in result.steps] == [1, 2]

    asyncio.run(_run_max_steps())

    assert seen_history == [0, 1]
    assert closed == [True]

    async def _failing_executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        del context
        raise RuntimeError("boom")

    async def _run_error() -> None:
        import pytest

        with pytest.raises(RuntimeError, match="boom"):
            await StepwiseToolAgentTemplate(
                spec=spec,
                session=_Session(agent_kind="android"),
                observation_provider=AndroidObservationProvider(),
                prompt_projector=_Projector(),
                step_executor=_failing_executor,
            ).run(query="Inspect Settings", max_steps=1)

    asyncio.run(_run_error())

    assert closed == [True, True]
