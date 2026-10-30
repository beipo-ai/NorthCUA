import base64
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from nexau.core.messages import Message, Role, TextBlock, ToolUseBlock

from northcua.agents.desktop.prompt.builders import PromptEnvelopeBuilder
from northcua.agents.desktop.prompt.envelope import HistoryTurn
from northcua.agentinfra.providers.reasoning_replay import ReasoningReplayPolicy, reasoning_replay_policy_from_config
from northcua.core.config import NorthCUAAppConfig
from northcua.core.config_loader import NorthCUAConfigLoader
from northcua.core.coordinates import CoordinatePolicy, CoordinateMapper, CoordinateFrame, CoordinatePoint
from northcua.core.enums import CoordinateSpace
from northcua.core.models import WindowState
from northcua.integrations.osworld1 import run_task
from northcua.integrations.osworld1.coordinates import normalize_message, normalize_text
from northcua.integrations.osworld1.prompt import PROMPT_SHA256, prompt_path

ROOT = Path(__file__).resolve().parents[3]


def config(space="normalized_1000"):
    return NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "fixture.yaml",
            "prompt": {"system_prompt_profile": "osworld1_formal", "history_window_rounds": 100},
            "desktop": {"coordinate_policy": {"input_space": space}},
            "evaluation": {"enabled": False},
        }
    )


def test_single_prompt_sha_and_portable_config(monkeypatch):
    assert hashlib.sha256(prompt_path().read_bytes()).hexdigest() == PROMPT_SHA256
    assert len(list(prompt_path().parent.glob("*.j2"))) == 1
    monkeypatch.setenv("NORTHCUA_MODEL", "fixture")
    monkeypatch.setenv("NORTHCUA_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    bootstrap = NorthCUAConfigLoader.from_app_config(ROOT / "examples/osworld1/app.yaml")
    assert bootstrap.app.execution.model_response_recovery
    assert bootstrap.app.execution.no_tool_call_attempts == 1
    assert bootstrap.app.execution.max_agent_iterations_per_round == 100
    assert bootstrap.agent.llm_config.max_retries == 3
    assert bootstrap.agent.llm_config.temperature == 0.7
    assert (
        reasoning_replay_policy_from_config(bootstrap.agent.llm_config) == ReasoningReplayPolicy.CHAT_REASONING_CONTENT
    )


@pytest.mark.parametrize("space", ["normalized_1000", "image_pixels"])
def test_prompt_image_coordinates_and_original_roles(space):
    buf = BytesIO()
    Image.new("RGB", (1920, 1080)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    cfg = config(space)
    policy = CoordinatePolicy(CoordinateSpace(space))
    builder = PromptEnvelopeBuilder(
        ROOT / "northcua/agents/desktop/prompt/templates", cfg.prompt, coordinate_policy=policy
    )
    turn = HistoryTurn(
        assistant_text="Clicked at (960, 540)",
        reasoning="fixture reasoning",
        assistant_tool_calls=[
            {"id": "c", "type": "function", "function": {"name": "click", "arguments": '{"x":500,"y":500}'}}
        ],
        tool_results=[{"tool_call_id": "c", "name": "click", "content": "Clicked at (960, 540)", "image_url": url}],
    )
    envelope = builder.build(
        task="Open Save",
        platform_name="osworld",
        current_window=WindowState(),
        current_clean_screenshot_url=url,
        history_turns=[turn],
        tools=[],
        tool_choice="auto",
    )
    normalized = space == "normalized_1000"
    assert ("1000 x 1000" if normalized else "1920 x 1080") in envelope.system_prompt
    assert "User query:\nOpen Save" in envelope.persistent_user_text
    assert "Long-lived instruction:" not in envelope.system_prompt
    assert "Current observation:" in envelope.observation_user_text
    assert "[INFEASIBLE]" in envelope.system_prompt and "ask_user" not in envelope.system_prompt
    text = " ".join(b.text for m in envelope.history_messages for b in m.content if isinstance(b, TextBlock))
    assert ("(500, 500)" if normalized else "(960, 540)") in text
    uses = [b for m in envelope.history_messages for b in m.content if isinstance(b, ToolUseBlock)]
    assert uses[0].input == {"x": 500, "y": 500}
    assert turn.assistant_text == "Clicked at (960, 540)"  # original evidence not mutated
    assert envelope.history_messages[0].metadata["reasoning"][0]["summary"][0]["text"] == "fixture reasoning"
    point = CoordinateMapper.to_frame_point(CoordinatePoint(500, 500), policy=policy, frame=CoordinateFrame(1920, 1080))
    assert point.as_int_tuple() == ((960, 540) if normalized else (500, 500))


def test_normalization_preserves_call_and_reasoning_metadata():
    message = Message(
        role=Role.ASSISTANT,
        content=[
            TextBlock(text="x=960 y=540 1920x1080"),
            ToolUseBlock(id="c", name="click", input={"x": 500, "y": 500}),
        ],
        metadata={"fixture": "raw"},
    )
    converted = normalize_message(message, 1920, 1080)
    assert converted.content[0].text == "x=500 y=500 1000 x 1000"
    assert converted.content[1] == message.content[1]
    assert converted.metadata == message.metadata
    assert normalize_text("delta_x=100 delta_y=200", 1920, 1080) == "delta_x=100 delta_y=200"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,status,limited,action",
    [
        ("completed", "finish", False, "DONE"),
        ("infeasible", "fail", False, "FAIL"),
        ("max_steps", "fail", True, None),
        ("model_reported_failure", "fail", False, None),
        ("empty_response", "fail", False, None),
        ("runtime_error", "error", False, None),
    ],
)
async def test_prepared_actor_boundary_and_scoring(reason, status, limited, action):
    app = SimpleNamespace(
        app_config=config(),
        run_query=AsyncMock(
            return_value=SimpleNamespace(
                terminal_status=status, terminal_decision={"reason": reason}, hit_step_limit=limited
            )
        ),
    )
    evaluate = AsyncMock(return_value=0.5)
    result = await run_task(app=app, session_id="fixture", task_id="task", query="Original query", evaluate=evaluate)
    invalid = reason in {"empty_response", "runtime_error"}
    assert result["score"] == (None if invalid else 0.5)
    assert evaluate.await_count == (0 if invalid else 1)
    if not invalid:
        evaluate.assert_awaited_once_with(action)
    request = app.run_query.call_args.args[0]
    assert request.query == "Original query" and request.max_steps == request.max_session_steps == 100


@pytest.mark.asyncio
async def test_evaluator_error_never_becomes_zero():
    app = SimpleNamespace(
        app_config=config(),
        run_query=AsyncMock(
            return_value=SimpleNamespace(
                terminal_status="finish", terminal_decision={"reason": "completed"}, hit_step_limit=False
            )
        ),
    )
    result = await run_task(
        app=app, session_id="s", task_id="t", query="Task", evaluate=AsyncMock(side_effect=OSError("offline"))
    )
    assert result["score"] is None and not result["score_valid"] and "offline" in result["error"]
