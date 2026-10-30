import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northcua.agents.desktop.prompt.builders import PromptEnvelopeBuilder
from northcua.core.config import NorthCUAAppConfig
from northcua.core.coordinates import CoordinatePolicy
from northcua.core.enums import CoordinateSpace
from northcua.integrations.osworld2 import run_task, score_record, summarize_scores
from northcua.integrations.osworld2.prompt import SYSTEM_TEMPLATE_SHA256, system_template_path


def formal_config():
    return NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "fixture.yaml",
            "prompt": {"system_prompt_profile": "osworld2_formal"},
            "execution": {"no_tool_call_attempts": 3},
            "evaluation": {"enabled": False},
            "desktop": {"coordinate_policy": {"input_space": "normalized_1000"}},
        }
    )


def test_single_frozen_prompt_and_runtime_query():
    assert hashlib.sha256(system_template_path().read_bytes()).hexdigest() == SYSTEM_TEMPLATE_SHA256
    config = formal_config()
    root = Path(__file__).resolve().parents[3] / "northcua/agents/desktop/prompt/templates"
    builder = PromptEnvelopeBuilder(
        root, config.prompt, coordinate_policy=CoordinatePolicy(CoordinateSpace.NORMALIZED_1000)
    )
    system = builder.system_builder.render(
        1920,
        1080,
        desktop_environment="osworld",
        home_directory="/home/user",
        coordinate_policy=CoordinatePolicy(CoordinateSpace.NORMALIZED_1000),
    )
    assert "normalized 0-1000" in system and "ask_user" in system
    assert "{{" not in system
    assert builder.observation_builder.render_persistent("  original query\n") == "  original query\n"
    assert "[INFEASIBLE]" not in builder.observation_builder.render_persistent("task")
    assert len(list(system_template_path().parent.glob("*.j2"))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,status,limited,expected",
    [
        ("completed", "finish", False, 0),
        ("max_steps", "fail", True, 0),
        ("model_reported_failure", "fail", False, 0),
        ("empty_response", "fail", False, None),
        ("runtime_error", "error", False, None),
        ("runtime_error", "fail", False, None),
        ("", None, False, None),
        ("content_filter", "fail", False, None),
    ],
)
async def test_same_app_original_query_limit_and_error_accounting(reason, status, limited, expected):
    app = SimpleNamespace(
        app_config=formal_config(),
        run_query=AsyncMock(
            return_value=SimpleNamespace(
                terminal_status=status, terminal_decision={"reason": reason}, hit_step_limit=limited
            )
        ),
    )
    evaluate = AsyncMock(return_value=0)
    result = await run_task(
        app=app, session_id="fixture-session", task_id="fixture-task", query="original task", evaluate=evaluate
    )
    assert result["score"] == expected
    assert evaluate.await_count == (0 if expected is None else 1)
    request = app.run_query.call_args.args[0]
    assert request.query == "original task" and request.max_steps == 500 and request.max_session_steps == 500
    # No environment factory/reset/step/close/release interface exists on this app fixture.


@pytest.mark.asyncio
async def test_scorer_failure_not_zero():
    app = SimpleNamespace(
        app_config=formal_config(),
        run_query=AsyncMock(
            return_value=SimpleNamespace(
                terminal_status="finish", terminal_decision={"reason": "completed"}, hit_step_limit=False
            )
        ),
    )
    result = await run_task(
        app=app,
        session_id="fixture-session",
        task_id="fixture-task",
        query="task",
        evaluate=AsyncMock(side_effect=OSError("scorer offline")),
    )
    assert result["score"] is None and not result["score_valid"] and "scorer offline" in result["error"]


@pytest.mark.parametrize("raw", [None, float("nan"), float("inf"), -1, 1.1, True, "0"])
def test_invalid_scores_excluded(raw):
    assert not score_record(task_id="t", raw_score=raw)["score_valid"]


def test_summary_denominator_zero_and_duplicate_attempts():
    rows = [
        score_record(task_id="a", raw_score=1),
        score_record(task_id="b", raw_score=0),
        score_record(task_id="c", raw_score=0, error="timeout"),
    ]
    summary = summarize_scores(rows)
    assert summary["mean_score"] == 0.5 and summary["valid_tasks"] == 2 and summary["excluded_tasks"] == 1
    assert summary["valid_zero_tasks"] == 1 and summary["binary_rate"] == 0.5
    assert summarize_scores(rows[2:])["mean_score"] is None
    with pytest.raises(ValueError):
        summarize_scores([rows[0], rows[0]])


def test_ask_user_uses_original_callback_without_synthetic_answers():
    from northcua.integrations.osworld2.tools import ask_user_tool

    questions = []
    tool = ask_user_tool(lambda question: questions.append(question) or "original answer")
    assert tool.implementation(question="Where is the input?") == "original answer"
    assert questions == ["Where is the input?"]
    with pytest.raises(ValueError):
        tool.implementation(question=" ")


def test_portable_example_and_auxiliary_binding(monkeypatch, tmp_path):
    import shutil
    from northcua.integrations.osworld2 import load_config

    root = Path(__file__).resolve().parents[3]
    config_dir = tmp_path / "local-config"
    shutil.copytree(root / "examples/osworld2", config_dir)
    monkeypatch.chdir(tmp_path)
    for name in ("NORTHCUA_MODEL", "OPENAI_API_KEY", "NORTHCUA_SUMMARY_MODEL", "NORTHCUA_SUMMARY_API_KEY"):
        monkeypatch.setenv(name, "fixture")
    for name in ("NORTHCUA_BASE_URL", "NORTHCUA_SUMMARY_BASE_URL"):
        monkeypatch.setenv(name, "https://fixture.invalid/v1")
    bootstrap = load_config(config_dir / "app.yaml")
    assert bootstrap.app.context_management.summary.model.agent_config_path == str(config_dir / "summary-agent.yaml")
    assert bootstrap.agent.llm_config.get_param("extra_body") == {
        "presence_penalty": 1.0,
        "top_k": 20,
        "min_p": 0,
        "repetition_penalty": 1.0,
    }
    assert bootstrap.agent.llm_config.temperature == 0.8
    assert bootstrap.agent.llm_config.timeout == 600
    assert bootstrap.agent.llm_config.max_tokens == 65536
    assert bootstrap.app.execution.no_tool_call_attempts == 3
