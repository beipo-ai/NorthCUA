from __future__ import annotations

import pytest
from pydantic import ValidationError

from northcua.agentkernel.context.application import ContextBudgetPlanner
from northcua.agentkernel.context.domain.config import ContextBudgetConfig, ContextManagementConfig
from northcua.agentkernel.context.domain.enums import ContextManagementMode
from northcua.core.config import NorthCUAAppConfig


def test_context_management_defaults_to_strict_off() -> None:
    config = NorthCUAAppConfig(agent_config_path="agent.yaml")

    assert config.context_management.mode == ContextManagementMode.OFF
    assert config.context_management.summary.minimum_reduction_ratio == 0.01
    assert config.context_management.resolve("desktop").mode == ContextManagementMode.OFF


def test_agent_override_is_typed_and_does_not_change_global_default() -> None:
    config = ContextManagementConfig.model_validate(
        {
            "agent_overrides": {
                "desktop": {"mode": "enforce", "policy_profile": "desktop"},
                "evaluator": {"mode": "shadow", "policy_profile": "evaluator"},
            }
        }
    )

    assert config.mode == ContextManagementMode.OFF
    assert config.resolve("desktop").mode == ContextManagementMode.ENFORCE
    assert config.resolve("evaluator").mode == ContextManagementMode.SHADOW
    assert config.resolve("web_round_evaluator").mode == ContextManagementMode.SHADOW
    assert config.resolve("desktop_round_completion_evaluator").policy_profile == "evaluator"
    assert config.resolve("third-party-agent").mode == ContextManagementMode.OFF


def test_context_config_rejects_unknown_and_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ContextManagementConfig.model_validate({"unknown": True})
    with pytest.raises(ValidationError):
        ContextManagementConfig.model_validate({"budget": {"target_ratio": 0.8, "soft_trigger_ratio": 0.7}})
    with pytest.raises(ValidationError):
        ContextManagementConfig.model_validate({"summary": {"model": {"source": "agent_config"}}})


def test_budget_formula_reserves_output_and_safety_margin() -> None:
    budget = ContextBudgetPlanner().resolve(
        context_limit=200_000,
        output_reserve=2_000,
        config=ContextBudgetConfig(),
    )

    assert budget.safety_margin == 15_840
    assert budget.hard_input_budget == 182_160
    assert budget.soft_trigger_tokens == 127_512
    assert budget.target_tokens == 100_188


def test_budget_rejects_capacity_exhausted_by_reserves() -> None:
    with pytest.raises(ValueError, match="exhausted"):
        ContextBudgetPlanner().resolve(
            context_limit=1_000,
            output_reserve=100,
            config=ContextBudgetConfig(safety_margin_min_tokens=900),
        )


def test_partial_agent_override_deep_merges_global_nested_config() -> None:
    config = ContextManagementConfig.model_validate(
        {
            "budget": {"soft_trigger_ratio": 0.8},
            "summary": {"max_output_tokens": 3000, "timeout_seconds": 30, "total_deadline_seconds": 60},
            "agent_overrides": {
                "web": {
                    "budget": {"recent_iterations": 9},
                    "summary": {"timeout_seconds": 7},
                }
            },
        }
    )

    resolved = config.resolve("web")

    assert resolved.budget.soft_trigger_ratio == 0.8
    assert resolved.budget.recent_iterations == 9
    assert resolved.summary.max_output_tokens == 3000
    assert resolved.summary.timeout_seconds == 7
    assert resolved.summary.total_deadline_seconds == 60


def test_checkpoint_backend_is_closed_enum() -> None:
    with pytest.raises(ValidationError):
        ContextManagementConfig.model_validate({"checkpoint": {"backend": "nonsense"}})
