from __future__ import annotations

from types import SimpleNamespace

import pytest

from northcua.evaluation import (
    disable_agentic_evaluator_for_external_script,
    uses_external_script_evaluator,
)


def test_external_script_mode_disables_agentic_evaluator_only_when_explicit() -> None:
    evaluation = SimpleNamespace(enabled=True)
    app = SimpleNamespace(app_config=SimpleNamespace(evaluation=evaluation))

    assert not disable_agentic_evaluator_for_external_script(app, {})
    assert evaluation.enabled
    assert disable_agentic_evaluator_for_external_script(app, {"evaluation_mode": "external_script"})
    assert not evaluation.enabled
    assert uses_external_script_evaluator({"evaluation_mode": "external_script"})


def test_external_script_mode_fails_closed_without_policy() -> None:
    with pytest.raises(ValueError, match="requires an app evaluation policy"):
        disable_agentic_evaluator_for_external_script(SimpleNamespace(), {"evaluation_mode": "external_script"})
