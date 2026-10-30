from __future__ import annotations

import subprocess
import sys

from northcua.batch.post_run import PostRunHookRunner, PostRunHookSpec
from northcua.evaluation.script import ScriptEvaluatorRunner, ScriptEvaluatorSpec


def test_script_evaluator_public_api_reuses_existing_command_json_runner() -> None:
    assert ScriptEvaluatorRunner is PostRunHookRunner
    assert ScriptEvaluatorSpec is PostRunHookSpec
    spec = ScriptEvaluatorSpec(hook_id="reward", command="printf '{\"score\": 1}'")
    assert spec.kind.value == "command_json"


def test_script_public_api_does_not_eagerly_load_batch_runner() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from northcua.evaluation.script import ScriptEvaluatorSpec; "
                "assert ScriptEvaluatorSpec; assert 'northcua.batch.runner.service' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
