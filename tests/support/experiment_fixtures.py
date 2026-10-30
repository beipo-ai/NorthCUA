"""Golden Chrome/Calc/GIMP profile definitions."""

from __future__ import annotations

import uuid

from northcua.experiments.contracts import (
    ExperimentRunConfig,
    ExperimentAppProfile,
    ExperimentControllerSpec,
    ExperimentSpec,
    ExperimentTaskSpec,
    SandboxFileInjectionSpec,
    TraceExpectationSpec,
)
from northcua.sandboxes.contracts import SandboxCommandSpec, SandboxProviderSpec, SandboxResourceSpec
from northcua_protocol.capabilities import Capability, PlatformKind

_REMOTE_ROOT = "/tmp/northcua-golden"
_GOLDEN_TASK_MAX_STEPS = 100


def _detached_gui_command(*, label: str, argv: str, log_name: str, wait_seconds: int) -> str:
    """Build a sandbox startup command that launches a GUI app without blocking E2B."""

    return (
        "bash -lc '"
        f"mkdir -p {_REMOTE_ROOT}; "
        "export DISPLAY=${DISPLAY:-:0}; "
        f"setsid {argv} </dev/null >{_REMOTE_ROOT}/{log_name}.out 2>{_REMOTE_ROOT}/{log_name}.err & "
        f"echo $! > {_REMOTE_ROOT}/{log_name}.pid; "
        f"sleep {wait_seconds}; "
        f"echo {label}-launched pid=$(cat {_REMOTE_ROOT}/{log_name}.pid)'"
    )


def golden_e2b_experiment_spec(*, public_controller_url: str = "", provider: str = "e2b") -> ExperimentSpec:
    """Return the bundled three-app golden experiment spec.

    Args:
        public_controller_url: Public websocket URL reachable from E2B.
        provider: Sandbox provider id. Tests pass ``fake``; live runs use
            ``e2b``.

    Returns:
        Experiment spec containing Chrome, LibreOffice Calc, and GIMP tasks.
    """

    # Keep the top-level spec readable; individual app tasks carry long
    # startup commands and fixture wiring that are easier to verify in helpers.
    return ExperimentSpec(
        experiment_id=f"e2b-golden-osworld-demo-{uuid.uuid4().hex[:8]}",
        description="Three-sandbox golden run for Chrome, LibreOffice Calc, and GIMP.",
        controller=ExperimentControllerSpec(
            public_url=public_controller_url,
            insecure_dev_no_auth=provider == "fake",
        ),
        sandbox=_golden_sandbox(provider),
        concurrency=3,
        trace_expectations=TraceExpectationSpec(evaluator_kind="desktop_round_completion"),
        tasks=[_chrome_fixture_task(), _calc_fixture_task(), _gimp_fixture_task()],
        metadata={
            "purpose": "golden_e2b_evaluator_data_collection",
            "required_capabilities": [
                Capability.DESKTOP_INPUT.value,
                Capability.DESKTOP_SCREENSHOT.value,
                Capability.DESKTOP_WINDOW.value,
                Capability.A11Y_TREE.value,
            ],
        },
    )


def _golden_sandbox(provider: str) -> SandboxProviderSpec:
    return SandboxProviderSpec(
        provider=provider,
        template="northcua_golden_osworld_demo",
        desktop=True,
        platform=PlatformKind.LINUX,
        resources=SandboxResourceSpec(timeout_seconds=900, cpu_count=2, memory_mb=4096),
        resolution=(1280, 900),
        device_python="python3.12",
        metadata={"suite": "e2b-golden-osworld-demo"},
    )


def _chrome_fixture_task() -> ExperimentTaskSpec:
    return ExperimentTaskSpec(
        task_id="chrome-fixture-heading",
        app_profile=ExperimentAppProfile.CHROME,
        query=(
            "Google Chrome should already be open to file:///tmp/northcua-golden/chrome_fixture.html. "
            "Read the page and finish by reporting the main heading shown on the page."
        ),
        max_steps=_GOLDEN_TASK_MAX_STEPS,
        fixtures=[_fixture("chrome_fixture.html")],
        startup_commands=[
            SandboxCommandSpec(
                label="open-chrome-fixture",
                command=_detached_gui_command(
                    label="open-chrome-fixture",
                    argv=(
                        "env CHROME_NO_DEFAULT_BROWSER_CHECK=1 "
                        "google-chrome --no-sandbox --disable-dev-shm-usage "
                        "--user-data-dir=/tmp/northcua-golden/chrome-profile "
                        "--no-first-run --no-default-browser-check --disable-fre "
                        "--disable-sync --disable-search-engine-choice-screen "
                        "--disable-features=SigninIntercept,ChromeWhatsNewUI,Translate "
                        "--new-window file:///tmp/northcua-golden/chrome_fixture.html"
                    ),
                    log_name="chrome",
                    wait_seconds=8,
                ),
                timeout_seconds=45,
            )
        ],
        timeout_seconds=900,
        metadata={"golden_app": "chrome"},
    )


def _calc_fixture_task() -> ExperimentTaskSpec:
    return ExperimentTaskSpec(
        task_id="calc-total-csv",
        app_profile=ExperimentAppProfile.CALC,
        query=(
            "LibreOffice Calc should already be open with /tmp/northcua-golden/calc_fixture.csv. "
            "Confirm the three values in column B are visible, compute their total, and finish by reporting the total."
        ),
        max_steps=_GOLDEN_TASK_MAX_STEPS,
        fixtures=[_fixture("calc_fixture.csv")],
        startup_commands=[
            SandboxCommandSpec(
                label="open-calc-fixture",
                command=_detached_gui_command(
                    label="open-calc-fixture",
                    argv="libreoffice --calc --norestore /tmp/northcua-golden/calc_fixture.csv",
                    log_name="calc",
                    wait_seconds=12,
                ),
                timeout_seconds=60,
            )
        ],
        timeout_seconds=900,
        metadata={"golden_app": "libreoffice-calc"},
    )


def _gimp_fixture_task() -> ExperimentTaskSpec:
    return ExperimentTaskSpec(
        task_id="gimp-open-image",
        app_profile=ExperimentAppProfile.GIMP,
        query=(
            "GIMP should already be open with /tmp/northcua-golden/gimp_fixture.ppm. Confirm the image is "
            "visible, make one simple visible edit if a drawing tool is ready, and finish by reporting the "
            "visible final state."
        ),
        max_steps=_GOLDEN_TASK_MAX_STEPS,
        fixtures=[_fixture("gimp_fixture.ppm")],
        startup_commands=[
            SandboxCommandSpec(
                label="open-gimp-fixture",
                command=_detached_gui_command(
                    label="open-gimp-fixture",
                    argv="gimp /tmp/northcua-golden/gimp_fixture.ppm",
                    log_name="gimp",
                    wait_seconds=18,
                ),
                timeout_seconds=90,
            )
        ],
        timeout_seconds=900,
        metadata={"golden_app": "gimp"},
    )


def _fixture(name: str) -> SandboxFileInjectionSpec:
    return SandboxFileInjectionSpec(generated_name=name, remote_path=f"{_REMOTE_ROOT}/{name}", mode="0644")


async def run_fixture_experiment(
    operations,
    *,
    public_controller_url,
    output_dir,
    concurrency=3,
    provider="fake",
    dry_run=False,
    require_real_e2b=False,
):
    spec = golden_e2b_experiment_spec(public_controller_url=public_controller_url, provider=provider)
    return await operations.run(
        ExperimentRunConfig(
            spec=spec.model_copy(update={"concurrency": concurrency}),
            output_dir=output_dir,
            public_controller_url=public_controller_url,
            dry_run=dry_run,
            require_real_e2b=require_real_e2b,
        )
    )
