"""Run public local OSWorld task JSON without a hosted dataset service."""

import argparse
import json
from pathlib import Path

from .runner import run_osworld_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, action="append", required=True, help="Upstream task JSON; repeatable")
    parser.add_argument("--environment-config", type=Path, required=True, help="DesktopEnv constructor kwargs as JSON")
    parser.add_argument("--app-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/osworld"))
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    from desktop_env.desktop_env import DesktopEnv  # pyright: ignore[reportMissingImports]

    config = json.loads(args.environment_config.read_text())
    if not isinstance(config, dict):
        parser.error("environment config must be a JSON object")
    failed = False
    for path in args.task:
        report = run_osworld_task(
            task=json.loads(path.read_text()),
            environment_factory=lambda: DesktopEnv(**config),
            app_config=args.app_config,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
        )
        print(json.dumps(report, ensure_ascii=False))
        failed |= bool(report["error"] or report["cleanup_error"] or report["score"] != 1.0)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
