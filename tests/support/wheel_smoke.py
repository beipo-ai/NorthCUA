"""Run with a wheel-only interpreter, outside the checkout; no test dependencies."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--with-nexau", action="store_true")
    args = parser.parse_args()
    import northcua

    assert "site-packages" in str(northcua.__file__), northcua.__file__
    for name in ("northcua", "northcua-device", "northcua-protocol", "northcua-runtime-shared"):
        assert metadata.version(name) == "0.2.0"
        packaged = {str(path) for path in metadata.distribution(name).files or []}
        assert any(path.endswith("/licenses/LICENSE") for path in packaged), name
        assert any(path.endswith("/licenses/NOTICE") for path in packaged), name
    assert (importlib.util.find_spec("nexau") is not None) == args.with_nexau
    if args.with_nexau:
        direct = json.loads(metadata.distribution("nexau").read_text("direct_url.json"))
        assert direct["url"] == "https://github.com/hzhua/nexau.git"
        assert direct["vcs_info"]["commit_id"] == "28511d9f0362191a6c378fd441b2be55309a9e32"
        from northcua.agents.desktop.prompt import PromptEnvelopeBuilder
        from northcua.core.config import PromptPolicyConfig
        from northcua.core.coordinates import CoordinatePolicy
        from northcua.core.enums import CoordinateSpace
        from northcua.core.models import WindowState
        import northcua.agents.desktop.prompt as prompt_package

        templates = Path(prompt_package.__file__).with_name("templates")
        assert "site-packages" in str(templates)
        envelope = PromptEnvelopeBuilder(
            templates,
            PromptPolicyConfig(),
            coordinate_policy=CoordinatePolicy(CoordinateSpace.NORMALIZED_1000),
        ).build(
            task="Open Calculator",
            platform_name="linux",
            current_window=WindowState(),
            history_turns=[],
            tools=[],
            tool_choice="auto",
        )
        assert "For 0-1000 coordinates" in envelope.system_prompt
        assert "(1000, 1000)" in envelope.system_prompt
        assert "Open Calculator" in envelope.persistent_user_text
    with tempfile.TemporaryDirectory(prefix="northcua-wheel-smoke-") as temporary:
        root = Path(temporary)
        env = dict(os.environ, NORTHCUA_MODEL="fixture-model", OPENAI_API_KEY="fixture-key")
        env.pop("PYTHONPATH", None)

        def run(*arguments):
            return subprocess.run(
                [sys.executable, *arguments], cwd=root, env=env, check=True, text=True, capture_output=True, timeout=30
            )

        run("-m", "northcua", "config", "init", "--directory", str(root / "config"))
        config = root / "config" / "app.example.yaml"
        assert config.is_file()
        if args.with_nexau:
            run("-m", "northcua", "config", "validate", "--app-config", str(config))
        run("-m", "northcua_device.cli", "doctor", "--metadata-only")
        run("-m", "northcua_device.cli", "list-tools", "--json")
        from northcua.experiments.materialization.runtime_bundle import RuntimeWheelBundleBuilder

        bundle = RuntimeWheelBundleBuilder(wheelhouse=args.wheelhouse).build(output_dir=root / "bundle")
        assert len(bundle.uploads) == 4
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (root / "controller.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "northcua",
                    "controller",
                    "serve",
                    "--app-config",
                    str(config),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--db-path",
                    str(root / "controller.sqlite"),
                    "--blob-dir",
                    str(root / "blobs"),
                    "--artifact-root-dir",
                    str(root / "artifacts"),
                    "--insecure-dev-no-auth",
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            raise AssertionError(log.read())
                        time.sleep(0.1)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    print("Wheel-only config, worker, controller and runtime bundle checks passed.")


if __name__ == "__main__":
    main()
