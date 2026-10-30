from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help_builds_without_nexau_dependency() -> None:
    """Base/controller installs should build CLI help without importing NexAU."""

    repo_root = Path(__file__).resolve().parents[2]
    script = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "nexau" or name.startswith("nexau."):
        raise ModuleNotFoundError("No module named 'nexau'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from northcua.cli.parser import build_parser

parser = build_parser()
help_text = parser.format_help()
assert "NorthCUA command-line interface" in help_text
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_agentkit_evaluator_specs_import_without_nexau_dependency() -> None:
    """Low-code evaluator specs should not require NexAU runtime imports."""

    repo_root = Path(__file__).resolve().parents[2]
    script = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "nexau" or name.startswith("nexau."):
        raise ModuleNotFoundError("No module named 'nexau'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from northcua.agentkit import EvaluatorSpec, EvidencePolicy, RoundEvaluationReport

spec = EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator")
assert spec.evidence_policy == EvidencePolicy()
assert RoundEvaluationReport.__name__ == "RoundEvaluationReport"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_agentkit_evaluator_client_probe_tools_import_without_nexau_dependency() -> None:
    """Client probe declarations should stay usable without NexAU-backed tool factories."""

    repo_root = Path(__file__).resolve().parents[2]
    script = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "nexau" or name.startswith("nexau."):
        raise ModuleNotFoundError("No module named 'nexau'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from northcua.agentkit.evaluator import standard_client_probe_tools
from northcua.agentkernel.tools.surface import ToolSurfaceService
from northcua_runtime_shared.tool_contracts import ResolvedToolSurface, ToolSurfaceMode

tools = standard_client_probe_tools()
assert tools[0].name == "client_screenshot"
surface = ToolSurfaceService().build_from_resolved_surface(
    ResolvedToolSurface(
        profile_name="empty",
        mode=ToolSurfaceMode.LOCAL,
        contracts=(),
        validation_descriptors=(),
        provider_tools=(),
        route_by_tool_id={},
        visibility={},
        aliases={},
        fingerprints={},
        lossiness_manifest={},
    )
)
assert surface.tools == []
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_controller_core_stays_domain_neutral() -> None:
    """AgentKit controller core should not import bundled agent domains."""

    repo_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "northcua/agentkit/controller_dispatch.py",
        "northcua/agentkit/controller_runner.py",
    ):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "northcua.agents.web" not in source
        assert "northcua_controller_plugins.desktop_runner" not in source
