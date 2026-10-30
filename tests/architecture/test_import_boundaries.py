from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shared_layers_do_not_import_concrete_agents() -> None:
    forbidden = _collect_forbidden_imports(
        roots=("northcua/runtime", "northcua/platforms", "northcua/agentkernel"),
        forbidden_prefixes=("northcua.agents",),
    )

    assert forbidden == []
    assert (
        _collect_forbidden_string_references(
            roots=("northcua/runtime", "northcua/platforms", "northcua/agentkernel"),
            forbidden_fragments=("northcua.agents",),
        )
        == []
    )


def test_device_package_does_not_hard_import_main_northcua_package() -> None:
    forbidden = _collect_forbidden_imports(
        roots=("northcua_device",),
        forbidden_prefixes=("northcua",),
        top_level_only=True,
    )

    assert forbidden == []


def test_removed_compatibility_entrypoints_stay_removed() -> None:
    removed_modules = (
        "northcua/agentkit/budget.py",
        "northcua/agentkit/terminal.py",
        "northcua/agents/desktop/evaluation/config.py",
        "northcua/agents/desktop/evaluation/tools.py",
        "northcua/agents/desktop/image_payloads.py",
        "northcua/batch/preparation/cleanup.py",
        "northcua/controller/execution/processes/fake_worker_main.py",
        "northcua/runtime/mcp/desktop_native/models.py",
        "northcua/runtime/accessibility/protocol.py",
        "northcua/platforms/projection.py",
        "northcua/transport/ws/external_tools.py",
        "northcua/transport/ws/local_stepwise.py",
        "northcua_runtime_shared/desktop_surface_a11y.py",
        "northcua_runtime_shared/desktop_surface_actions.py",
        "northcua_runtime_shared/desktop_surface_data.py",
        "northcua_runtime_shared/desktop_surface_models.py",
    )
    assert [path for path in removed_modules if (REPO_ROOT / path).exists()] == []

    forbidden = [
        finding
        for finding in _collect_forbidden_string_references(
            roots=("northcua", "northcua_runtime_shared", "tests"),
            forbidden_fragments=(
                "northcua.agentkit.budget",
                "northcua.agentkit.terminal",
                "northcua.agents.desktop.evaluation.config",
                "northcua.agents.desktop.evaluation.tools",
                "northcua.agents.desktop.image_payloads",
                "northcua.batch.preparation.cleanup",
                "northcua.controller.execution.processes.fake_worker_main",
                "northcua.runtime.mcp.desktop_native.models",
                "northcua.runtime.accessibility.protocol",
                "northcua.platforms.projection",
                "northcua.transport.ws.external_tools",
                "northcua.transport.ws.local_stepwise",
                "northcua_runtime_shared.desktop_surface_a11y",
                "northcua_runtime_shared.desktop_surface_actions",
                "northcua_runtime_shared.desktop_surface_data",
                "northcua_runtime_shared.desktop_surface_models",
            ),
        )
        if not finding.startswith("tests/architecture/test_import_boundaries.py ")
    ]
    assert forbidden == []


def _collect_forbidden_imports(
    *,
    roots: tuple[str, ...],
    forbidden_prefixes: tuple[str, ...],
    top_level_only: bool = False,
) -> list[str]:
    findings: list[str] = []
    for root in roots:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = _iter_top_level_import_names(tree) if top_level_only else _iter_import_names(tree)
            for imported_name in names:
                if any(
                    imported_name == prefix or imported_name.startswith(f"{prefix}.") for prefix in forbidden_prefixes
                ):
                    findings.append(f"{path.relative_to(REPO_ROOT)} imports {imported_name}")
    return sorted(findings)


def _collect_forbidden_string_references(
    *,
    roots: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> list[str]:
    findings: list[str] = []
    for root in roots:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in forbidden_fragments:
                if fragment in text:
                    findings.append(f"{path.relative_to(REPO_ROOT)} references {fragment!r}")
    return sorted(findings)


def _iter_import_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _iter_top_level_import_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names
