"""Import-boundary tests for remote harness protocol/device/controller layers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module_refs_under(path: Path) -> set[str]:
    refs: set[str] = set()
    for file_path in path.rglob("*.py"):
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                refs.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                refs.add(node.module)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                refs.add(node.args[0].value)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "northcua." in node.value:
                refs.add(node.value)
    return refs


def test_protocol_package_has_no_runtime_or_agent_dependencies() -> None:
    imports = _module_refs_under(ROOT / "northcua_protocol")

    forbidden_prefixes = ("northcua.", "northcua_device", "nexau", "pyautogui", "playwright")
    assert not [name for name in imports if name.startswith(forbidden_prefixes)]


def test_device_package_does_not_import_agents_or_nexau() -> None:
    imports = _module_refs_under(ROOT / "northcua_device")

    forbidden_prefixes = ("northcua.agents", "northcua.controller", "nexau")
    assert not [name for name in imports if name.startswith(forbidden_prefixes)]


def test_controller_core_does_not_import_agent_domains_or_device_implementation() -> None:
    imports = _module_refs_under(ROOT / "northcua" / "controller")

    forbidden_prefixes = ("northcua.agents", "northcua_device", "nexau")
    assert not [name for name in imports if name.startswith(forbidden_prefixes)]


def test_concrete_agents_do_not_import_controller_or_device_implementation() -> None:
    imports = _module_refs_under(ROOT / "northcua" / "agents")

    forbidden_prefixes = ("northcua.controller", "northcua_device")
    assert not [name for name in imports if name.startswith(forbidden_prefixes)]


def test_runtime_shared_does_not_import_agents_or_controller() -> None:
    imports = _module_refs_under(ROOT / "northcua_runtime_shared")

    forbidden_prefixes = ("northcua.agents", "northcua.controller", "northcua.runtime", "nexau")
    assert not [
        name
        for name in imports
        if name.startswith(forbidden_prefixes) and not name.startswith("northcua.runtime_shared.")
    ]


def test_repo_has_no_shadow_remote_package_trees() -> None:
    """Keep one canonical source tree per remote-infra package."""

    assert not (ROOT / "packages" / "northcua_protocol").exists()
    assert not (ROOT / "packages" / "northcua_runtime_shared").exists()
    assert not (ROOT / "packages" / "northcua_device").exists()
    for package_name in ("northcua-protocol", "northcua-runtime-shared", "northcua-device"):
        assert not (ROOT / "packages" / package_name / "src").exists()


def test_remote_package_manifests_use_root_source_trees() -> None:
    """Subdistribution manifests should package the canonical root sources."""

    expected = {
        "northcua-protocol": ("northcua_protocol", 'where = ["../.."]'),
        "northcua-runtime-shared": ("northcua_runtime_shared", 'where = ["../.."]'),
        "northcua-device": ("northcua_device", 'where = ["../.."]'),
    }
    for package_name, (import_name, source_root_line) in expected.items():
        manifest = (ROOT / "packages" / package_name / "pyproject.toml").read_text(encoding="utf-8")
        assert f'include = ["{import_name}", "{import_name}.*"]' in manifest
        assert source_root_line in manifest
        assert (ROOT / import_name).is_dir()
