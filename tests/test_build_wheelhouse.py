from __future__ import annotations

from pathlib import Path

from scripts.build_wheelhouse import _clean_build_artifacts


def test_clean_build_artifacts_removes_stale_root_and_package_build_trees(tmp_path: Path) -> None:
    stale_paths = (
        tmp_path / "build" / "lib" / "deleted_runtime_module.py",
        tmp_path / "root.egg-info" / "SOURCES.txt",
        tmp_path / "packages" / "component" / "build" / "lib" / "deleted_component.py",
        tmp_path / "packages" / "component" / "component.egg-info" / "SOURCES.txt",
        tmp_path / "packages" / "src-layout" / "src" / "component.egg-info" / "SOURCES.txt",
    )
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")

    _clean_build_artifacts(root=tmp_path)

    assert not (tmp_path / "build").exists()
    assert not tuple(tmp_path.glob("*.egg-info"))
    assert not tuple((tmp_path / "packages").glob("*/build"))
    assert not tuple((tmp_path / "packages").glob("*/*.egg-info"))
    assert not tuple((tmp_path / "packages").glob("*/src/*.egg-info"))
