"""Portable repository contracts shared by documentation and release builds."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_four_distributions_share_version_license_and_public_dependency():
    metadata = [
        tomllib.loads(path.read_text())["project"]
        for path in [ROOT / "pyproject.toml", *sorted((ROOT / "packages").glob("*/pyproject.toml"))]
    ]
    assert len(metadata) == 4
    assert {item["version"] for item in metadata} == {"0.2.0"}
    assert {item["license"] for item in metadata} == {"Apache-2.0"}
    assert all(item["license-files"] == ["LICENSE", "NOTICE"] for item in metadata)
    source = metadata[0]["optional-dependencies"]["local"]
    assert source == [
        "nexau @ git+https://github.com/hzhua/nexau.git@28511d9f0362191a6c378fd441b2be55309a9e32",
        "ag-ui-protocol==0.1.17",
    ]
    uv_source = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["nexau"]
    assert uv_source == {
        "git": "https://github.com/hzhua/nexau.git",
        "rev": "28511d9f0362191a6c378fd441b2be55309a9e32",
    }
    for path in (ROOT / "packages").glob("*/LICENSE"):
        assert path.read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_paper_snapshot_is_identified_and_unchanged():
    manifest = json.loads((ROOT / "docs/paper/manifest.json").read_text())
    assert hashlib.sha256((ROOT / "docs/paper/main.pdf").read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["title"] == (
        "NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale"
    )
    assert manifest["snapshot_date"] == "2026-10-31"
    assert manifest["page_count"] == 123
    assert manifest["sha256"] == "bf2ce9e665cd7eb2212c0d5635d276aaba46232ad89f9f34f7ce6b2367244611"
    assert "runtime validation" in manifest["status"]
    assert manifest["links"]["harness"] == "https://github.com/beipo-ai/NorthCUA"


@pytest.mark.parametrize("directory,collection", [("brand", "assets"), ("demos", "files")])
def test_bundled_brand_and_recording_assets_match_provenance(directory, collection):
    base = ROOT / "docs/assets" / directory
    manifest = json.loads((base / "manifest.json").read_text())
    for item in manifest[collection]:
        asset = base / item["file"]
        assert asset.stat().st_size == item["bytes"]
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == item["sha256"]
    if directory == "brand":
        report = json.loads((ROOT / "docs/paper/manifest.json").read_text())
        assert manifest["source"]["report_sha256"] == report["sha256"]
        assert manifest["source"]["wordmark_text"] == "NorthCUA"


@pytest.mark.parametrize("resource", sorted((ROOT / "northcua/resources/config").glob("*.yaml")), ids=lambda p: p.name)
def test_installed_config_matches_source_example(resource):
    assert resource.read_bytes() == (ROOT / "config" / resource.name).read_bytes()


def test_document_links_resolve():
    documents = [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "examples").rglob("*.md"),
        *(ROOT / "skills").rglob("*.md"),
        *(ROOT / "packages").rglob("README.md"),
    ]
    broken = []
    for document in documents:
        for match in re.finditer(r"\[[^\]]*\]\(([^\s)]+)\)", document.read_text()):
            target = match.group(1).strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            destination = unquote(target.split("#", 1)[0])
            if destination and not (document.parent / destination).exists():
                broken.append(f"{document.relative_to(ROOT)}: {target}")
    assert not broken, broken


def test_document_artwork_has_provenance_and_matching_hashes():
    manifest = json.loads((ROOT / "docs/assets/manifest.json").read_text())
    for item in manifest["figures"]:
        image = ROOT / "docs/assets" / item["file"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == item["sha256"]
        assert "not a recorded agent run" in item["kind"]
        assert image.stat().st_size < 1024 * 1024
    for path in (ROOT / "packages").glob("*/NOTICE"):
        assert path.read_bytes() == (ROOT / "NOTICE").read_bytes()
