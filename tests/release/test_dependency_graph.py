"""Dependency submission must keep marker variants and accepted risks visible."""

import json

import pytest

from scripts.submit_dependency_graph import build_snapshot, main


def test_snapshot_retains_every_platform_version_and_reviewed_risk():
    lock = b"""[[package]]
name = "anyio"
version = "4.14.2"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "anyio"
version = "4.16.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "transformers"
version = "4.57.6"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "anyio" }]
[[package]]
name = "northcua"
version = "0.2.0"
source = { editable = "." }
dependencies = [{ name = "anyio" }]
[package.optional-dependencies]
dev = [{ name = "transformers" }]
"""
    result = build_snapshot(lock, "a" * 40, "123")
    manifest = result["manifests"]["uv.lock"]
    assert manifest["file"]["source_location"] == "uv.lock"
    assert len(manifest["resolved"]) == 3
    assert manifest["resolved"]["transformers@4.57.6"]["package_url"] == "pkg:pypi/transformers@4.57.6"
    assert manifest["resolved"]["transformers@4.57.6"]["dependencies"] == ["anyio@4.14.2", "anyio@4.16.0"]
    assert manifest["resolved"]["transformers@4.57.6"]["relationship"] == "direct"
    assert result["metadata"]["non_registry_packages"] == 1
    assert result["sha"] == "a" * 40
    assert len(result["metadata"]["lock_sha256"]) == 64
    assert "TOKEN" not in json.dumps(result)


@pytest.mark.parametrize(
    "repository,ref", [("another/repo", "refs/heads/main"), ("beipo-ai/NorthCUA", "refs/pull/1/merge")]
)
def test_submission_rejects_untrusted_context_before_reading_credentials(monkeypatch, repository, ref):
    monkeypatch.setenv("GITHUB_REPOSITORY", repository)
    monkeypatch.setenv("GITHUB_REF", ref)
    monkeypatch.delenv("DEPENDENCY_GRAPH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="canonical main"):
        main()
