"""Keep GitHub's dependency graph aligned with every public version in uv.lock.

Runs only on trusted canonical main. It parses metadata without installing or
importing any dependency and never removes reviewed-risk packages from the graph.
"""

from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import json
import os
import subprocess
import tomllib
from urllib.parse import quote
from urllib.request import Request, urlopen

from scripts.audit_dependencies import ROOT, normalized, requirement_groups

REPOSITORY = "beipo-ai/NorthCUA"


def build_snapshot(lock_bytes: bytes, sha: str, run_id: str) -> dict:
    lock = tomllib.loads(lock_bytes.decode())
    groups, non_registry = requirement_groups(lock)
    direct = set()
    for package in lock["package"]:
        if package["name"] in {"northcua", "northcua-device", "northcua-protocol", "northcua-runtime-shared"}:
            declared = list(package.get("dependencies", []))
            for extra in package.get("optional-dependencies", {}).values():
                declared.extend(extra)
            direct.update(normalized(dependency["name"]) for dependency in declared)
    resolved = {}
    by_name: dict[str, list[str]] = {}
    for requirement in (value for group in groups for value in group):
        name, version = requirement.split("==", 1)
        key = f"{name}@{version}"
        by_name.setdefault(name, []).append(key)
        resolved[key] = {
            "package_url": f"pkg:pypi/{quote(name, safe='._-')}@{quote(version, safe='._-')}",
            "relationship": "direct" if name in direct else "indirect",
            "scope": "runtime",
            "dependencies": [],
        }
    for package in lock["package"]:
        key = f"{normalized(package['name'])}@{package['version']}"
        if key not in resolved:
            continue
        edges = set()
        for dependency in package.get("dependencies", []):
            choices = by_name.get(normalized(dependency["name"]), [])
            edges.update(
                value
                for value in choices
                if not dependency.get("version") or value.endswith("@" + dependency["version"])
            )
        resolved[key]["dependencies"] = sorted(edges)
    return {
        "version": 0,
        "sha": sha,
        "ref": "refs/heads/main",
        "scanned": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job": {
            "id": run_id,
            "correlator": "northcua-uv-lock-main",
            "html_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
        },
        "detector": {
            "name": "northcua-uv-lock",
            "version": "1.0.0",
            "url": f"https://github.com/{REPOSITORY}/blob/main/scripts/submit_dependency_graph.py",
        },
        "metadata": {
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "scope": "all locked PyPI versions across platforms and extras, including development dependencies",
            "non_registry_packages": len(non_registry),
        },
        "manifests": {"uv.lock": {"name": "uv.lock", "file": {"source_location": "uv.lock"}, "resolved": resolved}},
    }


def main() -> None:
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY or os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise SystemExit("Dependency submission is restricted to canonical main")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if sha != os.environ.get("GITHUB_SHA"):
        raise SystemExit("Checked-out revision differs from the workflow revision")
    snapshot = build_snapshot((ROOT / "uv.lock").read_bytes(), sha, os.environ["GITHUB_RUN_ID"])
    output = ROOT / "reports/dependency-graph-snapshot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n")
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}/dependency-graph/snapshots",
        data=json.dumps(snapshot).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["DEPENDENCY_GRAPH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "NorthCUA-dependency-graph",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        receipt = json.load(response)
    count = len(snapshot["manifests"]["uv.lock"]["resolved"])
    print(f"Submitted {count} locked public versions for {sha}; receipt {receipt.get('id')}.")


if __name__ == "__main__":
    main()
