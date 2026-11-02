"""Audit every locked PyPI version and enforce reviewed, expiring risk exceptions.

The scanner runs in an isolated tool environment. Reports retain every finding;
no scanner ignore flag is used. Registry packages are never installed by this audit.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, UTC
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_groups(lock: dict) -> tuple[list[list[str]], list[dict]]:
    """Separate marker-dependent versions so pip-audit sees every exact version."""
    versions: dict[str, set[str]] = {}
    non_registry = []
    for package in lock["package"]:
        name, version, source = package["name"], package["version"], package["source"]
        if source.get("registry") == "https://pypi.org/simple":
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
                raise ValueError("Invalid package requirement in lock")
            versions.setdefault(normalized(name), set()).add(version)
        elif (name == "nexau" and "git" in source) or (
            name in {"northcua", "northcua-device", "northcua-protocol", "northcua-runtime-shared"}
            and "editable" in source
        ):
            non_registry.append(
                {"name": name, "version": version, "scope": "source review, not a PyPI advisory lookup"}
            )
        else:
            raise ValueError("Unrecognized dependency source; review before transmitting package metadata")
    if not versions:
        raise ValueError("No registry dependencies found")
    groups = [[] for _ in range(max(map(len, versions.values())))]
    for name, values in sorted(versions.items()):
        for index, version in enumerate(sorted(values)):
            groups[index].append(f"{name}=={version}")
    return groups, non_registry


def load_exceptions(path: Path, today: date) -> list[dict]:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported dependency-risk policy")
    records = policy["exceptions"]
    for record in records:
        if not all(record.get(key) for key in ("package", "version", "ids", "owner", "reason", "reference")):
            raise ValueError("Dependency-risk exception lacks review metadata")
        if today >= date.fromisoformat(record["expires"]):
            raise ValueError("Dependency-risk exception expired; a new review is required")
        if any(not re.fullmatch(r"(?:CVE|GHSA|PYSEC)-[A-Za-z0-9-]+", item) for item in record["ids"]):
            raise ValueError("Dependency-risk exception must list exact advisory identifiers")
    return records


def evaluate_report(report: dict, requirements: list[str], exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    expected = set(requirements)
    observed = set()
    accepted, unexpected = [], []
    for dependency in report["dependencies"]:
        if dependency.get("skip_reason"):
            raise ValueError("Advisory lookup skipped a locked dependency")
        package, version = normalized(dependency["name"]), dependency["version"]
        observed.add(f"{package}=={version}")
        for finding in dependency["vulns"]:
            identifiers = {finding["id"], *finding.get("aliases", [])}
            reviewed = any(
                package == normalized(record["package"])
                and version == record["version"]
                and identifiers <= set(record["ids"])
                for record in exceptions
            )
            item = {"package": package, "version": version, "id": finding["id"], "aliases": sorted(identifiers)}
            (accepted if reviewed else unexpected).append(item)
    if observed != expected:
        raise ValueError("Advisory report does not cover exactly the requested locked versions")
    return accepted, unexpected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    parser.add_argument("--exceptions", type=Path, default=ROOT / ".github/security/dependency-exceptions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dependency-audit")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date()
    try:
        exceptions = load_exceptions(args.exceptions, today)
        groups, non_registry = requirement_groups(tomllib.loads(args.lock.read_text()))
        summary = {
            "checked_on": str(today),
            "non_registry": non_registry,
            "accepted": [],
            "unexpected": [],
            "errors": [],
        }
        summary["audited_versions"] = sum(map(len, groups))
        for index, requirements in enumerate(groups):
            request = args.output / f"requirements-{index}.txt"
            request.write_text("\n".join(requirements) + "\n")
            for service in ("pypi", "osv"):
                result = args.output / f"{service}-{index}.json"
                log = args.output / f"{service}-{index}.log"
                command = [
                    "uvx",
                    "--from",
                    "pip-audit==2.10.1",
                    "pip-audit",
                    "--requirement",
                    str(request),
                    "--no-deps",
                    "--disable-pip",
                    "--vulnerability-service",
                    service,
                    "--timeout",
                    "10",
                    "--format",
                    "json",
                    "--output",
                    str(result),
                ]
                try:
                    # A failed lookup must not consume a previous run's successful report.
                    result.unlink(missing_ok=True)
                    with log.open("w") as stream:
                        completed = subprocess.run(command, stdout=stream, stderr=stream, timeout=240, check=False)
                    if completed.returncode not in (0, 1) or not result.is_file():
                        raise ValueError("Advisory service or scanner failed; inspect the retained log")
                    accepted, unexpected = evaluate_report(json.loads(result.read_text()), requirements, exceptions)
                    summary["accepted"].extend({**item, "service": service} for item in accepted)
                    summary["unexpected"].extend({**item, "service": service} for item in unexpected)
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                    summary["errors"].append({"service": service, "group": index, "error": type(error).__name__})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"Audited {summary['audited_versions']} locked PyPI versions with both services; "
            f"accepted risk records: {len(summary['accepted'])}; "
            f"unreviewed findings: {len(summary['unexpected'])}; errors: {len(summary['errors'])}. "
            "Accepted records are disclosed risks, not a claim of zero vulnerabilities."
        )
        return int(bool(summary["unexpected"] or summary["errors"]))
    except (OSError, ValueError, KeyError) as error:
        print(f"Dependency audit cannot complete: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
