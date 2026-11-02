"""The publication gate must not silently expand an accepted dependency risk."""

from copy import deepcopy
from datetime import date
import json

import pytest

from scripts.audit_dependencies import evaluate_report, load_exceptions, requirement_groups


def exception():
    return {
        "package": "transformers",
        "version": "4.57.6",
        "ids": ["CVE-2026-5241", "GHSA-fgcw-684q-jj6r"],
        "owner": "maintainer",
        "reason": "upstream constraint",
        "reference": "risk.md",
        "expires": "2026-12-03",
    }


def report(name="transformers", version="4.57.6", advisory="CVE-2026-5241", aliases=None):
    return {"dependencies": [{"name": name, "version": version, "vulns": [{"id": advisory, "aliases": aliases or []}]}]}


def test_groups_cover_each_platform_version_without_conflicting_requirements():
    lock = {
        "package": [
            {"name": "anyio", "version": version, "source": {"registry": "https://pypi.org/simple"}}
            for version in ["4.14.2", "4.16.0"]
        ]
    }
    groups, non_registry = requirement_groups(lock)
    assert groups == [["anyio==4.14.2"], ["anyio==4.16.0"]]
    assert non_registry == []


def test_unknown_registry_is_not_sent_to_an_advisory_service():
    with pytest.raises(ValueError, match="Unrecognized"):
        requirement_groups({"package": [{"name": "private", "version": "1", "source": {"registry": "internal"}}]})


def test_only_exact_known_package_version_and_identifiers_are_accepted():
    result = report(aliases=["GHSA-fgcw-684q-jj6r"])
    accepted, unexpected = evaluate_report(result, ["transformers==4.57.6"], [exception()])
    assert len(accepted) == 1
    assert unexpected == []


@pytest.mark.parametrize("change", ["package", "version", "id", "alias"])
def test_new_risk_cannot_reuse_the_old_exception(change):
    result = report()
    dependency = result["dependencies"][0]
    if change == "package":
        dependency["name"] = "another-package"
    elif change == "version":
        dependency["version"] = "4.57.7"
    elif change == "id":
        dependency["vulns"][0]["id"] = "CVE-2026-999999"
    else:
        dependency["vulns"][0]["aliases"] = ["GHSA-new1-new2-new3"]
    requirements = [f"{dependency['name']}=={dependency['version']}"]
    accepted, unexpected = evaluate_report(result, requirements, [exception()])
    assert accepted == []
    assert len(unexpected) == 1


@pytest.mark.parametrize("today", [date(2026, 12, 3), date(2027, 1, 1)])
def test_expired_exception_blocks_publication(tmp_path, today):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "exceptions": [exception()]}))
    with pytest.raises(ValueError, match="expired"):
        load_exceptions(policy, today)


def test_missing_or_skipped_lookup_cannot_be_reported_as_a_pass():
    with pytest.raises(ValueError, match="exactly"):
        evaluate_report({"dependencies": []}, ["transformers==4.57.6"], [exception()])
    skipped = deepcopy(report())
    skipped["dependencies"][0]["skip_reason"] = "service failed"
    with pytest.raises(ValueError, match="skipped"):
        evaluate_report(skipped, ["transformers==4.57.6"], [exception()])
