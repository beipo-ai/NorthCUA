# Dependency security and the retained NexAU constraint

Reviewed **3 November 2026**. The maintainer chose to retain the public NexAU
upstream and disclose the remaining dependency risk rather than publish a fork.
**This release does not claim zero known dependency vulnerabilities.**

## The accepted upstream risk

NexAU is pinned to
[`hzhua/nexau@28511d9f0362191a6c378fd441b2be55309a9e32`](https://github.com/hzhua/nexau/tree/28511d9f0362191a6c378fd441b2be55309a9e32).
Its package metadata requires `transformers>=4.40,<5`. The lock therefore retains
`transformers==4.57.6`; the following known advisories require a newer line to
leave their affected ranges. Transformers 5.10 covers these reported ranges,
but conflicts with that upstream constraint. We do not override the dependency
metadata or use `--no-deps` to fabricate a compatible installation.

| Retained advisory | Upstream affected-range boundary |
|---|---|
| [CVE-2025-14929 / PYSEC-2025-217](https://osv.dev/vulnerability/PYSEC-2025-217) | Last affected: 5.0.0rc0 |
| [CVE-2026-1839 / GHSA-69w3-r845-3855](https://github.com/advisories/GHSA-69w3-r845-3855) | Fixed: 5.0.0rc3 |
| [CVE-2026-4372 / GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | Fixed: 5.3.0 |
| [CVE-2026-5241 / GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | Fixed: 5.5.0 |
| [CVE-2026-9856 / GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | Fixed: 5.10.0 |
| [CVE-2026-80047 / GHSA-x9r9-c232-4q39](https://github.com/advisories/GHSA-x9r9-c232-4q39) | Last affected: 5.8.1 |

These are six advisory groups, not nine different vulnerabilities: advisory
services may return multiple records or aliases for the same issue.

## What this means when running tasks

The normal Harness Chat/Responses paths do not load a Hugging Face tokenizer.
That limits their direct exposure to the loading paths; it does not make the
installed Transformers package safe for arbitrary use.

NexAU's `generate_with_token` path can call
`TokenTraceSession._get_hf_tokenizer → AutoTokenizer.from_pretrained`. Independent
offline testing of 4.57.6 reproduced a nested LightGlue configuration forwarding
`trust_remote_code=True` despite a caller's `False`, stopping before any download
or code execution. This is an actual reachable trust-boundary issue, not merely
an unused training-class warning. Other listed advisories concern additional
model, checkpoint, generation or save paths.

Use trusted model/tokenizer configuration and controlled local files. Do not load
untrusted Hub repositories, custom tokenizer configurations or checkpoint files
inside the Harness environment. Keep model serving in a separate, appropriately
updated inference environment. These precautions reduce exposure; they are not
a substitute for upgrading the affected library.

## How CI handles the decision

[The exception policy](../../.github/security/dependency-exceptions.json) names
the exact package/version, all 17 known IDs and aliases, the maintainer owner,
reason and this disclosure. It expires on **3 December 2026** and must be reviewed
before that date. It is not a general exemption for Transformers or NexAU.

The [audit command](../../scripts/audit_dependencies.py) queries both PyPI and
OSV through pinned pip-audit, retains the raw findings and reports accepted risks
separately. New IDs or aliases, a different affected version, other vulnerable
packages, missing lookups, service failures and an expired policy fail the gate.
There is no scanner `--ignore-vuln` switch. A green CI badge means these rules
passed; it does not mean the known exceptions were fixed.

Run the same gate locally with uv 0.13.6+:

```sh
uv run --no-sync python -m scripts.audit_dependencies
```

The audit sends only public package names and versions. NexAU's VCS revision and
the four workspace packages receive source/provenance review instead of PyPI
version lookups. Pre-commit runs in an isolated tool environment so its newer
environment-builder dependencies do not conflict with NexAU's runtime graph.

Remove this exception when a compatible public NexAU revision permits a fixed
Transformers version, then rerun source, installed-wheel and actual task checks.
