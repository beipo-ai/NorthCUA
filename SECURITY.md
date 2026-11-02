# Security

This policy covers the NorthCUA Harness distributed in the NorthCUA repository.

Computer-use tasks can change files, accounts and remote services. Run tasks in an
appropriate isolated environment and grant only the tools/capabilities they require.
Use authenticated TLS controller connections outside loopback development. Do not expose
an unauthenticated controller or a personal browser profile to untrusted tasks.

Model credentials belong on the controller/model host. Provider keys and sandbox tokens
must not be committed to configuration or evidence. Artifacts can contain task data and
screenshots even when credential redaction is enabled; control access and retention.
Working-memory audit files are task artifacts and follow the same access policy.

The release retains tool-schema validation, read-only probe restrictions, path boundaries,
URL policy, lease ownership, artifact size limits and cleanup ownership checks. Those
controls do not make arbitrary shell tools or model instructions a security sandbox.

The public release does not support older private CLI/config APIs.
Report vulnerabilities privately to repository maintainers using GitHub private reporting
when enabled, or email
[zhangchaoyun@qijizhifeng.com](mailto:zhangchaoyun@qijizhifeng.com) or
[libingxuan@qijizhifeng.com](mailto:libingxuan@qijizhifeng.com).
Dependency fixes must rerun the affected provider/runtime and wheel checks.

## Checks before publication

CI scans the fetched Git history and a tracked source export with checksum-pinned
[Gitleaks](https://github.com/gitleaks/gitleaks). Findings fail the job and reports
redact matched values. This does not inspect pixels in screenshots or video,
prove that every secret format is detectable, or make task artifacts safe to share.
Review research media, document metadata and distribution contents separately.

Dependency auditing queries both PyPI and OSV for every locked public package
version. The retained NexAU constraint currently requires a reviewed exception
for six Transformers advisories; read the [exact scope and practical risk](docs/security/dependencies.md).
Raw findings remain visible. New risks, failed lookups or expired exceptions
block CI; a passing policy gate is not a claim of zero known vulnerabilities.

Keep `.env`, local configuration, run traces, private inventories and build
environments outside version control. Use the published templates and retain
environment-variable references for keys. A sample credential in a security test
is synthetic test data, not a key for operating the project.
