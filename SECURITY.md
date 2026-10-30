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

The current release candidate has no promise of supporting older private CLI/config APIs.
Report vulnerabilities privately to repository maintainers using GitHub private reporting
when enabled. Dependency fixes must rerun the affected provider/runtime and wheel checks.
