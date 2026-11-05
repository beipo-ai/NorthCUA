# Changelog

## Unreleased

- Default Desktop accessibility recording to off, as documented, so the recommended `examples/desktop-settings` recipes start without an accessibility server; `config validate` now runs the local Desktop observation-probe checks and reports an undeclared probe server before a run (#6).

- Link the public arXiv report (2611.02102v1), add an arXiv badge and matching full-author article citations in English/Chinese and GitHub citation metadata; retain the identified offline PDF snapshot.

- Replace the repository showcases with the supplied 30-second Ruby Heart and Blender UFO edits; embed GitHub-native video players while retaining byte-identical offline MP4s and checksummed result frames.

- Remove benchmark task/lifecycle adapters, external scoring callbacks, score aggregation and user-simulator bindings; retain core Desktop/Web Harness behavior and recommended Desktop settings.
- Remove all OSWorld-specific prompts, prompt-profile selectors and coordinate-text rewriting. Retain one ordinary Desktop prompt, general action-coordinate mapping and response-policy recipes; resolve dedicated summary-model paths relative to the app config.

- Move the canonical repository and all installation, citation and package links to `beipo-ai/NorthCUA`.
- Make the English/Chinese README start with a complete no-key browser task; keep model setup explicit and provide a separate agent operator path.
- Update NorthCUA-Pro availability, exact public revision, weight-license attribution and separate-serving guidance after the 3 November check.
- Scan fetched Git history and tracked source exports in CI with a checksum-pinned Gitleaks release; add community conduct and GitHub Actions dependency-update configuration.
- Synchronize the detached-descendant test fixture before early exit, eliminating a process-observation race while preserving cleanup and unrelated-process assertions.
- Upgrade compatible affected dependencies, enforce the Pillow 13 minimum in source and device wheels, and isolate pre-commit from the runtime dependency graph.
- Retain public NexAU by maintainer decision; disclose six known Transformers advisories under an exact-version/ID exception expiring 3 December, with raw PyPI/OSV findings retained and new risks blocked.
- Import POSIX-only `fcntl` conditionally in context trace and checkpoint persistence so Windows Desktop runs start; without `fcntl` these writers are serialized within the process, matching single-process local agents (#5).

## 0.2.0 — source release, 2026-10-31

- Publish the Desktop/Web NorthCUA Harness as a standalone source repository with a fresh Git history and Apache-2.0 licensing.
- Align the logo, research citations and mechanism figures with the 123-page NorthCUA Technical Report dated 2026-10-31. Bundle the exact report and selected recorded project showcases.
- Provide English/Chinese READMEs, model and resource guides, from-zero instructions, portable model templates and an agent operator skill.
- Retain local, WebSocket and controller/device tasks; context compaction, sliding image history, working notes, live outcome evaluation and recoverable local batches.
- Include optional Desktop prompt settings, bounded same-context response policies and generic harness-local extension tools.
- Correct callback dispatch/history, safe retry-artifact paths, cancellation, permanent-error classification, mutually exclusive retry policies and retry-input isolation.
- Pin public NexAU to `hzhua/nexau@28511d9f0362191a6c378fd441b2be55309a9e32`, with compatible MCP 2 / FastMCP 4 and locked dependencies.
- Build four coordinated wheels and define regression, real-browser, isolated native-X11 and installed-wheel CI lanes. The release workflow builds checked artifacts; it does not claim a PyPI publication.

See [release verification](docs/release-0.2.0.md) for executed checks and review scope.
Historical preparation records remain in the documentation archive and do not
substitute for the current source snapshot's validation.
