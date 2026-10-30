# Context, images and working memory

Each model step receives the task, current observations, retained tool exchanges,
selected visual evidence and the active working-memory view. Raw artifacts are stored
separately from the prompt projection.

## Images

Desktop has one observation path: GUI action tools return their post-action screenshot,
which supplies the next model step's clean state. The runtime captures an initial image
when no clean tool result is available. A text-only `note` keeps the grounding image;
it does not replace it with a new visual history entry. Local and remote execution use
this same rule. Optional accessibility recording is separate from image delivery.

`prompt.screenshot_window_images` is a **total** image budget for Desktop and Web.
The single sliding overlay/clean window retains a recent clean state and action overlays
anchored to the pre-action frame. Web Set-of-Mark grounding consumes a slot in the same
budget. At budget 1, the clean state wins; at budget 0, all prompt images are disabled.
Text-only notes never create screenshot slots. Raw post-action images and annotated
pre-action overlays keep separate artifact identities.

There is no per-step probe mode or separate current-user-image projection to select.
Older `observation.policy`, `observation.probe_chains`,
`desktop.use_post_action_image_as_current` and `prompt.image_window_policy` settings
have been removed, as have the unused `desktop.action_post_wait_seconds` and
`desktop.clipboard_first_typing` fields. Delete these fields when migrating a configuration; validation
rejects them rather than silently changing their meaning.

## Semantic compaction

`context_management.mode` is `off`, `shadow` or `enforce`. Off preserves the selected
history policy. Shadow computes diagnostics without replacing the provider prompt.
Enforce budgets the full prompt and summarizes older complete exchanges. System policy,
task, current observation, unfinished tool exchanges and active memory are pinned.
Provider overflow recovery, checkpoints and compaction/summary traces share the same
context infrastructure for Desktop, Web and evaluators.

Compaction must not break a tool-use/tool-result pair or silently remove pinned content.
A budget too small for pinned content fails according to the configured failure policy.
Tune context/model limits for the selected provider instead of assuming every model
supports the sample context size.

## Working memory

The `note` and `web_note` tools accept `text`, `kind`, `reason` and an optional `key`.
Reusing a key replaces its active value. Omitting the key appends an independent entry.
The controller/runner ingests successful note results and persists them transactionally
in `working_memory.sqlite3` under the session artifact directory. The device does not
own a second copy.

Events are idempotent by round, session step and tool-call ID. Old replay events cannot
replace a value recorded at a later session step. The immutable audit survives replacement;
active values are reloaded after process restart. Rounds and session artifact directories
isolate tasks. Corrupt storage raises an error instead of silently forgetting notes.

The prompt contains exactly one active view, outside replay history and summary checkpoints.
It is pinned and counted by the normal context token estimator. Its default serialized
budget is 12,000 characters; newest entries are selected first and omissions are stated.
Omitted entries remain in the durable store. These are agent notes, not higher-priority
instructions or an automatic cross-task knowledge base.

Ordinary NexAU skills remain configurable in agent YAML. Recursive skill induction,
checkpoint search and learned policy training are outside this repository's release scope.
