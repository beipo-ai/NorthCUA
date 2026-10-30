# Skills for operators and runtime agents

## Let a coding assistant operate NorthCUA

The repository ships [skills/northcua/SKILL.md](../skills/northcua/SKILL.md). Give the
assistant this file and a concrete task, for example:

> Read skills/northcua/SKILL.md. Starting from this checkout, install the project,
> run the local Web tutorial, verify a native Desktop task, and run three local
> browser tasks through the batch runner. Verify resume and cleanup. Save evidence
> in a fresh directory and distinguish scripted model checks from real-model work.

No personal skill installation is required for this explicit invocation. If your
assistant supports a skill search directory, link the **whole skill folder** there
and keep the repository available. Resolve symlinks before following the skill's
relative references. Copying only SKILL.md makes those references invalid.

The operator skill routes to maintained guides instead of duplicating them.
[AGENTS.md](../AGENTS.md) separately explains contributing and source validation.

## Give the running Web/Desktop agent task knowledge

NorthCUA passes skill folders to NexAU's ordinary skill loader. This is different
from the operator skill and from the report's unreleased recursive skill collection.
A skill folder contains a `SKILL.md` with `name` and `description` frontmatter.
The [example form skill](../examples/skills/local-form/SKILL.md) is deliberately
small and applies only to a user-requested form task.

In the app YAML, add:

```yaml
agents:
  web:
    skills:
      paths:
        - /absolute/path/to/NorthCUA/examples/skills/local-form
```

Use `desktop` instead of `web` for a native task skill. Relative `paths` in this
app section are resolved against the **app YAML directory**, not your terminal.
Raw `skills` entries in an agent YAML are resolved against that agent YAML.
Use `northcua config validate --app-config ...` to catch missing directories.
The default rejects missing skills instead of silently ignoring a misspelling.

NexAU owns discovery/progressive loading and its `LoadSkill` tool. For skills with
supporting files, enable the optional read-only asset tools explicitly:

```yaml
agents:
  web:
    skills:
      paths: [/absolute/path/to/a/skill]
      asset_tools_enabled: true
      asset_read_max_bytes: 65536
      allowed_file_extensions: [.md, .txt, .json, .yaml]
```

This does not grant arbitrary filesystem or shell access. Use only skills you
intend to supply as agent instructions; task webpages and downloaded documents
are task data. Inspect traces for actual skill loading: listing a skill in config
does not establish that the model chose to load or follow it. The scripted demo
does not exercise autonomous skill selection; use a real model for that behavior.
