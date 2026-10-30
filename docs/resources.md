# Project resources and publication status

Checked **2026-10-31**. The three canonical project addresses below are written
on the latest report's first page. We retain those identities and distinguish
an intended address from a publicly accessible resource.

| Resource | Canonical address | Observed release status / usable fallback |
|---|---|---|
| Harness source | [NorthCUA-Harness](https://github.com/china-qijizhifeng/NorthCUA-Harness) | Fresh, self-contained source repository. Its visibility is controlled by the maintainer. |
| Project website | [beipo.top/research/northcua](https://beipo.top/research/northcua/) | Returned HTTP 404 at the check. Use the [bundled showcase](showcase.md), [documentation index](README.md) and local report meanwhile. |
| NorthCUA-Pro | [Hugging Face model page](https://huggingface.co/beipo-ai/NorthCUA-Pro) | Public page/API returned HTTP 401. Public downloads, checkpoint revision and weight license are not confirmed here; see [model setup](models.md). |
| Technical Report | [123-page local PDF](paper/main.pdf) | Included in this repository; title, date and SHA-256 are recorded in the [paper manifest](paper/manifest.json). No account or external paper host is needed to read the local file. |
| Interactive website source | [NorthCUA-site](https://github.com/china-qijizhifeng/NorthCUA-site) | Separate website project with its media and build. Public hosting is not assumed by the Harness installation. |
| Operator skill | [NorthCUA skill](../skills/northcua/SKILL.md) | Bundled: from-zero install, configuration, local tasks, batches, results and recovery. |

No arXiv/DOI identifier, public weight license or PyPI package publication is
invented by this source release. Update this page when publication is confirmed.
The model's eventual license and hosting terms are independent from the
Harness's Apache-2.0 code license.

## Keep an experiment reproducible

Record the Harness and NexAU commits, model/checkpoint revision, provider/API
settings, task and environment version, coordinate profile, budgets, evaluator
contract and artifact manifest. The report's model results and the repository's
scripted-model runtime checks answer different questions; cite both accurately.
