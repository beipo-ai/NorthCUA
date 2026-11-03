# Project resources and publication status

Checked **2026-11-03**. This page covers the Harness source, report, model resource
and operator skill. For external resources, an intended address is distinguished
from a publicly accessible resource.

| Resource | Canonical address | Observed release status / usable fallback |
|---|---|---|
| Harness source | [NorthCUA](https://github.com/beipo-ai/NorthCUA) | Fresh, self-contained source repository. Its visibility is controlled by the maintainer. |
| NorthCUA-Pro | [Hugging Face model page](https://huggingface.co/china-beipo/NorthCUA-Pro) | Public, ungated checkpoint. Card/config/license and all 122 shard URLs checked without authentication; no weights downloaded or inference accepted. [Pinned revision and setup](models.md). |
| Technical Report | [arXiv:2611.02102v1](https://arxiv.org/pdf/2611.02102v1) · [Abstract and metadata](https://arxiv.org/abs/2611.02102v1) | Public arXiv report. An unchanged [31 October, 123-page snapshot](paper/main.pdf) remains bundled for offline reading; its title, date and SHA-256 are recorded in the [paper manifest](paper/manifest.json). |
| Operator skill | [NorthCUA skill](../skills/northcua/SKILL.md) | Bundled: from-zero install, configuration, local tasks, batches, results and recovery. |

The report's original `beipo-ai/NorthCUA-Pro` address resolves to the model resource
above. Its published Apache-2.0 license was checked at revision
`2f1f0cbf4ed3f1243c37838889c446f50bc3d32c`; the 31 October HTTP 401 observation
is superseded. Model serving and experiment acceptance are separate checks.

The report is identified as **arXiv:2611.02102**; [BibTeX](../CITATION.bib) and
[GitHub citation metadata](../CITATION.cff) use its published author list.
PyPI publication is not assumed. The model's license,
notices and hosting terms apply to its own distribution independently from
the Harness's code license.

## Keep an experiment reproducible

Record the Harness and NexAU commits, model/checkpoint revision, provider/API
settings, task and environment version, coordinate profile, budgets, evaluator
contract and artifact manifest. The report's model results and the repository's
scripted-model runtime checks answer different questions; cite both accurately.

## Contact

For project questions, research discussions or collaboration:

- [zhangchaoyun@qijizhifeng.com](mailto:zhangchaoyun@qijizhifeng.com)
- [libingxuan@qijizhifeng.com](mailto:libingxuan@qijizhifeng.com)
