<div align="center">

<a href="docs/assets/brand/northcua-logo.svg#gh-light-mode-only">
  <img src="docs/assets/brand/northcua-logo.svg#gh-light-mode-only" width="420" height="112" alt="NorthCUA">
</a>
<a href="docs/assets/brand/northcua-logo-dark.svg#gh-dark-mode-only">
  <img src="docs/assets/brand/northcua-logo-dark.svg#gh-dark-mode-only" width="420" height="112" alt="NorthCUA">
</a>

**心中万象，指间成形。**

**让意图成为行动，跨越桌面与网页。**

开源计算机智能体 Harness：接入模型，运行任务，检查结果。

[**从零开始**](docs/first-run.md) · [**观看演示**](README.md#see-computer-use-in-action) · [**技术报告**](https://arxiv.org/pdf/2611.02102v1) · [**模型接入**](docs/models.md) · [**文档**](docs/README.md) · [**English**](README.md)

[![CI](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/beipo-ai/NorthCUA/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-0057B8?style=flat-square)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Built on NexAU](https://img.shields.io/badge/Built%20on-NexAU-0B8F87?style=flat-square)](https://github.com/hzhua/nexau/tree/28511d9f0362191a6c378fd441b2be55309a9e32)

[![Hugging Face: NorthCUA-Pro](https://img.shields.io/badge/Hugging%20Face-NorthCUA--Pro-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/china-beipo/NorthCUA-Pro)
[![arXiv: 2611.02102](https://img.shields.io/badge/arXiv-2611.02102-b31b1b?style=flat-square)](https://arxiv.org/pdf/2611.02102v1)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-From%20zero%20to%20tasks-0B8F87?style=flat-square)](skills/northcua/SKILL.md)
[![Contributions welcome](https://img.shields.io/badge/PRs-welcome-6554C0?style=flat-square)](CONTRIBUTING.md)

</div>

从可交互的三维心形到原生 Blender 动画，计算机操作不止于下一次点击。
**NorthCUA Harness** 把工具操作、视觉上下文、工作笔记、原环境评估和可恢复跑批连接起来，
让你能追踪从第一张观察截图到最终产物的整个过程。

本仓库提供最新 [NorthCUA Technical Report](https://arxiv.org/pdf/2611.02102v1) 中的 Desktop/Web Harness，
不绑定单个模型。接入支持图像和结构化工具调用的端点，或先用无密钥本地教程验证运行链路。
**从一台电脑开始，扩展到批量任务，留下可检查的证据。**

## 看看 Agent 能把工作推进到哪里

<table>
<tr>
<td width="50%" valign="top">

### Ruby Heart · 30 秒

**让一个想法，成为可交互的三维心形。**<br>
GeoGebra · 控件、构建、展示与保存检查。

https://github.com/user-attachments/assets/8afee957-87d7-43c9-8dff-21de7478dabf

[下载 MP4](https://github.com/beipo-ai/NorthCUA/raw/refs/heads/main/docs/assets/demos/ruby-heart-30s.mp4) · [结果画面](docs/assets/demos/ruby-heart.png)

</td>
<td width="50%" valign="top">

### Blender UFO · 30 秒

**从草图，到飞碟，再到原生动画。**<br>
Inkscape → Blender · 船体、虹膜机构、金属与玻璃。

https://github.com/user-attachments/assets/b959f684-a23a-4052-bfcb-45d8c0c8c546

[下载 MP4](https://github.com/beipo-ai/NorthCUA/raw/refs/heads/main/docs/assets/demos/blender-ufo-30s.mp4) · [结果画面](docs/assets/demos/blender-ufo.png)

</td>
</tr>
</table>

两支均为 **30 秒加速剪辑**，播放时长不代表任务执行速度。上方使用 GitHub 原生播放器，
MP4 也随仓库提供，克隆后可离线观看。演示来自更大的 NorthCUA 项目，
不代表本次发布版本重新运行的任务或模型基准。[演示说明与来源](docs/showcase.md)。

## 不止于点击：覆盖整个任务生命周期

[![NorthCUA：准备、持续执行、原环境评估、保存证据与释放资源](docs/assets/harness-lifecycle.png)](docs/system-tour.md)

*最新报告 Figure 4 的机制示意；实际执行验证单独记录。*

- **桌面与网页统一执行。** Desktop 使用截图与鼠标键盘；Web 使用截图、DOM/ref、表单与标签页。共享会话、步骤、工具契约和记录。
- **长任务持续推进。** 有界图像窗口、操作轨迹标注、任务保真压缩和持久化 note，让后续决策保留需要的信息。
- **检查实际结果。** 独立 evaluator 可以读取产物，并在清理前检查同一工作区；Agent 宣称完成与评估结论分别保存。
- **可恢复跑批。** Controller/device、隔离 attempt、持久化状态、重试、resume 与清理确认，支持从单任务扩展到批量执行。

[系统导览](docs/system-tour.md) · [论文与实现对应](docs/paper-map.md) · [架构](docs/architecture.md)

## 快速开始

需要 Git、[uv 0.13.6+](https://docs.astral.sh/uv/) 和 Python 3.12。以下命令可用于 Bash 和 PowerShell：

**依赖说明：** 当前公开 NexAU 要求 Transformers 4.x，部分已知安全公告按维护者决定暂时保留，
已在[有期限的风险记录](docs/security/dependencies.md)中说明。仅使用可信模型／tokenizer 配置，推理服务单独部署。

```sh
git clone https://github.com/beipo-ai/NorthCUA.git
cd NorthCUA
uv sync --locked --python 3.12 --extra local --extra device
uv run --no-sync playwright install chromium
uv run --no-sync northcua --help
```

Linux 安装浏览器系统依赖可用 `uv run --no-sync playwright install --with-deps chromium`。

### 不需要 API key，先得到一个可验证的结果

在**终端 1** 启动仓库内的教程服务，保持运行：

```sh
uv run --no-sync northcua demo serve --directory logs/tutorial
```

看到 `"ready": true` 后，在同一个仓库目录打开**终端 2**：

```sh
uv run --no-sync northcua config validate --app-config logs/tutorial/app.example.yaml
uv run --no-sync northcua local web run --app-config logs/tutorial/app.example.yaml --url http://127.0.0.1:8787/task/first-web --query 'Task ID: first-web. Activate the button and verify the receipt.' --session-id first-web --max-steps 6 --evaluate
```

这里使用**真实浏览器与本地确定性模型端点**。成功标准是退出码 0、评估结果 `success`，
以及 `logs/tutorial/events.jsonl` 中对应任务的操作与评估回执。
完整结果位于 `logs/tutorial/artifacts/first-web/result.json`，评估字段为 `evaluation_report.overall_verdict`。
这验证运行链路，不衡量模型的自主任务能力。下一步可运行[三个任务的小批次与恢复教程](docs/local-batch.md)。
完成后按 Ctrl+C 停止服务；再次运行请选择新的输出目录，教程会保留已有结果。

<details>
<summary><strong>使用自己的模型：Desktop 或 Web</strong></summary>

模型需支持图像与结构化工具调用。[模型指南](docs/models.md)覆盖 Chat、Responses、Anthropic 和 Gemini。
macOS/Linux 的 Bash、zsh 使用：

```sh
cp .env.example .env
# 编辑 .env：填写模型名与 API key，按需修改 base URL。
set -a
. ./.env
set +a
uv run --no-sync northcua config init --directory logs/my-config --model "$NORTHCUA_MODEL" --base-url "$NORTHCUA_BASE_URL"
```

Windows 使用 [PowerShell 配置步骤](docs/first-run.md#2a-use-your-model-edit-only-model-connection-settings)。
CLI 不会自动读取 `.env`；请勿提交真实密钥。

生成配置后，先校验：

```sh
uv run --no-sync northcua config validate --app-config logs/my-config/app.example.yaml
```

再选择一种任务。Web：

```sh
uv run --no-sync northcua local web run --app-config logs/my-config/app.example.yaml --url https://example.com --query 'Report the main heading.' --max-steps 8
```

Desktop 只有工具直接返回截图这一条观察路径。若模型输出 **0–1000 相对坐标**，
包括 NorthCUA 模型系列，请先在本次运行的 app YAML 中开启 `desktop.coordinate_policy.input_space: normalized_1000`；
默认值是 `image_pixels`。Web 点操作使用 CSS 像素，DOM/ref 是元素 ID。
[可复制的坐标配置](docs/configuration.md#desktop-coordinates)。

Desktop 需要交互桌面和相应权限；macOS 需要屏幕录制与辅助功能权限。见[安装指南](docs/getting-started.md#run-desktop)。

再运行 Desktop：

```sh
uv run --no-sync northcua local desktop run --app-config logs/my-config/app.example.yaml --query 'Open a calculator and compute 12 * 7.' --max-steps 15
```

</details>

**交给 Agent 操作？** 加载 [NorthCUA 使用 skill](skills/northcua/SKILL.md)，
覆盖从零安装、模型配置、任务验证、跑批与恢复。[完整上手教程](docs/first-run.md)。

## 从单任务到完整工作流

| 目标 | 文档 |
|---|---|
| 从零到可检查的结果 | [First run](docs/first-run.md) |
| 三个本地任务、恢复与清理 | [Local batch](docs/local-batch.md) |
| Python、CLI、WebSocket、Controller/device | [运行方式](docs/running.md) |
| Context、图像窗口与工作记忆 | [上下文](docs/context.md) |
| 输出检查、在线评估与轨迹查看器 | [评估](docs/evaluation.md) |
| Desktop 推荐参数与重试策略 | [模型、上下文与响应策略](docs/desktop-settings.md) |
| 开发、测试与扩展 | [贡献](CONTRIBUTING.md) · [测试](docs/testing.md) |

## 研究与发布状态

**[NorthCUA 技术报告已发布于 arXiv](https://arxiv.org/pdf/2611.02102v1)**，介绍完整 NorthCUA 系统和模型结果。
仓库同时附带 [2026-10-31、123 页的报告快照](docs/paper/main.pdf)，便于离线阅读。
本仓库发布 Desktop/Web Harness。Experience Flywheel、数据生成、分阶段训练与模型权重是独立研究组成，
不是运行本仓库所需的隐藏依赖。

NorthCUA 模型系列在 Hugging Face 单独发布：

| 模型 | 总参数 / 激活参数 | 架构 | 权重与模型卡 |
|---|---|---|---|
| **NorthCUA-Nano** | 9B / 9B | Dense | [Hugging Face](https://huggingface.co/china-beipo/NorthCUA-Nano) |
| **NorthCUA-Mini** | 35B / 3B | MoE | [Hugging Face](https://huggingface.co/china-beipo/NorthCUA-Mini) |
| **NorthCUA-Pro** | 397B / 17B | MoE | [Hugging Face](https://huggingface.co/china-beipo/NorthCUA-Pro) |

*参数规模来自模型卡与技术报告；激活参数反映每 token 的计算量，不等于加载模型所需的显存。*

[**NorthCUA-Pro 权重与模型卡**](https://huggingface.co/china-beipo/NorthCUA-Pro)
已在 **2026-11-03** 确认为公开可访问，模型卡标明 Apache-2.0 许可证；报告中的原模型链接会解析到这里。
模型 checkpoint 与 Harness 分别发布，本仓库集成测试不代表该 checkpoint 的推理或基准验收。
[模型指南](docs/models.md)记录具体 revision、端点要求与坐标配置；报告与演示均随仓库提供。

[发布验证记录](docs/release-0.2.0.md)区分从零安装、回归、真实浏览器、本地跑批、安装包和独立 review。
脚本模型集成测试不等于真实模型能力或完整 OSWorld／云端基准验收。
源代码安装不依赖已经存在的 PyPI 发行版。

## 引用

如本项目对你的研究有帮助，请引用技术报告，并记录代码/NexAU revision、模型设置、任务来源与评估协议。

```bibtex
@article{zhang2026northcua,
  title={NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale},
  author={Zhang, Chaoyun and Li, Bingxuan and Guo, Pinxue and Gao, Yangfan and Cai, Zikun and Xie, Tianci and Li, Hui and Huang, Yuqi and Cao, Yifei and Mao, Bo and Tang, Tian and Zheng, Qichang and Liu, Tongtong and Wang, Xiaoxing and Liu, Jinxiu and Lin, Lizhi and Shao, Zhou and Zhou, Maosen and Zheng, Miao and Zhang, Shuo and He, Yangji and Lei, Kepeng and Zhao, Ning and Shen, Wei and Jin, Zhenjiang and Zhang, Xinyue and Sun, Peng and Han, Zhenhua and Zheng, Rui and Yan, Hang},
  journal={arXiv preprint arXiv:2611.02102},
  year={2026},
  url={https://arxiv.org/pdf/2611.02102v1}
}
```

[BibTeX 文件](CITATION.bib) · [引用元数据](CITATION.cff)

[Apache-2.0](LICENSE) · [第三方声明](THIRD_PARTY_NOTICES.md) · [安全](SECURITY.md) · [社区](CODE_OF_CONDUCT.md) · [更新](CHANGELOG.md)

## 联系我们

项目使用、研究交流与合作，请联系：

- [zhangchaoyun@qijizhifeng.com](mailto:zhangchaoyun@qijizhifeng.com)
- [libingxuan@qijizhifeng.com](mailto:libingxuan@qijizhifeng.com)
