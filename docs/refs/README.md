# Agent 后训练方法调研索引

本目录用于保存 agent 后训练方法的调研笔记，目标不是写完整论文综述，而是服务本仓库的 HPL 复现、理解和改进。

## 保存原则

- 每篇论文或方法一个目录，放在 `docs/refs/methods/<method>/`。
- 默认只建 3 个文件：
  - `README.md`：一页摘要，回答这篇方法值不值得继续看。
  - `principle.md`：原理、训练信号、目标函数和算法流程。
  - `experiment-and-implementation.md`：实验设置、rollout 成本、代码入口和迁移到本仓库的修改点。
- 如果某个文件超过约 300-500 行，或某个主题反复增长，再拆分：
  - `rollout-cost.md`
  - `ideas.md`
  - `code-map.md`
- 实验设置必须有独立小节，避免和原理混在一起。
- rollout / 数据构造成本必须显式记录，因为 agent 后训练的主要成本通常在这里。
- 每篇方法都要有“对 HPL 的启发”或“可迁移改法”，否则不纳入本目录。

## 阅读路径

- 快速回忆：只读对应方法的 `README.md`。
- 准备深入理解：读 `principle.md`。
- 准备复现或改代码：读 `experiment-and-implementation.md`。
- 准备做数据转换或跨方法迁移：读 `data-formats/`。

## 当前方法

| 方法 | 全名 | 主要粒度 | 状态 |
|---|---|---|---|
| [HPL](methods/hpl/README.md) | Hierarchical Preference Learning | trajectory + step + group | 已建初版 |
| [IPR](methods/ipr/README.md) | Iterative Step-Level Process Refinement | step-level | 已建初版 |
| [ETO](methods/eto/README.md) | Exploration-Based Trajectory Optimization | trajectory-level | 已建初版 |
| [DEPO](methods/depo/README.md) | Dual-Efficiency Preference Optimization | efficiency preference | 已建初版 |
| [EEF](methods/eef/README.md) | Exploring Expert Failures | expert-failure reuse | 已建初版 |
| [HSL](methods/hsl/README.md) | Hindsight Supervised Learning | hindsight relabeling | 已建初版 |

## 当前数据格式记录

| 主题 | 说明 | 状态 |
|---|---|---|
| [WebShop GiGPO/SPEAR SFT 格式](data-formats/webshop-gigpo-spear-sft.md) | ETO ShareGPT 轨迹如何转成 WebShop agentic RL 的 cold-start SFT 数据 | 已建初版 |

## 资料来源

- HPL: https://arxiv.org/abs/2510.03253
- HPL code: https://github.com/PatrickG1014/HPL
- IPR: https://arxiv.org/abs/2406.11176
- IPR code: https://github.com/WeiminXiong/IPR
- ETO: https://arxiv.org/abs/2403.02502
- ETO code: https://github.com/Yifan-Song793/ETO
- DEPO: https://arxiv.org/abs/2511.15392
- DEPO project: https://opencausalab.github.io/DEPO/
- DEPO code/data: https://github.com/OpenCausaLab/DEPO
- EEF: https://arxiv.org/abs/2504.13145
- HSL: https://openreview.net/forum?id=QNfmqMSR7r
- HSL ICLR poster: https://iclr.cc/virtual/2026/poster/10009594
- GiGPO code: https://github.com/langfengQ/verl-agent
- SPEAR code: https://github.com/TencentYoutuResearch/SPEAR
