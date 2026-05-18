# HPL 实验与实现

## 实验设置

| 项 | 内容 |
|---|---|
| 任务 | WebShop, ALFWorld, InterCode-SQL |
| 训练框架 | 本仓库使用 LLaMA-Factory |
| 默认 SFT 模型 | README 中为 Qwen2.5-7B-Instruct |
| rollout agent | pipeline 脚本中常见为 `models/<task>-1.5b-sft` |
| 指标 | 任务成功率、reward、可能包含效率分析 |
| baseline | SFT, trajectory-level, step-level/IPR, ETO/MPO 等 |

## 关键实验结果

论文 WebShop 主结果显示，HPL 相对 SFT 的提升不算大：

| 模型 | SFT success | HPL Fixed-N(3) success | 提升 |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 26.00 | 30.17 | +4.17 |
| Qwen2.5-7B | 31.83 | 34.33 | +2.50 |

资源表在 ALFWorld/Qwen2.5-1.5B 上报告：

| 方法 | LLM calls | 生成时间 | 训练时间 |
|---|---:|---:|---:|
| ETO | 约 30k | 1h07min | 13min |
| IPR step-level part | 约 750k | 6h13min | 26min |
| HPL Fixed-N(3) group-level part | 约 207k | 3h35min | 25min |
| HPL Semantic group-level part | 约 221k | 3h21min | 26min |

注意：该资源表强调的是 HPL 的 `group-level part`，不是从零构造所有 trajectory/step/group 数据的总成本。

## 数据与 rollout 成本

WebShop 的 `data/webshop/train_indices.json` 有 1624 个任务编号，但这不是 SFT 数据本体。

本地状态：

- WebShop 环境数据在 `envs/webshop/data`。
- WebShop 索引在 `envs/webshop/search_index`。
- 论文 expert trajectories 当前未下载。
- `smoke_data/` 只有最小训练 smoke。

成本判断：

- 只跑 trajectory 级别：几万次 agent action 调用量级。
- 跑 IPR/step-level MC：容易到几十万次调用。
- 跑 HPL group-level：应低于逐 step MC，但仍需要负 group 生成和 group MC。
- 如果本地完整跑 `pipeline/step` + `pipeline/group`，总成本未必比 IPR 更低。

## 代码入口

- trajectory: `pipeline/trajectory/`
- step preference: `pipeline/step/`
- group preference: `pipeline/group/`
- WebShop task config: `configs/task/webshop.json`
- model API config: `configs/model/agent.json`
- SFT config: `train/sft/*.yaml`
- smoke config: `train/sft_smoke/*.yaml`

## 迁移到本仓库

优先顺序：

1. 不全量跑 step-level MC，先验证 WebShop group-level 小样本。
2. 明确使用本地 OpenAI-compatible API server，例如 vLLM 或 LLaMA-Factory API。
3. 先用 20-50 个 WebShop task，降低 `sample_num` 和 `neg_group_num`。
4. 记录每段 jsonl 输出数量、平均轨迹长度、平均耗时。

## 最小复现建议

```text
WebShop reset/step smoke
-> base agent API smoke
-> 生成少量 expert/action group 输入
-> run_negative.sh 小样本
-> run_negative_mc.sh 小样本
-> construct_preference_group.py
-> 检查 DPO 数据格式
```

## 待确认问题

- 论文 expert trajectories 的公开下载入口和格式。
- HPL 训练时各粒度 loss 的权重配置位置。
- group semantic 切分是否必须调用强模型，还是可用规则/小模型替代。
- HPL 报告的 WebShop 小幅提升是否值得本地全量复现，需先用 200-500 条做数据质量验证。
