# DEPO 实验与实现

## 实验设置

| 项 | 内容 |
|---|---|
| 任务 | WebShop, BabyAI, 以及 OOD 数学 benchmark |
| 目标 | 降低 token per step 和 steps per trajectory |
| 指标 | performance, token usage, step count |
| 训练 | efficiency preference optimization |

## 关键实验结果

DEPO 主要收益是效率，不只是成功率：

| 模型 | WebShop Succ. | Reward | T@All | S@All |
|---|---:|---:|---:|---:|
| Llama3.1-8B-BC | 0.47 | 0.79 | 840 | 6.38 |
| Llama3.1-8B-BC+KTO | 0.48 | 0.67 | 776 | 8.96 |
| Llama3.1-8B-BC+DEPO | 0.50 | 0.72 | 633 | 7.80 |
| Qwen2.5-7B-BC | 0.44 | 0.75 | 1014 | 6.34 |
| Qwen2.5-7B-BC+KTO | 0.54 | 0.76 | 886 | 8.18 |
| Qwen2.5-7B-BC+DEPO | 0.56 | 0.80 | 726 | 7.73 |

论文结论：WebShop 上 DEPO 对 Qwen2.5-7B 的主要收益是 token 降低，step 数相对 KTO 有小幅下降，但相对 BC 仍可能变多。

## 数据与 rollout 成本

DEPO 不必像 IPR 那样对每个 step 做 MC。它更依赖已有轨迹中的效率比较：

- 成功轨迹之间比较 token 数和 step 数。
- 成功与失败轨迹仍以成功率优先。
- 对同一 task 的多条候选轨迹最容易构造偏好。

论文数据生成并非无 rollout：

- 先用 MCTS 生成大量 ReAct-style trajectories。
- 用 reward 阈值区分 desirable / undesirable。
- WebShop desirable 要求 `reward = 1`，且论文还按步数过滤：`steps < 7` 进入 desirable，`steps >= 7` 进入 undesirable。
- 使用 GPT-4.1 mini 对 Thought 做 rephrasing，保持 Action 不变。

开源仓库提供了训练数据：

- `data/kto_data/webshop_kto.json`，约 29.9MB。
- `data/kto_data/babyai_kto.json`，约 7.0MB。
- 格式含 `messages`, `label`, `steps`, `avg_tokens_per_step`, `total_tokens`。
- 样例中 WebShop trajectory 已是 KTO 标注数据，字段 `label=true/false`，不是原始 rollout 脚本。

## 代码入口

官方代码和数据仓库：

- https://github.com/OpenCausaLab/DEPO

训练配置：

- `efficient_agent/qwen2_5_7b.yaml`
- `efficient_agent/llama3_1_8b.yaml`
- `stage: kto`
- `use_efficient_kto: true`
- Qwen2.5-7B 使用 `efficiency_lambda_steps: 2`、`efficiency_lambda_tokens: 2`
- Llama3.1-8B 使用 `efficiency_lambda_steps: 3`、`efficiency_lambda_tokens: 3`

重要：当前开源配置是 Qwen2.5-7B 和 Llama3.1-8B，不是 Qwen2.5-1.5B。

## 迁移到本仓库

适合先做成 HPL 的后处理或附加偏好规则：

1. 在 WebShop trajectory/group 输出中统计 assistant token 数。
2. 统计完成任务的 environment steps。
3. 对同 reward 或 reward 差距很小的样本，用效率作为 tie-breaker。
4. 生成额外 DPO 数据，或给 HPL curriculum 增加 efficiency difficulty。

## 最小复现建议

```text
收集同一 task 的多条成功轨迹
-> 计算 token_count 和 step_count
-> 构造 shorter chosen / longer rejected
-> 小规模 DPO
-> 检查成功率是否下降
```

## 待确认问题

- 论文中 efficiency preference 的具体打分公式。
- 和 HPL loss 组合时，是合并数据还是单独阶段训练更稳。
- 开源 `webshop_kto.json` 的原始 trajectory 是否由 DeepSeek-V3 MCTS 生成，仓库样本本身没有显式记录 generator 字段；论文写 MCTS 搜索过程使用 DeepSeek-V3。
