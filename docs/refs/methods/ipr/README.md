# IPR

## 一句话

IPR（Iterative Step-Level Process Refinement）沿专家轨迹逐步探索候选动作，用 Monte Carlo 估计 step-level reward，再构造 contrastive action pairs 训练 agent。

## 适合解决的问题

- 需要明确知道“哪一步错了”的 agent 任务。
- 专家轨迹可靠，环境可以从中间状态继续 rollout。
- 有足够预算做 step-level MC。

## 不适合的问题

- 单卡预算紧张，无法承担大量 MC rollout。
- 单步 reward 噪声很大，环境反馈不稳定。
- 错误来自多步组合而不是单个 action。

## 关键结论

- 你记的是 IPR，不是 IRP。
- IPR 的核心成本来自每个专家 step 都要探索并用 MC 估计长期回报。
- 它是理解 HPL 的重要 baseline：HPL 的 group-level 可以看作对 IPR 过细粒度的改进。

## 对 HPL 的启发

- IPR 提供了 step-level reward 的构造方式。
- HPL 可以保留 IPR 的局部 credit assignment 思路，但减少逐 step 评估。
- 对 WebShop 单卡复现，IPR 更适合作为小样本诊断工具，不适合作为默认全量路径。

## 资料来源

- paper: https://arxiv.org/abs/2406.11176
- code: https://github.com/WeiminXiong/IPR
