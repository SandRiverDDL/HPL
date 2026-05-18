# IPR 原理

## 问题定义

只用完整 expert trajectories 做 SFT，会学到“看起来像专家”的行为，但无法明确修正中间步骤错误。只看 outcome reward，也很难判断某一步 action 是否真的有助于最终成功。

## 核心流程

```text
expert trajectory
-> 沿专家轨迹走到第 t 步
-> agent 生成候选 action
-> 从该 action 后继续 rollout 到终局
-> 用 Monte Carlo 估计 step-level reward
-> 和 expert action 对比
-> 构造 chosen/rejected action pair
-> DPO 或类似偏好训练
```

## 训练信号

IPR 的训练信号是 step-level preference。它不是只比较最终轨迹，而是比较同一状态下不同 action 的长期价值。

## 关键假设

- 环境可以可靠 reset 到任务初始状态。
- 可以通过重放 expert prefix 回到第 t 步。
- 从第 t 步开始 rollout 的最终结果能代表该 step action 的质量。

## 和 HPL 的区别

- IPR：每个 step 都可能构造偏好，粒度最细。
- HPL：将多个 step 组成 group，学习子任务级偏好。

## 主要风险

- MC 次数不足时，step reward 估计噪声大。
- 成本随轨迹长度近似二次增长。
- 对 WebShop 这类短任务还能接受；对 ALFWorld 这类长任务会明显变贵。
