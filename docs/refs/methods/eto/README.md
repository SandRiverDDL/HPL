# ETO

## 一句话

ETO（Exploration-Based Trajectory Optimization）让 agent 通过试错收集失败轨迹，再把失败轨迹和成功轨迹组成 trajectory-level preference pair，用 DPO 等方法迭代优化策略。

## 适合解决的问题

- 有环境可交互，agent 能自己探索。
- 成功/失败轨迹可由最终 reward 判断。
- 希望减少对纯 expert imitation 的依赖。

## 不适合的问题

- 环境交互极贵，无法收集足够失败轨迹。
- 大多数失败轨迹质量太差，和成功轨迹差距过大。
- 需要非常精细的 step-level credit assignment。

## 关键结论

- ETO 的重点不是 step-level reward，而是 failure-success trajectory pair。
- 它比 IPR 的 step MC 更粗，但数据来源更自然。
- 对 HPL 的价值在于提供 negative trajectory / failure data 的来源。

## 对 HPL 的启发

- HPL 的 negative group 目前主要从 expert group 起点生成，可借鉴 ETO 做更自然的 failure collection。
- 可以先用 ETO 式完整失败轨迹，再切分出失败 group。
- 对没有完整 expert group 标注的任务，ETO 更容易启动。

## 资料来源

- paper: https://arxiv.org/abs/2403.02502
- code: https://github.com/Yifan-Song793/ETO
