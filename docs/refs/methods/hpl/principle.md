# HPL 原理

## 问题定义

long-horizon agent 的偏好学习存在粒度不匹配：

- trajectory-level：终局奖励稳定，但不知道是哪一步贡献或出错。
- step-level：监督更细，但 MC rollout 成本高，奖励噪声大。
- group-level：把连续动作看成子任务，试图保留局部性和多步结构。

## 核心流程

```text
expert trajectories
-> trajectory-level preference
-> step-level preference
-> semantic action grouping
-> negative group generation
-> group-level MC reward
-> curriculum DPO training
```

## 训练信号

- trajectory preference：比较整条轨迹的最终结果。
- step preference：比较某一步 agent action 和 expert action 的长期回报。
- group preference：比较一段连续 expert actions 和生成的 negative group。

## group-level 机制

HPL 先把专家轨迹切成语义连贯的 action groups。每个 group 可以理解成一个子任务，例如 WebShop 中的搜索、筛选、选择属性、购买。

然后从 group 起点开始生成一个 suboptimal group，用 rollout 或最终奖励估计该 group 的质量，构造 chosen/rejected。

## curriculum

curriculum 有两个维度：

- group length：越长表示子任务越复杂。
- sample difficulty：由 preferred 和 dispreferred group 的 reward gap 表示。

直觉是先学短、明显的偏好，再学长、难分的偏好。

## 和 IPR 的区别

- IPR 关注单步 action 的 process refinement。
- HPL 认为单步太碎，容易忽略多步行为组合。
- HPL 的 group-level 是 step 和 trajectory 之间的折中。

## 主要风险

- group 切分质量会显著影响数据质量。
- 负 group 生成如果太弱，偏好样本过于简单。
- 如果仍全量依赖 step MC，成本可能接近 IPR。
