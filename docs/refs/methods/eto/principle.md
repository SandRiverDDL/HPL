# ETO 原理

## 问题定义

只用成功专家轨迹做 imitation，agent 不知道自己容易在哪里失败，也无法利用失败经验改进策略。

## 核心流程

```text
expert trajectories
-> SFT base agent
-> agent 在环境中探索
-> 收集失败轨迹
-> 将成功轨迹和失败轨迹组成 preference pair
-> DPO 训练
-> 新 agent 继续探索
```

## 训练信号

ETO 使用 trajectory-level preference：

- chosen：成功或更高 reward 的轨迹。
- rejected：失败或更低 reward 的探索轨迹。

## 迭代优化

ETO 可以多轮执行：

```text
policy_i -> exploration_i -> preference_i -> DPO -> policy_{i+1}
```

每轮策略变强后，探索轨迹分布也会改变。

## 和 IPR / HPL 的区别

- ETO：整条轨迹级别，简单稳定。
- IPR：单步级别，细但贵。
- HPL：多粒度，核心是 group-level。

## 主要风险

- 如果失败轨迹太差，DPO 只学到明显偏好，收益有限。
- 如果失败轨迹太少，覆盖不到关键错误模式。
- 如果只做 trajectory-level，无法精确定位局部错误。
