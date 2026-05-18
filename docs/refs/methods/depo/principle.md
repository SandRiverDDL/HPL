# DEPO 原理

## 问题定义

agent 能完成任务不代表高效。长 CoT 和多余步骤会增加 token 成本、推理延迟和环境交互成本。

DEPO 把效率分成两个层面：

- step-level efficiency：每一步输出更短、更直接。
- trajectory-level efficiency：用更少环境步骤完成任务。

## 核心流程

```text
agent trajectories
-> 评价任务成功与效率
-> 构造偏好对
-> 偏好优化
-> 更短输出、更少步骤的 agent
```

## 训练信号

偏好不只看最终 reward，也看效率：

- 同样完成任务时，短 token 优于长 token。
- 同样完成任务时，少步骤优于多步骤。
- 如果效率和成功冲突，不能简单牺牲成功率。

## 和 HPL 的区别

- HPL 主要解决 credit assignment 粒度问题。
- DEPO 主要解决 agent reasoning/acting 过长的问题。
- 两者可以组合：HPL 负责学会做，DEPO 负责做得更省。

## 主要风险

- 过度奖励短输出，可能导致 Thought 不充分。
- 对 WebShop 来说，少 step 不一定总是好；跳过必要筛选可能买错商品。
- 需要设计成功率优先的效率偏好规则。
