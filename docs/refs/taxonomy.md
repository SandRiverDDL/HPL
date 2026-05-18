# Agent 后训练方法分类

## 按偏好粒度

| 粒度 | 代表方法 | 优点 | 主要问题 |
|---|---|---|---|
| trajectory-level | ETO, trajectory DPO | 信号稳定，数据构造简单 | credit assignment 粗，难定位哪一步错 |
| step-level | IPR | 信号细，能修正局部错误 | MC rollout 成本高，单步奖励噪声大 |
| group-level | HPL | 介于 step 和 trajectory 之间，适合多步子任务 | 需要分组策略和负 group 构造 |
| efficiency-level | DEPO | 直接优化 token / step 成本 | 需要避免过度压缩导致性能下降 |
| failure-segment reuse | EEF | 利用失败专家轨迹中的有益片段 | 需要能从中间 expert state 继续模拟 |
| hindsight relabeling | HSL | 把偏离原任务但实际完成的目标转成成功演示 | 需要强 relabeler 判断 achieved goals |

## 按数据来源

| 类型 | 方法 | 说明 |
|---|---|---|
| expert trajectory | HPL, IPR | 沿专家轨迹构造局部或分组偏好 |
| self-exploration | ETO | 让当前 agent 探索失败轨迹，再和成功轨迹形成偏好对 |
| efficiency annotation | DEPO | 从响应简洁性、完成步数等维度构造效率偏好 |
| failed expert trajectories | EEF | 不丢弃失败专家轨迹，筛出其中有价值的动作或 recovery segment |
| achieved goals in rollout | HSL | 用强模型把 rollout 重新标注为实际完成的自然语言目标 |

## 按训练目标

| 目标 | 代表方法 | 适合改进点 |
|---|---|---|
| 提高任务成功率 | HPL, IPR, ETO | WebShop / ALFWorld / InterCode-SQL 任务完成率 |
| 降低交互步数 | DEPO, HPL curriculum | WebShop 搜索与点击路径更短 |
| 降低 token 成本 | DEPO | 减少冗长 Thought 和无效解释 |
| 提高长程 credit assignment | HPL, IPR | 从终局奖励反推局部动作质量 |
| 提高探索覆盖 | EEF, RFT, ETO | 避免训练数据只覆盖容易成功的子任务 |
| 提高数据利用率 | HSL, EEF | 把原本会被丢掉的失败或偏离目标轨迹转成训练信号 |

## 当前优先级

1. **HPL**：本仓库目标方法，必须理解数据链路和代码入口。
2. **IPR**：HPL 的重要对比对象，理解 step-level MC 为什么贵。
3. **ETO**：理解 trajectory-level 自探索和 failure-success pairing。
4. **DEPO**：可作为 HPL 改进方向，引入 token / step efficiency。
5. **EEF**：可作为性价比路线，重点看是否能复用失败 expert trajectories。
6. **HSL**：ICLR 2026 accepted，作为比 EEF 更可靠的 hindsight relabeling 路线优先跟进。
