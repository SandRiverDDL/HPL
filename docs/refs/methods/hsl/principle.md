# HSL 原理

## 核心问题

长程 agent rollout 经常没有完成原始任务，但这不代表整条轨迹没有价值。传统 RFT/SFT 通常只保留原始任务成功轨迹，失败轨迹会被丢掉；HSL 的想法是把这些轨迹放到 hindsight 视角下重新解释。

## 核心流程

1. 用当前 agent 在环境中生成完整 trajectory。
2. 用强 relabeler 检查 trajectory 中实际完成的目标。
3. 把原始 instruction 替换或扩展为 hindsight goal。
4. 对无关 action 做 masking，降低低质量动作污染。
5. 对 relabeled demonstrations 做加权 SFT。
6. 可选：先 HSL，再继续 DPO，形成 `DPO+HSL`。

## 训练信号

HSL 本质仍是 supervised learning：

- 输入：重新标注后的自然语言目标和轨迹上下文。
- 输出：完成该 hindsight goal 的 action / response。
- 损失：加权 SFT loss。

论文还使用：

- irrelevant-action masking：屏蔽和 hindsight goal 无关的动作。
- sample reweighting：提高更有价值 relabeled demonstration 的权重。

## 和 EEF 的区别

- EEF：失败 expert trajectory 中找 beneficial actions 或 recovery segment。
- HSL：不要求原任务最终可恢复成功，而是把 trajectory 中实际达成的目标重新标注成成功任务。

直观上，EEF 是“失败中找局部正确动作”，HSL 是“给偏离目标的成功行为换一个正确标签”。

## 和 HPL 的区别

- HPL：构造 trajectory / step / group 偏好，依赖 reward 或 MC 估计。
- HSL：构造新的 SFT 正样本，依赖强 relabeler 判断 achieved goals。

HSL 不解决细粒度 credit assignment，但能提高 rollout 数据利用率。
