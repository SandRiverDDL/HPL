# EEF 原理

## 问题定义

RFT 只保留成功轨迹，容易产生 simplicity bias：模型不断学习容易成功的子任务，困难子任务长期 OOD。失败专家轨迹虽然最终 reward 为 0，但其中可能包含正确搜索、翻页、回退、恢复等有价值动作。

## 核心流程

```text
expert trajectories
-> 保留成功 expert trajectories 做初始 SFT
-> 当前 policy 从初始状态探索
-> 当前 policy 从若干 expert intermediate states 探索
-> 把成功探索轨迹加入 positive repository
-> 找出需要 recovery 的 expert states
-> 从 positive repository 中抽取对应 solution
-> 用 SFT loss 继续训练
```

## 关键机制

给定一条 expert trajectory，EEF 不模拟所有中间状态，而是选 `M` 个等间隔状态：

```text
l = floor(|trajectory| / (M + 1))
states = [s_l, s_2l, ..., s_Ml]
```

如果 policy 从前一个 expert state 成功、从后一个 expert state 失败，则认为中间 expert actions 可能有害，并把后一个 state 标记为需要 recovery。

## 和 RFT 的区别

- RFT：只从初始状态探索，成功就加入训练。
- EEF：额外从失败 expert trajectory 的中间状态探索，寻找困难子任务和 recovery 技能。

## 和 HPL 的区别

- HPL：构造偏好对，训练 DPO/KTO 类目标。
- EEF：筛选正向片段，仍用 SFT loss。
- HPL 关注粒度不匹配；EEF 关注失败专家数据中未利用的有益动作。

## 主要风险

- 需要能把环境恢复到 expert intermediate state。
- 如果状态恢复不可靠，筛选出来的片段可能不可信。
- 如果失败 expert 质量太差，筛选收益有限。
