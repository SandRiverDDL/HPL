# ETO 实验与实现

## 实验设置

| 项 | 内容 |
|---|---|
| 任务 | WebShop, ScienceWorld, ALFWorld |
| 数据 | expert trajectories + exploration failures |
| 训练 | SFT -> DPO -> iterative exploration |
| 代码框架 | 官方仓库使用 FastChat 改造训练和部署 |

## 数据与 rollout 成本

ETO 成本主要来自完整 episode 探索，而不是每个 step 做 MC。

相比 IPR：

- 单条失败轨迹只需跑一次完整 episode。
- 不需要对每个 expert prefix 反复 rollout。
- 但需要足够多探索轨迹覆盖错误模式。

## 代码入口

官方 ETO 仓库：

- `run_eto.sh`
- `construct_preference.py`
- `eval_agent/`
- `fastchat/`

## 迁移到本仓库

可以作为 HPL 的 negative data 来源：

1. 用当前 base agent 在 WebShop 上探索失败轨迹。
2. 把失败轨迹和 expert trajectory 对齐。
3. 从失败轨迹中抽取 group-level negative segments。
4. 接入 HPL 的 group preference 构造。

## 最小复现建议

```text
用 WebShop 20-50 个 task
-> base agent 生成完整探索轨迹
-> 根据 reward 区分 success/failure
-> 构造 trajectory-level DPO 数据
-> 人工检查失败是否有学习价值
```

## 待确认问题

- ETO 官方 expert trajectories 和数据是否仍可直接下载。
- ETO 的 failure-success pairing 是否要求同一 task 内配对。
- 和 HPL group negative 结合时，如何对齐动作边界。
