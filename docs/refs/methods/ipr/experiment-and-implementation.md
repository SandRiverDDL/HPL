# IPR 实验与实现

## 实验设置

| 项 | 内容 |
|---|---|
| 任务 | WebShop, ALFWorld, InterCode-SQL |
| 数据 | expert trajectories + agent exploration |
| 核心评估 | 成功率、reward、action efficiency |
| 训练 | SFT 后迭代构造 step-level preference，再偏好优化 |

## 数据与 rollout 成本

WebShop 若平均轨迹长度为 `L`，每条轨迹逐 step MC 的成本近似：

```text
K * (L + L-1 + ... + 1) = K * L(L+1)/2
```

其中 `K` 是每个 step 的 MC 采样次数。`L=8-10`、`K=4-5` 时，每条 expert trajectory 可能带来几十到数百次 agent action 调用。

## 代码入口

官方 IPR 仓库入口：

- `run_pipeline.sh`
- `monte_carlo_sample_webshop.py`
- `construct_preference_monte_carlo_webshop.py`

本 HPL 仓库中对应入口：

- `pipeline/step/run_mc_sample.sh`
- `pipeline/step/monte_carlo_sample.py`
- `pipeline/step/construct_preference_step.py`

## 迁移到本仓库

- 不建议先全量跑 IPR。
- 可以用 20 个 WebShop task 做 step-level reward 诊断。
- 用 IPR 小样本结果判断 HPL group 切分是否覆盖关键错误步骤。

## 最小复现建议

```text
选 20 个 WebShop task
-> 生成 trajectory
-> sample_num 降到 2
-> 跑 step MC
-> 构造 step preference
-> 人工检查 chosen/rejected 是否合理
```

## 待确认问题

- 官方 IPR expert trajectories 是否仍可下载。
- 本 HPL 仓库的 `construct_preference_step.py` 路径中部分默认目录仍写 `7b`，需要运行前核对。
