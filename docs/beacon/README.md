# BEACON 集成文档

本目录记录 HPL 侧如何控制相邻 BEACON 仓库进行 WebShop agentic RL。BEACON 是外部执行引擎，HPL 当前只保留轻量编排脚本、测试 split 约束、本地 patch 说明和运行记录。

## 文档分工

- `environment.md`：BEACON 仓库、`.venv`、依赖和当前环境状态。
- `patches.md`：对 BEACON 源码做过的本地 patch、原因、影响和回滚口径。
- `webshop-split.md`：WebShop train/val/test 任务 id 采样、防泄漏规则和验证口径。
- `rl-runbook.md`：HPL 控制 BEACON 训练的启动脚本、默认超参、训练量估计和待验证项。

## 当前入口

HPL 编排脚本：

```bash
scripts/beacon/run_webshop_lora_rl.sh
```

BEACON 训练入口：

```bash
cd /mnt/dataset/fengshuwen/agent-post-train/BEACON
.venv/bin/python -m verl.trainer.main_ppo ...
```

## 当前状态摘要

- BEACON 提交 `7146bfa` 已支持已有 LoRA adapter 初始化和 WebShop split 过滤。
- HPL 提交 `994daf7` 已同步 fork 与 BEACON 环境状态；当前工作区在此基础上调整 RL launcher 默认并发。
- BEACON `.venv` 已补 `gym==0.24.0` 与 WebShop runtime 依赖，WebShop import smoke 可通过。
- 当前 WebShop RL 主线已切回 BEACON/SPEAR 默认 small/synthetic 口径；正式 50 step 短跑使用 tmux `hpl_beacon_webshop_rl_n4_50step`，W&B run 为 `hpl_lora_migpo_qwen2_5_1p5b_webshop_small_n4_50step`。
- 该 run 使用 `TRAIN_BATCH_SIZE=3`、`ROLLOUT_N=4`、`PPO_MINI_BATCH_SIZE=12`、`TEST_FREQ=0`、`SAVE_FREQ=25`；每个 `global_step` 对应 3 个 prompt、12 条轨迹和 1 个 PPO mini-batch。
- 旧 run `hpl_lora_migpo_qwen2_5_1p5b_webshop_small_50step` 误用 `ROLLOUT_N=2`，只保留为排错记录，不作为正式结果。
- 当前活跃 25 step milestone run 使用 tmux `hpl_beacon_rl_to25_gpu125`，日志为 `logs/beacon_webshop_small_synth_b6n4_to25_seed1025.log`；固定 `ENV_SEED=1025`，从同名 `global_step_3` calibration checkpoint 自动 resume 到目标 step 25。
- BEACON 只有训练器 `_validate()` / `trainer.val_only=True` 这类重型评测路径；base、SFT LoRA、RL LoRA 三组横向评测优先走 HPL `evaluation.py`。评测前必须先把 HPL small/synthetic config 和 `indexes_1k` 小索引对齐。
