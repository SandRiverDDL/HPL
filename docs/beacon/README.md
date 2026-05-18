# BEACON 集成文档

本目录记录 HPL 侧如何控制相邻 BEACON 仓库进行 WebShop agentic RL。BEACON 是外部执行引擎，HPL 当前只保留轻量编排脚本、测试 split 约束、本地 patch 说明和运行记录。

## 文档分工

- `environment.md`：BEACON 仓库、`.venv`、依赖和当前环境状态。
- `patches.md`：对 BEACON 源码做过的本地 patch、原因、影响和回滚口径。
- `webshop-split.md`：WebShop train/val/test 任务 id 采样、防泄漏规则和验证口径。
- `rl-runbook.md`：HPL 控制 BEACON 训练的启动入口、当前默认超参和评测命令。
- `rl-results.md`：RL run、checkpoint、W&B 和固定 dev/test 结果。
- `rl-troubleshooting.md`：OOM、GPU、WebShop 和 BEACON patch 风险。

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

- BEACON 当前本地 HEAD `e99cd46` 已支持已有 LoRA adapter 初始化、WebShop split 过滤和 W&B 指标 allowlist。
- HPL 已完成 true-1K small/synthetic WebShop RL25 与 test200 横向评测，当前结果记录见 `docs/beacon/rl-results.md`。
- BEACON `.venv` 已补 `gym==0.24.0` 与 WebShop runtime 依赖，WebShop import smoke 可通过。
- 当前 WebShop RL 主线已切回 BEACON/SPEAR 默认 small/synthetic 口径，并进一步对齐真 1K 小索引：`items_shuffle_1000.json`、`items_ins_v2_1000.json`、`indexes_1k`、`human_goals=False`。
- true-1K 25 step run 已从 0 跑完并保存独立 LoRA adapter：`saves/beacon_lora_adapters/hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_25step_seed1025/global_step_25`。
- seed2048 的独立 true-1K 50 step run 已完成；它不是从 RL25 checkpoint resume，因此只能作为独立 RL50 短跑结果解读。
- `lr=2e-6`、clip-higher 的新 25-step run 曾在 `VLLM_GPU_MEMORY_UTILIZATION=0.65` 下 OOM；`0.55` 重跑等待会话未开始训练，已关闭。
- 旧 run `hpl_lora_migpo_qwen2_5_1p5b_webshop_small_50step` 误用 `ROLLOUT_N=2`，只保留为排错记录，不作为正式结果。
- BEACON 只有训练器 `_validate()` / `trainer.val_only=True` 这类重型评测路径；base、SFT LoRA、RL LoRA 三组横向评测优先走 HPL `evaluation.py`。当前 test200 结果：base `SR=0.0050 / reward=0.1302`，SFT `SR=0.3550 / reward=0.7117`，RL25 `SR=0.3700 / reward=0.6731`。
- 正式 RL 仍默认使用 W&B，但 HPL launcher 只上传核心指标；`critic/rewards/*` 这类 legacy 统计名不再进入 W&B。console 保留全量日志，具体 allowlist 见 `docs/beacon/rl-runbook.md`。
