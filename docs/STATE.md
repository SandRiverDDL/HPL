# State

本文件只做当前状态 dashboard：当前入口、当前运行中任务、可用产物、主要风险和下一步。环境安装流水、历史 smoke、BEACON patch 细节分别见 `docs/ENVIRONMENT_SETUP.md`、`docs/ARCHITECTURE.md` 和 `docs/beacon/`。

## 当前定位

- 仓库：`/mnt/dataset/fengshuwen/HPL`
- HPL fork：`git@github.com:SandRiverDDL/HPL.git`
- HPL upstream：`https://github.com/PatrickG1014/HPL`
- 主要目标：WebShop 上复现/学习 agent 后训练链路；当前主线已从 HPL 原始 rollout 转向 HPL 侧编排 BEACON 做 small/synthetic WebShop RL。
- 当前训练框架：SFT 走本仓库 `.venv` + `LLaMA-Factory`；RL 走相邻 BEACON 仓库 `/mnt/dataset/fengshuwen/agent-post-train/BEACON`，由 HPL 脚本编排。

## 当前环境摘要

- Python：本仓库 `.venv`，Python 3.11。
- PyTorch：`torch==2.7.1+cu126`。
- Java：WebShop/pyserini 使用 `/usr/lib/jvm/java-11-openjdk-amd64`。
- WebShop 数据：`envs/webshop/data`
- WebShop 全量索引：`envs/webshop/search_index/indexes`
- WebShop small/synthetic 1K 索引：`envs/webshop/search_engine/indexes_1k`
- ETO WebShop trajectory：`data/hf_cache/datasets--agent-eto--eto-sft-trajectory/snapshots/a8f745ef799a5201d2288b9c2a47e40a0ff70a4c/data/webshop_sft.json`
- ETO step-level SFT 数据：`data/processed/gigpo/webshop_eto/webshop_step_sft_2048_raw.jsonl`

环境复现、依赖版本和 WebShop smoke 细节见 `docs/ENVIRONMENT_SETUP.md`。

## 当前入口与产物

- SFT 训练配置：`train/sft/webshop_qwen2_5_1p5b_lora_step.yaml`
- SFT 产物：`saves/qwen2_5_1p5b_lora_step_sft`
- RL25 LoRA：`saves/beacon_lora_adapters/hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_25step_seed1025/global_step_25`
- WebShop ETO step 数据转换：`tools/data/convert_eto_webshop_step_sft.py`
- HPL 评测入口：`evaluation.py`
- BEACON RL 启动脚本：`scripts/beacon/run_webshop_lora_rl.sh`
- vLLM 启动脚本：`scripts/start_posttrain_vllm.sh`
- GPU 占卡脚本：`scripts/train_model.py`

WebShop 运行时环境变量见 `docs/ENVIRONMENT_SETUP.md`；不要只激活 `.venv` 就直接跑 WebShop。

## 当前 WebShop 口径

当前 RL/SFT 对照主线使用 BEACON/SPEAR 风格 small/synthetic WebShop：

```text
use_small=True, human_goals=False
商品/属性：items_shuffle_1000.json / items_ins_v2_1000.json
索引：indexes_1k
split：test 0..199, dev 200..249, train 500..6909
goal_seed=233
```

该口径不可与 full human WebShop 直接混比。split 和防泄漏细节见 `docs/beacon/webshop-split.md`。

## 最近 RL 与评测

```text
最近有效完成 run:
  tmux: hpl_beacon_true1k_rl_50_seed2048_gpu125
  log: logs/beacon_webshop_true1k_b6n4_50step_seed2048.log
  W&B run id: o34l3975
  GPU: 1,2,5
  status: 已完成 50/50

最近失败/取消 run:
  log: logs/beacon_webshop_true1k_b6n4_25step_lr2e6_seed2048.log
  W&B run id: i6xi1m5e
  status: 首个 rollout 前 FSDP->vLLM LoRA 参数同步 OOM，未完成任何 step
  cancelled: hpl_wait_rl25_lr2e6_vllm055_gpu125 仅等待 GPU2/5 空闲，未开始训练，已按用户要求关闭
```

有效 50-step run 关键参数：`ENV_SEED=2048`、`TRAIN_BATCH_SIZE=6`、`ROLLOUT_N=4`、`PPO_MINI_BATCH_SIZE=24`、`PPO_MICRO_BATCH_SIZE_PER_GPU=4`、`VLLM_GPU_MEMORY_UTILIZATION=0.65`、`TOTAL_TRAINING_STEPS=50`、`SAVE_FREQ=25`、`TEST_FREQ=0`。

当前没有正在运行的 RL 训练或自动轮询；仅保留 GPU1 上的本项目占卡进程 `scripts/train_model.py --gpus 1`。

注意：这个 run 使用新 experiment name `hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_50step_seed2048`，从 SFT adapter 重新开始；它不是从 RL25 checkpoint resume 到 50 step。

## 当前结果

small/synthetic dev50（旧结果；当时未固定 synthetic goal seed，仅作历史参考）：

```text
SFT:             avg_reward=0.7898, SR=0.5000
RL25 seed1025:  avg_reward=0.7265, SR=0.4800
RL50 seed2048:  avg_reward=0.7370, SR=0.4400
```

small/synthetic test200（旧结果；当时未固定 synthetic goal seed，仅作历史参考）：

```text
base:            avg_reward=0.1302, SR=0.0050
SFT:             avg_reward=0.7117, SR=0.3550
RL25 seed1025:  avg_reward=0.6731, SR=0.3700
RL25 seed2048:  avg_reward=0.7026, SR=0.4450
RL50 seed2048:  avg_reward=0.6387, SR=0.3200
```

固定 synthetic goals 后的 small/synthetic test200：

```text
SFT:             avg_reward=0.7140, SR=0.3700
RL25 seed2048:  avg_reward=0.7101, SR=0.4450
RL50 seed2048:  avg_reward=0.6354, SR=0.3250
```

一致性验证：SFT vs RL25、SFT vs RL50 的 task instruction 均为 `200/200` 完全一致。

当前最重要的观察：固定 test200 后，同一次 seed2048 重训中 `global_step_25` 的 SR 高于 SFT，但 avg reward 略低；`global_step_50` 在 SR 和 avg reward 上都低于 SFT，继续训到 50 step 仍有明显退化风险。此前旧结果里 step25/step50 的部分差异混入了 synthetic goal 随机漂移，不能再作为主要证据。

## 当前风险

- HPL 原仓库没有附带论文完整 expert trajectories；当前只使用 ETO WebShop SFT trajectory 和本地转换数据。
- 原论文 `train/sft/*.yaml` 默认 7B 全参 + ZeRO-3，不适合作为本机快速迭代入口。
- WebShop 原始依赖很旧，不要照单安装 WebShop `requirements.txt`；只补 runtime 缺口。
- HPL small/synthetic 评测必须使用 `num_products=1000/indexes_1k`，否则会和 BEACON RL 训练口径不一致。
- HPL small/synthetic 评测还必须使用固定 `goal_seed=233`；WebShop 原实现会随机生成 synthetic goal 价格和 instruction，未固定时不同模型评测的 test ids 不代表同一批 exact goals。
- BEACON 没有独立轻量 WebShop eval runner；横向评测优先走 HPL `evaluation.py`。
- 当前 BEACON LoRA adapter 初始化、WebShop split、W&B allowlist 都是本地 patch；重置 BEACON 后必须按 `docs/beacon/patches.md` 复查。
- `lr=2e-6` + clip-higher 的 25-step 新实验还没有有效结果；`VLLM_GPU_MEMORY_UTILIZATION=0.65` 已确认会在首个 rollout 前 OOM，后续若重启需改为 `0.55` 并确认 GPU2/5 真空闲。

## 下一步

- 若继续 RL，优先把 `global_step_25` 作为当前 seed2048 的候选 checkpoint，而不是 `global_step_50`。
- dev50 只有 50 题；后续 checkpoint 选择最好扩大 dev 或固定多个 dev 子集。
- 若要继续做对照实验，固定 split、seed、`ROLLOUT_N=4` 和评测脚本，避免把采样差异误当方法差异。

## 文档地图

- 稳定规则：`AGENTS.md`
- 架构分层：`docs/ARCHITECTURE.md`
- 环境配置：`docs/ENVIRONMENT_SETUP.md`
- BEACON 集成：`docs/beacon/README.md`
- BEACON patch：`docs/beacon/patches.md`
- BEACON RL runbook：`docs/beacon/rl-runbook.md`
- BEACON RL 结果：`docs/beacon/rl-results.md`
- BEACON RL 排错：`docs/beacon/rl-troubleshooting.md`
- WebShop split：`docs/beacon/webshop-split.md`
