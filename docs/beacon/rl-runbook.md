# BEACON WebShop RL Runbook

本文件只保留“怎么启动”和“当前默认参数”。实验结果见 `rl-results.md`，OOM/排错见 `rl-troubleshooting.md`。

## 启动入口

HPL 侧编排脚本：

```bash
scripts/beacon/run_webshop_lora_rl.sh
```

脚本会进入 BEACON 仓库并调用：

```bash
/mnt/dataset/fengshuwen/agent-post-train/BEACON/.venv/bin/python -m verl.trainer.main_ppo
```

长时间训练必须放在 tmux 中运行；正式训练必须启用 W&B。

## 默认参数

脚本默认是 smoke 口径：

```text
CUDA_VISIBLE_DEVICES=1,2,4
TRAIN_BATCH_SIZE=3
VAL_BATCH_SIZE=3
ROLLOUT_N=2
PPO_MINI_BATCH_SIZE=6
PPO_MICRO_BATCH_SIZE_PER_GPU=2
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
VLLM_GPU_MEMORY_UTILIZATION=0.55
ENV_SEED=0
TOTAL_EPOCHS=1
SAVE_FREQ=20
TEST_FREQ=20
```

当前 RL 主线的关键算法参数：

```text
algorithm.adv_estimator=migpo
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.optim.lr=2e-6
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
algorithm.migpo.mode=mean_norm
algorithm.migpo.step_advantage_w=1
algorithm.migpo.gamma=0.95
algorithm.migpo.threshold=0.95
```

LoRA 初始化：

```text
base model: Qwen2.5-1.5B-Instruct 本地 HF snapshot
adapter: saves/qwen2_5_1p5b_lora_step_sft
```

## 正式 25 Step 模板

建议只在 GPU1/2/5 真空闲时启动：

```bash
tmux new-session -d -s hpl_beacon_rl25_lr2e6_gpu125 \
  'cd /mnt/dataset/fengshuwen/HPL && \
  CUDA_VISIBLE_DEVICES=1,2,5 \
  ENV_SEED=2048 \
  TRAIN_BATCH_SIZE=6 \
  VAL_BATCH_SIZE=6 \
  ROLLOUT_N=4 \
  PPO_MINI_BATCH_SIZE=24 \
  PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4 \
  VLLM_GPU_MEMORY_UTILIZATION=0.55 \
  TOTAL_EPOCHS=25 \
  TOTAL_TRAINING_STEPS=25 \
  SAVE_FREQ=25 \
  TEST_FREQ=0 \
  TRAINER_EXPERIMENT_NAME=hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_25step_lr2e6_vllm055_seed2048 \
  scripts/beacon/run_webshop_lora_rl.sh \
  > logs/beacon_webshop_true1k_b6n4_25step_lr2e6_vllm055_seed2048.log 2>&1'
```

`TEST_FREQ=0` 表示训练中不跑 `_validate()`；训练后用 HPL `evaluation.py` 单独评测。

## Batch 语义

BEACON WebShop 有两套 `n`：

```text
actor_rollout_ref.rollout.n = 1
env.rollout.n = 每个 prompt 的轨迹数
```

不要把 `actor_rollout_ref.rollout.n` 改大；WebShop group rollout 由 `env.rollout.n` 控制。

一个 `global_step`：

```text
data.train_batch_size 个 prompt
* env.rollout.n 条轨迹
= 本 step 的 trajectories
```

例如正式模板：

```text
TRAIN_BATCH_SIZE=6
ROLLOUT_N=4
每步 trajectories=24
PPO_MINI_BATCH_SIZE=24
```

## W&B 指标

默认只上传核心指标，console 保留 BEACON 原始全量日志。若临时需要恢复全量 W&B：

```bash
WANDB_METRIC_ALLOWLIST='*'
```

重点看：

```text
episode/webshop_task_score (not success_rate)
episode/success_rate
episode/valid_action_ratio
response_length/clip_ratio
prompt_length/clip_ratio
actor/entropy_loss
actor/ppo_kl
actor/pg_clipfrac
actor/pg_clipfrac_lower
actor/grad_norm
actor/lr
timing_s/step
timing_s/gen
timing_s/update_actor
```

当前没有真实 value critic；`critic/rewards/*`、`critic/score/*` 不作为主监控指标。

## 训练后评测

BEACON 没有轻量独立 WebShop eval runner；横向评测优先走 HPL：

```bash
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:${LD_LIBRARY_PATH:-} \
PYTHONPATH=envs:envs/webshop:. \
.venv/bin/python evaluation.py \
  --exp_config webshop_small_synth_test200 \
  --agent_config agent \
  --split test \
  --max_tasks 200 \
  --api_base http://127.0.0.1:8011/v1 \
  --api_key EMPTY \
  --model_name <served-model-name> \
  --output_dir outputs/webshop_small_synth_fixed_test200/<run-name> \
  --override
```

评测必须对齐 `webshop_small_synth_test200` / `goal_seed=233` / `indexes_1k` 口径。

## 当前不做

- 不跑 full WebShop 多 worker RL，避免再次炸内存。
- 不使用 BEACON `trainer.val_only=True` 做轻量横向评测。
- 不把 `VLLM_GPU_MEMORY_UTILIZATION` 调回 `0.65`，除非先确认不会在 LoRA 参数同步阶段 OOM。
