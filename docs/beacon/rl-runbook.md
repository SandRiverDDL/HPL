# BEACON WebShop RL Runbook

## 启动入口

HPL 侧脚本：

```bash
scripts/beacon/run_webshop_lora_rl.sh
```

脚本会进入 BEACON 仓库并调用：

```bash
.venv/bin/python -m verl.trainer.main_ppo
```

长时间训练必须用 tmux 启动。

## 默认关键参数

```text
CUDA_VISIBLE_DEVICES=1,2,4
TRAIN_BATCH_SIZE=3
VAL_BATCH_SIZE=3
ROLLOUT_N=2
ENV_SEED=0
data.train_batch_size=${TRAIN_BATCH_SIZE}
data.val_batch_size=${VAL_BATCH_SIZE}
env.rollout.n=${ROLLOUT_N}
env.seed=${ENV_SEED}
env.webshop.use_small=True
env.webshop.human_goals=False
env.webshop.train_start_idx=500
env.webshop.val_start_idx=200
env.webshop.val_end_idx=250
data.max_prompt_length=4096
data.max_response_length=512
actor_rollout_ref.rollout.max_model_len=4608
actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.55}
actor_rollout_ref.actor.ppo_mini_batch_size=6
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
actor_rollout_ref.actor.optim.lr=1e-5
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=False
trainer.logger=['console','wandb']
trainer.test_freq=20
trainer.save_freq=20
trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-null}
trainer.max_actor_ckpt_to_keep=1
trainer.max_critic_ckpt_to_keep=1
VERL_SEPARATE_LORA_ADAPTER_DIR=saves/beacon_lora_adapters/${TRAINER_EXPERIMENT_NAME}
```

LoRA 初始化：

```text
base model: Qwen2.5-1.5B-Instruct 本地 HF snapshot
adapter: saves/qwen2_5_1p5b_lora_step_sft
```

## Batch 与训练量语义

BEACON WebShop 多轮环境里有两套 `n`：

```text
actor_rollout_ref.rollout.n = 1
env.rollout.n = group size / 每个 prompt 采样轨迹数
```

不要把 `actor_rollout_ref.rollout.n` 改成大于 1；BEACON `main_ppo.py` 明确断言 WebShop 环境任务用 `env.rollout.n` 实现 group rollout。

一个 `global_step` 的含义：

```text
1 global_step
= 从 dataloader 取 1 个 prompt batch
= 取 data.train_batch_size 个 prompt
= 每个 prompt rollout env.rollout.n 条轨迹
= 得到 data.train_batch_size * env.rollout.n 条 trajectories
= 用这批 trajectories 计算 reward / advantage
= 在该 batch 内按 PPO mini/micro batch 完成 actor update
```

当前有效 n=4 run：

```text
data.train_batch_size = 3
env.rollout.n = 4
每步 rollout trajectory = 3 * 4 = 12
ppo_mini_batch_size = 12
ppo_micro_batch_size_per_gpu = 2
启动时 WebShop train actor 数 = 3 * 4 = 12
```

当前 `ppo_mini_batch_size=12` 等于每步轨迹数，因此每个 global step 内只有 1 个 PPO mini-batch；`ppo_micro_batch_size_per_gpu=2` 是 mini-batch 内部的显存切分和梯度累计，不增加采样轨迹数。

## 日志规则

正式 WebShop RL 默认启用：

```bash
TRAINER_LOGGER="['console','wandb']"
```

smoke test 可以临时只使用终端输出：

```bash
TRAINER_LOGGER="['console']" scripts/beacon/run_webshop_lora_rl.sh
```

正式训练、正式评测或长时间 RL 必须使用 W&B，并在启动前确认 W&B project/name。

当前有效 50 step 正式短跑：

```bash
tmux: hpl_beacon_webshop_rl_n4_50step
log: logs/beacon_webshop_small_synth_n4_50step.log
TRAINER_EXPERIMENT_NAME=hpl_lora_migpo_qwen2_5_1p5b_webshop_small_n4_50step
TOTAL_TRAINING_STEPS=50
ROLLOUT_N=4
PPO_MINI_BATCH_SIZE=12
TEST_FREQ=0
SAVE_FREQ=25
```

`TEST_FREQ=0` 表示训练中不跑 `_validate()`，只看 W&B/console 的训练侧指标；dev/test 评测训练结束后单独跑。

废弃 run：

```text
hpl_lora_migpo_qwen2_5_1p5b_webshop_small_50step
```

该 run 误用了 smoke 默认 `ROLLOUT_N=2`，每个 prompt 只有 2 条轨迹，不作为正式结果使用。

W&B：

```text
project: verl_agent_webshop
run: hpl_lora_migpo_qwen2_5_1p5b_webshop_small_n4_50step
```

废弃 n=2 run 的第 1 step 记录：

```text
training/global_step=1
episode/success_rate=0.0
episode/webshop_task_score=0.219
episode/valid_action_ratio=0.949
episode/reward/mean=0.0
timing_s/step=129.323
timing_s/gen=87.016
timing_s/update_actor=30.400
perf/max_memory_allocated_gb=25.357
perf/max_memory_reserved_gb=26.127
```

因此 `nvidia-smi` 看到的低显存不是峰值；WebShop 交互/生成阶段和 actor update 阶段显存占用不同，日志中的 `perf/max_memory_*` 才是该 step 的峰值口径。当前有效 n=4 run 的耗时和峰值需要以 `logs/beacon_webshop_small_synth_n4_50step.log` 的新 step 指标为准。

当前有效 n=4 run 已观测到的前 4 个 step：

```text
step 1: success_rate=0.000, webshop_task_score=0.167, timing_s/step=178.999, timing_s/gen=91.670, timing_s/update_actor=63.195
step 2: success_rate=0.083, webshop_task_score=0.300, timing_s/step=159.292, timing_s/gen=90.013, timing_s/update_actor=50.187
step 3: success_rate=0.167, webshop_task_score=0.361, timing_s/step=148.888, timing_s/gen=83.301, timing_s/update_actor=47.427
step 4: success_rate=0.000, webshop_task_score=0.069, timing_s/step=181.527, timing_s/gen=93.916, timing_s/update_actor=63.363
```

这说明当前链路已经越过 LoRA 初始化、rollout、old log prob、advantage、actor update 和 step 级日志写入；后续仍需确认 `SAVE_FREQ=25` 时 checkpoint/独立 LoRA adapter 能正常保存。

当前活跃 25 step milestone run（2026-05-18）：

```bash
tmux: hpl_beacon_rl_to25_gpu125
log: logs/beacon_webshop_small_synth_b6n4_to25_seed1025.log
CUDA_VISIBLE_DEVICES=1,2,5
TRAINER_EXPERIMENT_NAME=hpl_lora_migpo_qwen2_5_1p5b_webshop_small_b6n4_calib3v2_to25
ENV_SEED=1025
TRAIN_BATCH_SIZE=6
ROLLOUT_N=4
PPO_MINI_BATCH_SIZE=24
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
VLLM_GPU_MEMORY_UTILIZATION=0.65
TOTAL_EPOCHS=25
TOTAL_TRAINING_STEPS=25
SAVE_FREQ=25
TEST_FREQ=0
```

该 run 使用 `trainer.resume_mode=auto`，同名目录已有 `global_step_3` calibration checkpoint，因此目标 `TOTAL_TRAINING_STEPS=25` 表示从 step 3 继续到 step 25，而不是额外训练 25 step。日志已确认 `env.seed=1025`、`WebShop train goal count: 6410`、`WebShop val goal count: 50`。

当前活跃 true-1K 25 step run（2026-05-18）：

```bash
tmux: hpl_beacon_true1k_rl_25_gpu125
log: logs/beacon_webshop_true1k_b6n4_25step_seed1025.log
CUDA_VISIBLE_DEVICES=1,2,5
TRAINER_EXPERIMENT_NAME=hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_25step_seed1025
ENV_SEED=1025
TRAIN_BATCH_SIZE=6
ROLLOUT_N=4
PPO_MINI_BATCH_SIZE=24
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
VLLM_GPU_MEMORY_UTILIZATION=0.65
TOTAL_EPOCHS=25
TOTAL_TRAINING_STEPS=25
SAVE_FREQ=25
TEST_FREQ=0
```

该 run 使用新的 experiment name，启动前确认 checkpoint 目录不存在，因此从 0 开始，不加载 `global_step_3` calibration checkpoint。BEACON 已在 `use_small=True` 时向 WebShop 传入 `num_products=1000`，训练和后续 HPL eval 均使用 `indexes_1k` 小索引。

BEACON 原 WebShop MIGPO 脚本：

```text
data.train_batch_size=16
env.rollout.n=8
trainer.total_epochs=200
总 rollout trajectory = 25600
最大 env step 上限 = 384000
```

## LR 调度

BEACON PPO 默认：

```yaml
actor_rollout_ref.actor.optim.warmup_style: constant
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio: 0.0
```

因此当前脚本是恒定学习率，无 warmup、无 cosine decay。

## 当前 Split

当前 RL 主线对齐 BEACON/SPEAR/verl-agent 默认 WebShop 口径：

```text
商品集：items_shuffle_1000.json + items_ins_v2_1000.json
goals：synthetic goals, human_goals=False
test：0..199
dev/validate：200..249
train：500..6909
unused eval reserve：250..499
```

HPL 侧 eval config：

```text
configs/task/webshop_small_synth_test200.json
configs/task/webshop_small_synth_dev50.json
```

注意：HPL 评测要和 BEACON small/synthetic 口径一致，不仅要使用 `items_shuffle_1000.json` 和 `items_ins_v2_1000.json`，还必须使用 WebShop 小索引。当前 HPL 侧 small/synthetic config 已设置 `num_products=1000`，会使用 `envs/webshop/search_engine/indexes_1k`；该索引已验证为 1000 docs。

## 当前风险

- WebShop RL 当前正式 n=4 run 已完成至少 4 个 update step；不再是“未跑通 1 个 update”的状态。
- full WebShop actor 并发过高会导致 Ray OOM；当前训练默认使用 small/synthetic，避免再次创建大量 full WebShop actor。
- 旧排错阶段修复过 MiGPO milestone numpy truth-value、PEFT/Transformers tensor-parallel 兼容、LoRA adapter 保存字符串/Enum 兼容等问题；这些都属于已有 adapter 冷启动路径的本地 patch，重置 BEACON 后必须一次性恢复。
- HPL launcher 默认 `trainer.max_actor_ckpt_to_keep=1`、`trainer.max_critic_ckpt_to_keep=1`，只保留最近一个 resume checkpoint；LoRA adapter 另存到 `saves/beacon_lora_adapters/${TRAINER_EXPERIMENT_NAME}/global_step_*`，用于评测、merge 和手工挑选。
- 正式短跑建议先用 `TOTAL_TRAINING_STEPS=50`、`TRAINER_LOGGER="['console','wandb']"`、`SAVE_FREQ=25`、`TEST_FREQ=0`。`TEST_FREQ=0` 会关闭训练中的 `_validate()`，dev/test 评测改为训练后单独跑，避免 WebShop validate 拖慢或干扰训练曲线。
- BEACON 没有独立轻量 WebShop eval runner；`trainer.val_only=True` 也会启动 Ray/vLLM/PPO trainer 体系。三组模型横向评测优先走 HPL `evaluation.py`。
- `lr=1e-5` 且关闭 KL loss 偏激进；如果 invalid action 或格式漂移上升，应改为 `3e-6` 或开启小 KL。
- BEACON WebShop 当前使用 `gym==0.24.0`，会输出停止维护 warning；不要升级到 `0.26.2`，否则旧 WebShop env 会因 `action_space/observation_space` 检查失败。
- 运行前必须确认 GPU 空闲；若目标 GPU 上是 HPL 的 `scripts/train_model.py` 占卡程序，可按 `AGENTS.md` 规则停止，并在任务结束后用默认参数恢复。
- BEACON `.venv` 必须单独安装 WebShop runtime 依赖；HPL `.venv` 可跑 WebShop 不代表 BEACON 可跑。
- WebShop/pyserini 必须显式使用 Java 11；启动脚本已设置 `JAVA_HOME` 与 `LD_LIBRARY_PATH`。

## 已验证

```bash
/mnt/dataset/fengshuwen/agent-post-train/BEACON/.venv/bin/python -m py_compile \
  /mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_package/webshop/envs.py \
  /mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_manager.py \
  /mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/workers/fsdp_workers.py

bash -n scripts/beacon/run_webshop_lora_rl.sh

JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_package/webshop/webshop \
/mnt/dataset/fengshuwen/agent-post-train/BEACON/.venv/bin/python - <<'PY'
from web_agent_site.envs import WebAgentTextEnv
from pyserini.search.lucene import LuceneSearcher
import spacy
spacy.load("en_core_web_sm")
print("IMPORT_OK")
PY
```

## 待验证

- 当前 n=4 50 step run 能否跑到 `SAVE_FREQ=25` 并正常保存 resume checkpoint 与独立 LoRA adapter。
- 当前 true-1K 25 step run 是否能从 0 跑到 `global_step_25` 并保存独立 LoRA adapter。
- 训练结束后用 HPL `evaluation.py` 对 Qwen2.5-1.5B-Instruct、SFT LoRA、RL LoRA 三组模型跑同一个 BEACON small/synthetic test200。
