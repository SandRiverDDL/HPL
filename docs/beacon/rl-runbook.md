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
data.train_batch_size=9
env.rollout.n=4
data.max_prompt_length=4096
data.max_response_length=512
actor_rollout_ref.rollout.max_model_len=4608
actor_rollout_ref.rollout.gpu_memory_utilization=0.55
actor_rollout_ref.actor.ppo_mini_batch_size=12
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.actor.optim.lr=1e-5
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=False
trainer.logger=['console','wandb']
trainer.test_freq=20
trainer.save_freq=20
```

LoRA 初始化：

```text
base model: Qwen2.5-1.5B-Instruct 本地 HF snapshot
adapter: saves/qwen2_5_1p5b_lora_step_sft
```

## Batch 与训练量

当前口径：

```text
每步 rollout trajectory = data.train_batch_size * env.rollout.n = 9 * 4 = 36
```

如果 `len(train_dataloader)=1` 且 `trainer.total_epochs=200`：

```text
总 update step = 200
总 rollout trajectory = 7200
最大 env step 上限 = 7200 * env.max_steps(15) = 108000
```

当前 `trainer.save_freq=20`，因此 200 step 训练会在：

```text
global_step 20, 40, 60, ..., 200
```

保存 checkpoint。最后一步满足 `is_last_step`，也会保存。

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

## 当前风险

- WebShop RL 还未跑通完整 1 个 update step；最新重试已到 `WebShop train goal count: 11389`，说明 full WebShop train env 已能加载，后续失败点是 Ray actor 暂时不可用，尚未进入 GPU/vLLM/FSDP 阶段。
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

- BEACON WebShop env reset/step smoke。
- `scripts/beacon/run_webshop_lora_rl.sh` 跑通至少 1 个 update step。
