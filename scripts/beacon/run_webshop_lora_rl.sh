#!/usr/bin/env bash
set -euo pipefail

# HPL 侧只做编排；实际训练逻辑仍由 BEACON/verl-agent 执行。
BEACON_DIR="${BEACON_DIR:-/mnt/dataset/fengshuwen/agent-post-train/BEACON}"
PYTHON_BIN="${PYTHON_BIN:-${BEACON_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/fengshuwen/data/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/mnt/dataset/fengshuwen/HPL/saves/qwen2_5_1p5b_lora_step_sft}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${BEACON_DIR}:${PYTHONPATH:-}"

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

cd "${BEACON_DIR}"

"${PYTHON_BIN}" -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size 9 \
  --val_data_size 32

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=migpo \
  data.train_files="${HOME}/data/verl-agent/text/train.parquet" \
  data.val_files="${HOME}/data/verl-agent/text/test.parquet" \
  data.train_batch_size=9 \
  data.val_batch_size=32 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=12 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.max_model_len=4608 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  +algorithm.migpo.step_advantage_w=1 \
  +algorithm.migpo.mode=mean_norm \
  +algorithm.migpo.gamma=0.95 \
  +algorithm.migpo.threshold=0.95 \
  env.env_name=Webshop \
  env.seed=0 \
  env.max_steps=15 \
  env.rollout.n=4 \
  env.resources_per_worker.num_cpus=0.1 \
  env.webshop.use_small=False \
  env.webshop.human_goals=True \
  env.webshop.train_start_idx=500 \
  env.webshop.val_end_idx=500 \
  env.webshop.exclude_goal_indices_path=/mnt/dataset/fengshuwen/HPL/data/webshop/test_indices.json \
  trainer.critic_warmup=0 \
  trainer.resume_mode=auto \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=verl_agent_webshop \
  trainer.experiment_name=hpl_lora_migpo_qwen2_5_1p5b_gpu124 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.total_epochs=200 \
  trainer.val_before_train=False
