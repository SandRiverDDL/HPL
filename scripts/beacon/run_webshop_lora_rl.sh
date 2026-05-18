#!/usr/bin/env bash
set -euo pipefail

# HPL 侧只做编排；实际训练逻辑仍由 BEACON/verl-agent 执行。
BEACON_DIR="${BEACON_DIR:-/mnt/dataset/fengshuwen/agent-post-train/BEACON}"
PYTHON_BIN="${PYTHON_BIN:-${BEACON_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/fengshuwen/data/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/mnt/dataset/fengshuwen/HPL/saves/qwen2_5_1p5b_lora_step_sft}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"
# 三卡默认 smoke：real batch = TRAIN_BATCH_SIZE * ROLLOUT_N = 6，可被 3 张卡整除。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-3}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-3}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-6}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
ENV_SEED="${ENV_SEED:-0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-hpl_lora_migpo_qwen2_5_1p5b_webshop_small_smoke}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
SEPARATE_LORA_ADAPTER_DIR="${SEPARATE_LORA_ADAPTER_DIR:-/mnt/dataset/fengshuwen/HPL/saves/beacon_lora_adapters/${TRAINER_EXPERIMENT_NAME}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${BEACON_DIR}:${PYTHONPATH:-}"
export VERL_SEPARATE_LORA_ADAPTER_DIR="${SEPARATE_LORA_ADAPTER_DIR}"

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
TOTAL_TRAINING_STEPS_ARG=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  TOTAL_TRAINING_STEPS_ARG=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

cd "${BEACON_DIR}"

"${PYTHON_BIN}" -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size "${TRAIN_BATCH_SIZE}" \
  --val_data_size "${VAL_BATCH_SIZE}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=migpo \
  data.train_files="${HOME}/data/verl-agent/text/train.parquet" \
  data.val_files="${HOME}/data/verl-agent/text/test.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
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
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len=4608 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  +algorithm.migpo.step_advantage_w=1 \
  +algorithm.migpo.mode=mean_norm \
  +algorithm.migpo.gamma=0.95 \
  +algorithm.migpo.threshold=0.95 \
  env.env_name=Webshop \
  env.seed="${ENV_SEED}" \
  env.max_steps=15 \
  env.rollout.n="${ROLLOUT_N}" \
  env.resources_per_worker.num_cpus=0.1 \
  env.webshop.use_small=True \
  env.webshop.human_goals=False \
  env.webshop.train_start_idx=500 \
  env.webshop.val_start_idx=200 \
  env.webshop.val_end_idx=250 \
  env.webshop.exclude_goal_indices_path=null \
  trainer.critic_warmup=0 \
  trainer.resume_mode=auto \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=verl_agent_webshop \
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${TOTAL_TRAINING_STEPS_ARG[@]}" \
  trainer.val_before_train=False
