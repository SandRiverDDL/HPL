# BEACON RL 结果记录

本文件保存 run 和 eval 结果。启动命令见 `rl-runbook.md`，排错见 `rl-troubleshooting.md`。

## 当前 WebShop 口径

```text
商品/属性：items_shuffle_1000.json + items_ins_v2_1000.json
索引：indexes_1k
goals：synthetic, human_goals=False
test：0..199
dev：200..249
train：500..6909
goal_seed：233
```

该口径不可与 full human WebShop 直接混比。

## 有效训练

### seed2048 50 step

```text
experiment: hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_50step_seed2048
log: logs/beacon_webshop_true1k_b6n4_50step_seed2048.log
W&B: https://wandb.ai/zer0ptr-tsinghua-university/verl_agent_webshop/runs/o34l3975
seed: 2048
steps: 50
batch: TRAIN_BATCH_SIZE=6, ROLLOUT_N=4, PPO_MINI_BATCH_SIZE=24
vllm util: 0.65
status: 已完成
```

注意：这是从 SFT adapter 重新开始的独立 run，不是从 RL25 checkpoint resume 到 50 step。

### seed1025 25 step

```text
experiment: hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_25step_seed1025
log: logs/beacon_webshop_true1k_b6n4_25step_seed1025.log
seed: 1025
steps: 25
用时: 55m19s
step25 train success_rate: 0.208
step25 train webshop_task_score: 0.523
adapter: saves/beacon_lora_adapters/hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_25step_seed1025/global_step_25
status: 已完成
```

## 失败/取消

### seed2048 lr2e-6 25 step, vllm 0.65

```text
log: logs/beacon_webshop_true1k_b6n4_25step_lr2e6_seed2048.log
W&B run id: i6xi1m5e
status: 首个 rollout 前 OOM，未完成任何 step
原因: FSDP -> vLLM LoRA 参数同步阶段显存不足
```

### seed2048 lr2e-6 25 step, vllm 0.55

```text
tmux: hpl_wait_rl25_lr2e6_vllm055_gpu125
status: 仅等待 GPU2/5 空闲，未开始训练，已按用户要求关闭
```

## 固定 Test200

固定 synthetic goals 后的 `webshop_small_synth_test200`：

```text
SFT LoRA:
  output: outputs/webshop_small_synth_fixed_test200/sft_lora_step
  avg_reward: 0.7140
  success_rate: 0.3700

RL25 LoRA seed2048:
  adapter: saves/beacon_lora_adapters/hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_50step_seed2048/global_step_25
  output: outputs/webshop_small_synth_fixed_test200/rl_lora_step25_seed2048
  avg_reward: 0.7101
  success_rate: 0.4450

RL50 LoRA seed2048:
  adapter: saves/beacon_lora_adapters/hpl_lora_migpo_qwen2_5_1p5b_webshop_true1k_b6n4_50step_seed2048/global_step_50
  output: outputs/webshop_small_synth_fixed_test200/rl_lora_step50_seed2048
  avg_reward: 0.6354
  success_rate: 0.3250
```

一致性验证：

```text
instruction_same_sft_vs_rl25: 200/200
instruction_same_sft_vs_rl50: 200/200
```

结论：同一 seed2048 run 里，`global_step_25` 的 SR 高于 SFT 但 avg reward 略低；`global_step_50` 在 SR 和 avg reward 上都低于 SFT。

## 旧结果

以下结果未固定 synthetic goal seed，只作历史参考：

```text
dev50:
  SFT:             avg_reward=0.7898, SR=0.5000
  RL25 seed1025:  avg_reward=0.7265, SR=0.4800
  RL50 seed2048:  avg_reward=0.7370, SR=0.4400

test200:
  base:            avg_reward=0.1302, SR=0.0050
  SFT:             avg_reward=0.7117, SR=0.3550
  RL25 seed1025:  avg_reward=0.6731, SR=0.3700
  RL25 seed2048:  avg_reward=0.7026, SR=0.4450
  RL50 seed2048:  avg_reward=0.6387, SR=0.3200
```

旧结果里不同模型的相同 test id 可能对应不同 synthetic price threshold，不作为严格横向比较依据。
