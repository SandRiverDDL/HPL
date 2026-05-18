# BEACON RL 排错与风险

本文件记录 WebShop RL 的运行风险和已定位问题。启动命令见 `rl-runbook.md`，结果见 `rl-results.md`。

## GPU 与占卡

- 正式训练前必须确认目标 GPU 真空闲。
- 若目标 GPU 上是本项目 `scripts/train_model.py` 占卡，可以直接停止并在任务结束后恢复。
- 不要杀未知 Python、Ray、vLLM、训练或评测进程。
- 如果 GPU 被其他用户进程占用，等待或换卡，不要自动抢。

## 已知 OOM

`VLLM_GPU_MEMORY_UTILIZATION=0.65` 在 seed2048 lr2e-6 25-step run 中失败：

```text
阶段：首个 rollout 前
位置：FSDP -> vLLM LoRA 参数同步
错误：CUDA out of memory
结果：未完成任何 global_step
```

后续重跑优先：

```text
VLLM_GPU_MEMORY_UTILIZATION=0.55
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
```

如果仍然 OOM，再把两个 micro batch 降到 `2`。

## WebShop 资源风险

- full WebShop actor 并发过高会导致 Ray OOM。
- 当前 RL 默认使用 small/synthetic 1K 商品索引。
- 不要把 full human WebShop 结果和 small/synthetic 结果直接比较。
- WebShop/pyserini 必须显式使用 Java 11。

## BEACON Patch 风险

当前 BEACON LoRA adapter 初始化、WebShop split、W&B allowlist 都依赖本地 patch。重置 BEACON 后必须复查：

```text
docs/beacon/patches.md
docs/beacon/webshop-split.md
scripts/beacon/run_webshop_lora_rl.sh
```

已有 patch 涉及：

- 已有 LoRA adapter 加载。
- WebShop train/dev/test split。
- small/synthetic 1K 商品索引。
- 独立 LoRA adapter 保存。
- W&B metric allowlist。
- MiGPO milestone 兼容问题。

## 训练曲线解释

- 当前没有真实 value critic。
- `critic/rewards/*`、`critic/score/*` 是 verl 历史命名，不作为主监控指标。
- 训练侧 `episode/webshop_task_score` 和独立 HPL eval 的 avg reward 口径不同，最终以固定 split 的 HPL eval 为准。
- `actor/entropy_loss` 名字带 loss，但本地代码里可作为聚合后的 response token entropy 观察。

## 不推荐操作

- 不把 `actor_rollout_ref.rollout.n` 改大；WebShop group rollout 由 `env.rollout.n` 控制。
- 不在训练中打开频繁 `_validate()`；训练后单独评测。
- 不用 BEACON `trainer.val_only=True` 做轻量横向评测。
- 不照装 WebShop 原始 `requirements.txt`。
