# BEACON 本地 Patch

## 提交

```text
7146bfa feat: support webshop lora adapter and split filtering
```

## LoRA Adapter 初始化

原 BEACON 支持：

- `actor_rollout_ref.model.path=/path/to/full_hf_model`
- `actor_rollout_ref.model.lora_rank>0` 新建随机 LoRA

本地新增：

```bash
actor_rollout_ref.model.path=/path/to/base_model
actor_rollout_ref.model.lora_adapter_path=/path/to/peft_adapter
```

修改文件：

- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/workers/fsdp_workers.py`
- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/trainer/config/ppo_trainer.yaml`

行为：

- tokenizer/config 仍从 base model 读取。
- LoRA rank 从 `adapter_config.json` 自动推断。
- 如果手写 `lora_rank` 且与 adapter rank 不一致，直接报错。
- 使用 `PeftModel.from_pretrained(..., is_trainable=True)` 让 adapter 可继续 RL 更新。

## WebShop Split 过滤

原 BEACON WebShop 采样：

```text
val:   0..499
train: 500..len(goals)-1
```

本地新增配置：

```yaml
env:
  webshop:
    train_start_idx: 500
    val_end_idx: 500
    exclude_goal_indices_path: null
```

修改文件：

- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_package/webshop/envs.py`
- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_manager.py`
- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/trainer/config/ppo_trainer.yaml`

回滚口径：

- 删除 `lora_adapter_path` 配置和 `PeftModel.from_pretrained` 分支即可恢复原 LoRA 行为。
- 删除 `train_start_idx/val_end_idx/exclude_goal_indices_path` 配置透传即可恢复原 WebShop split。
