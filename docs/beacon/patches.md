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
- 加载已有 LoRA 前会为 `peft==0.19.1` 与 `transformers==4.51.1` 做 tensor_parallel 兼容：当前 `tensor_model_parallel_size=1`，不需要 adapter TP 分片，因此将 PEFT 的 `_maybe_shard_state_dict_for_tp` 置为 no-op，避免缺失 `ALL_PARALLEL_STYLES/EmbeddingParallel` 等符号时失败。
- 保存 LoRA checkpoint 时兼容 PEFT config 中 `task_type/peft_type` 已经是字符串的情况；原代码固定访问 `.value`，会在 `global_step_1` 保存时报 `AttributeError: 'str' object has no attribute 'value'`。
- BEACON 原本会在 actor checkpoint 内额外保存 `lora_adapter/`；本地新增 `VERL_SEPARATE_LORA_ADAPTER_DIR`，保存时会再复制一份到独立目录，避免 FSDP checkpoint rotation 删除历史 LoRA adapter。

### 一次性适配已有 adapter 的坑点

这条路径不是 BEACON 原生主路径，后续如果重建环境或重置 BEACON patch，需要一次性检查以下点：

1. `ppo_trainer.yaml` 必须有 `actor_rollout_ref.model.lora_adapter_path`，默认 `null`。
2. `fsdp_workers.py` 初始化 actor 时，`lora_adapter_path` 非空应走 `PeftModel.from_pretrained(actor_module, path, is_trainable=True)`，不要再新建随机 LoRA。
3. 从 adapter 的 `adapter_config.json` 自动推断 rank；如果用户同时传 `lora_rank` 且不一致，应直接报错。
4. `peft==0.19.1` 与 `transformers==4.51.1` 组合会触发 tensor parallel 符号缺失；当前 `tensor_model_parallel_size=1`，因此把 PEFT 的 `_maybe_shard_state_dict_for_tp` 置为 no-op。
5. 保存 adapter config 时，`task_type/peft_type` 可能已经是字符串，不能固定访问 `.value`。
6. FSDP resume checkpoint 与可迁移 LoRA adapter 分开管理：resume checkpoint 只保留最近 1 个，LoRA adapter 另存到 `VERL_SEPARATE_LORA_ADAPTER_DIR/global_step_*`。

这条路径带来的历史问题：

- LoRA 加载阶段：`ALL_PARALLEL_STYLES/EmbeddingParallel` import 失败。
- LoRA 保存阶段：`AttributeError: 'str' object has no attribute 'value'`。
- checkpoint 管理阶段：FSDP rank shard 和轻量 LoRA adapter 混在同一个 `global_step_*` 下，不便长期保存和评测。

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
    val_start_idx: 0
    val_end_idx: 500
    exclude_goal_indices_path: null
```

修改文件：

- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_package/webshop/envs.py`
- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_manager.py`
- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/trainer/config/ppo_trainer.yaml`

回滚口径：

- 删除 `lora_adapter_path` 配置和 `PeftModel.from_pretrained` 分支即可恢复原 LoRA 行为。
- 删除 `train_start_idx/val_start_idx/val_end_idx/exclude_goal_indices_path` 配置透传即可恢复原 WebShop split。

## MiGPO milestone 数组判断

原 `compute_advantage` 中使用：

```python
data.non_tensor_batch.get('milestone_achieved') or data.non_tensor_batch.get('subgoal_completed', None)
```

当 `milestone_achieved` 是 numpy bool 数组时，Python 会因为数组 truth value 歧义报错。本地改为显式 key 判断：

```python
if 'milestone_achieved' in data.non_tensor_batch:
    milestone_achieved = data.non_tensor_batch['milestone_achieved']
else:
    milestone_achieved = data.non_tensor_batch.get('subgoal_completed', None)
```

修改文件：

- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/trainer/ppo/ray_trainer.py`

## W&B 指标过滤

原 BEACON `Tracking.log()` 会把同一份完整 metric dict 同时写入 console 和 W&B，正式 RL 时 W&B 页面会混入大量低价值或命名误导的指标。

本地新增环境变量：

```bash
VERL_WANDB_METRIC_ALLOWLIST='metric_a,metric_prefix/*'
```

行为：

- 只过滤 `wandb` / `vemlp_wandb` 后端，console、MLflow、TensorBoard 等后端仍按原样记录完整指标。
- pattern 以 `*` 结尾时按前缀匹配；单独设置 `*` 表示 W&B 全量记录。
- HPL launcher 默认通过 `WANDB_METRIC_ALLOWLIST` 设置核心指标，不包含 `critic/rewards/*`。当前 MiGPO `use_critic=False`，这些 `critic/*` 名字只是 verl 数据统计的历史命名，不是真实 critic 模型输出。

修改文件：

- `/mnt/dataset/fengshuwen/agent-post-train/BEACON/verl/utils/tracking.py`
- `/mnt/dataset/fengshuwen/HPL/scripts/beacon/run_webshop_lora_rl.sh`

## HPL WebShop synthetic goal 固定

问题：HPL `evaluation.py` 只固定了 WebShop task id，但 small/synthetic WebShop 会在 env 初始化时随机生成商品价格和 synthetic goal 的 price upper。因此未固定 seed 时，不同模型评测的同一个 test id 可能对应不同 instruction，尤其是 price threshold 会漂移。

本地新增：

```json
"goal_seed": 233
```

修改文件：

- `/mnt/dataset/fengshuwen/HPL/configs/task/webshop_small_synth_test200.json`
- `/mnt/dataset/fengshuwen/HPL/configs/task/webshop_small_synth_dev50.json`
- `/mnt/dataset/fengshuwen/HPL/evaluation.py`
- `/mnt/dataset/fengshuwen/HPL/envs/webshop/web_agent_site/envs/web_agent_text_env.py`

行为：

- `evaluation.py` 将 `goal_seed` 透传给 `WebAgentTextEnv`。
- `WebAgentTextEnv` 将 `goal_seed` 传给 `SimServer`。
- `SimServer` 在 `load_products()` 前设置 seed，覆盖商品价格随机生成和 synthetic goal price upper 随机采样。

验证结果：

```text
configs/task/webshop_small_synth_test200.json: same=True, n=20
configs/task/webshop_small_synth_dev50.json: same=True, n=20
instruction_same_sft_vs_rl25: 200/200
instruction_same_sft_vs_rl50: 200/200
```
