# BEACON WebShop Split

## 为什么要改

HPL 的正式 WebShop test ids 在：

```text
data/webshop/test_indices.json
```

当前统计：

```text
数量：200
范围：314..11896
```

BEACON 原始 train split 是 `500..len(goals)-1`。如果切到 full human goals，HPL test ids 中大量 id 会落入 BEACON train split，造成 RL 训练污染 test set。

## 当前采样规则

HPL 启动 BEACON 时默认传入：

```bash
env.webshop.use_small=False
env.webshop.human_goals=True
env.webshop.train_start_idx=500
env.webshop.val_end_idx=500
env.webshop.exclude_goal_indices_path=/mnt/dataset/fengshuwen/HPL/data/webshop/test_indices.json
```

实际 train candidates：

```text
range(500, len(goals)) - data/webshop/test_indices.json
```

val candidates：

```text
range(0, 500)
```

## 验证结果

按 full human goals 粗算：

```text
excluded test ids: 200
train candidates: 11389
overlap: 0
min_train: 500
max_train: 12086
```

## Prompt 与任务 ID

`examples.data_preprocess.prepare --mode text` 只生成 dummy parquet：

- `prompt` 内容基本为空字符串。
- `extra_info.index` 不是 WebShop task id。
- 真实 WebShop task id 在环境 reset 时从 `goal_idxs` 中随机采样。

因此防止 test 泄漏必须改 WebShop env 的 `goal_idxs`，不能只改 parquet。
