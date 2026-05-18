# BEACON WebShop Split

## 当前 small/synthetic 主线

为对齐 BEACON/SPEAR/verl-agent 默认 WebShop agentic RL 口径，当前 RL 主线使用：

```text
env.webshop.use_small=True
env.webshop.human_goals=False
```

实际小商品集 synthetic goals 数量已验证为 `6910`。固定 split：

```text
test：0..199
dev/validate：200..249
unused eval reserve：250..499
train：500..6909
```

HPL eval 配置：

```text
configs/task/webshop_small_synth_test200.json
configs/task/webshop_small_synth_dev50.json
```

评测前还必须对齐 WebShop 搜索索引：

```text
商品文件：envs/webshop/data/items_shuffle_1000.json
属性文件：envs/webshop/data/items_ins_v2_1000.json
搜索索引：envs/webshop/search_engine/indexes_1k
```

原 WebShop 代码中 `num_products=1000` 会选择 `indexes_1k`；`num_products=null` 会选择 `indexes`。因此 HPL small/synthetic 评测配置必须显式使用 `num_products=1000`，并保证 `indexes_1k` 指向与 BEACON small/synthetic 口径一致的小索引。当前该索引对齐仍是待办，未修正前不要把 HPL `webshop_small_synth_*` 评测结果作为 BEACON 可比结果。

HPL 启动 BEACON 时默认传入：

```bash
env.webshop.use_small=True
env.webshop.human_goals=False
env.webshop.train_start_idx=500
env.webshop.val_start_idx=200
env.webshop.val_end_idx=250
env.webshop.exclude_goal_indices_path=null
```

## full human 旧口径风险

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

## full human 旧采样规则

此前 HPL 启动 BEACON 时默认传入：

```bash
env.webshop.use_small=False
env.webshop.human_goals=True
env.webshop.train_start_idx=500
env.webshop.val_start_idx=0
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
