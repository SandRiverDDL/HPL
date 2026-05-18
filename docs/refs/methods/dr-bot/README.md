# Dr.BoT 设置记录

Dr.BoT 是 SPEAR 论文提出的强 GRPO/GiGPO baseline，不是 SPEAR 本体。它把若干 agentic RL 训练 trick 组合在一起，目标是在不引入 self-imitation replay 的情况下提升探索、稳定性和训练效率。

## 论文设置

| 设置 | 含义 | 作用 |
|---|---|---|
| Remove KL divergence | 去掉 reference KL loss / reward KL | 允许策略偏离初始模型以适应工具交互 |
| Clip-Higher | 非对称 PPO clip，放宽上界 | 对正 advantage 的低概率 token 给更大提升空间 |
| Remove intra-group normalization | advantage 只减 group mean，不除 group std | 避免低方差 prompt 被过度放大 |
| Remove length normalization | 不按单条 response 长度均分 loss | 缓解长度归一带来的偏置 |
| Filter over-long samples | 截断或超过最大生成长度的样本不进 loss | 避免错误终止样本污染训练 |
| Filter void-turn samples | 没有有效工具调用的轨迹不进 loss | 避免格式漂移、重复文本和无效交互导致 collapse |
| Filter low-variance groups | 全对/全错等低信息 group 不更新 | 只保留有对比信号的 rollout group |

SPEAR 在 Dr.BoT 之上再加入 self-imitation replay、advantage recalibration、intrinsic reward curriculum 和 covariance-based token clipping。

## 本地 BEACON 对齐状态

当前 HPL launcher：`scripts/beacon/run_webshop_lora_rl.sh`。

| Dr.BoT 设置 | 当前状态 | 本地配置 |
|---|---|---|
| Remove KL divergence | 已对齐 | `actor_rollout_ref.actor.use_kl_loss=False`、`algorithm.use_kl_in_reward=False` |
| Clip-Higher | 已对齐 | `actor_rollout_ref.actor.clip_ratio_low=0.2`、`actor_rollout_ref.actor.clip_ratio_high=0.28` |
| Remove intra-group normalization | 已对齐到 MiGPO 口径 | `+algorithm.migpo.mode=mean_norm` |
| Remove length normalization | 未对齐 | 当前沿用 verl 默认 `actor_rollout_ref.actor.loss_agg_mode=token-mean` |
| Filter over-long samples | 部分对齐 | prompt 过滤已开；response 截断样本是否 mask 需另查训练后处理 |
| Filter void-turn samples | 未确认 | WebShop 环境有 invalid action penalty，但不等价于 void-turn mask |
| Filter low-variance groups | 未开启 | 当前未设置 `algorithm.filter_groups.enable=True` |

## 对当前实验的启发

- 当前本地实验不能称为完整 Dr.BoT，更准确是 `MiGPO + no KL + mean_norm + invalid action penalty`。
- 无 KL 时不建议继续使用偏大的 LoRA RL 学习率；当前已把 `1e-5` 降到 `2e-6`。
- 若后续要靠 Dr.BoT 思路提高稳定性，优先补齐 `filter_groups.enable=True`，再考虑 response overlong / void-turn mask。

## 资料来源

- SPEAR: https://arxiv.org/abs/2509.22601
- SPEAR code: https://github.com/TencentYoutuResearch/SPEAR
