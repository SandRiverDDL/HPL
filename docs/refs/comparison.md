# 方法横向对比

| 方法 | 核心思想 | 偏好粒度 | 是否依赖 MC rollout | 对 WebShop 成本 | 对 HPL 的启发 |
|---|---|---|---|---|---|
| HPL | 结合 trajectory、step、group 三种偏好，用 group curriculum 缓解粒度不匹配 | 多粒度，核心是 group-level | 是，group/step 奖励估计依赖 rollout | 中高 | 当前主线，重点看 group 构造和 curriculum |
| IPR | 沿专家轨迹每一步探索候选动作，用 MC 估计 step-level reward | step-level | 强依赖 | 高 | 可作为 HPL 为什么要 group-level 的反例和 baseline |
| ETO | agent 自己探索失败轨迹，与成功轨迹形成 DPO 偏好对 | trajectory-level | 弱于 IPR，主要收集完整探索轨迹 | 中 | 可借鉴 failure trajectory 数据来源 |
| DEPO | 同时优化 token-per-step 和 steps-per-trajectory 的双效率 | efficiency preference | 不一定依赖 step MC | 中 | 可给 HPL 增加效率维度，避免会做但太啰嗦/太慢 |
| EEF | 从失败专家轨迹中筛选有益动作和 recovery 片段，再用 SFT 学习 | failed expert segment | 需要从若干 expert 中间状态模拟 | 中 | 比 HPL 更像 RFT 增强版，可能更适合低成本 WebShop 改进 |
| HSL | 用强模型把 rollout 中实际完成的目标重新标注成 hindsight goal，再做 SFT/DPO | hindsight relabeling | 不依赖 step MC，但依赖强 relabeler | 中 | 可把失败或偏离原目标的轨迹转成可学习正样本 |

## 对本仓库的实际判断

- 如果目标是理解 HPL，先不要全量跑 `pipeline/step/run_mc_sample.sh`。
- WebShop 上最值得先验证的是 group-level：`group_semantic.py` -> `run_negative.sh` -> `run_negative_mc.sh` -> `construct_preference_group.py`。
- IPR 的价值是帮助理解 step-level reward 的上限和成本，不适合作为单卡 4090 的默认全量复现路线。
- DEPO 的价值不是替代 HPL，而是给 HPL 的 group preference 增加效率约束：更短 Thought、更少点击、更快完成。
- EEF 的价值是利用原本会丢掉的失败 expert trajectories；如果能拿到 GPT-4/GPT-3.5 失败轨迹，性价比可能高于 HPL。
- HSL 比 EEF 更像低成本主线候选：它不要求失败 expert 轨迹本身可恢复成功，而是把 rollout 中已经达成的子目标重新标注后学习。
