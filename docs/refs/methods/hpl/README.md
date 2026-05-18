# HPL

## 一句话

HPL（Hierarchical Preference Learning）用 trajectory、step、group 三种粒度的偏好优化 long-horizon LLM agent，其中核心创新是 **语义 action group + 双层 curriculum**，用于缓解 trajectory 太粗、step 太碎的问题。

## 适合解决的问题

- WebShop、ALFWorld、InterCode-SQL 这类长程交互任务。
- 最终奖励稀疏，但专家轨迹中存在可复用的多步子任务结构。
- step-level MC 成本太高、trajectory-level credit assignment 太粗的场景。

## 不适合的问题

- 没有专家轨迹，或专家轨迹质量很差。
- 环境 reset/step 很慢，且无法并行 rollout。
- 只关心单轮问答质量，不涉及多步 action。

## 关键结论

- HPL 不是只做 step-level DPO；它把 trajectory、step、group 三类偏好放在一起。
- group-level 是主要创新点：先把专家动作切成语义连贯 group，再生成负 group 做 preference。
- curriculum 有两个轴：group length 表示子任务复杂度，reward gap 表示样本难度。
- 本地仓库没有论文 expert trajectories，只能先做小样本链路验证。

## 对 HPL 改进的启发

- 可优先研究 group 切分策略，而不是先全量复现 step-level MC。
- 可以把 DEPO 的效率目标加入 group-level preference：同样成功时偏好更少 token、更少点击。
- 可以借鉴 ETO 的 failure trajectory 来源，减少对 expert-only 数据的依赖。

## 资料来源

- paper: https://arxiv.org/abs/2510.03253
- code: https://github.com/PatrickG1014/HPL
- local repo: `/mnt/dataset/fengshuwen/HPL`
