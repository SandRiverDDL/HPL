# HSL

## 一句话

HSL（Hindsight Supervised Learning）把 agent rollout 中实际完成的目标重新标注成自然语言 hindsight goal，再把这些轨迹作为成功演示做 SFT，也可以接在 DPO 前后使用。

## 论文状态

- 论文名：Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsight for Successful Demonstrations。
- 会议状态：ICLR 2026 accepted，poster 时间为 2026-04-24。
- 方法关键词：hindsight relabeling, agent post-training, SFT, DPO。

## 适合解决的问题

- 原始任务失败，但轨迹中其实完成了别的有意义子目标。
- agent rollout 成本已经付出，希望提高失败或偏离目标轨迹的数据利用率。
- 不想做 IPR/HPL 那种 step-level 或 group-level MC rollout。

## 不适合的问题

- 环境目标非常单一，没有可重新标注的 achieved goal。
- 没有强 relabeler，无法可靠判断轨迹实际完成了什么。
- 希望完全不依赖外部大模型或 API。

## 关键结论

- HSL 是 ICLR 2026 accepted，比 EEF 这种未确认接收的工作更适合作为主线候选。
- WebShop 上 HSL 对 SFT 和 DPO 都有提升，但提升幅度小于 ALFWorld。
- 成本主要来自 rollout collection 和 70B/API relabeler，不是 step-level MC。

## 对 HPL 的启发

- HPL 把失败轨迹切成 group 找偏好；HSL 提醒我们也可以先问“这条轨迹实际完成了什么”。
- 对 WebShop 来说，HSL 可作为低成本数据增强：保留可重标注轨迹，再进行 SFT 或 DPO。
- 如果 HPL 的 group negative 太贵，可以先做 HSL-style relabeling，验证失败轨迹是否有可学习信号。

## 资料来源

- paper: https://openreview.net/forum?id=QNfmqMSR7r
- ICLR poster: https://iclr.cc/virtual/2026/poster/10009594
