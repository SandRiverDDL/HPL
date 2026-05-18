# EEF

## 一句话

EEF（Exploring Expert Failures）不是把失败专家轨迹整体当负样本，而是从失败专家轨迹里筛出有益动作或 recovery 片段，再作为 SFT 正样本训练 agent。

## 适合解决的问题

- 有大量 expert trajectories，其中很多失败，但失败轨迹里包含有价值的中间动作。
- RFT/SFT 只学习成功轨迹，导致训练集偏向简单任务。
- WebShop 这类任务存在 Next、Back、恢复搜索等关键 recovery 技能。

## 不适合的问题

- 环境不能从 expert intermediate state 继续模拟。
- 没有失败 expert trajectories。
- 失败轨迹大多是随机错误，没有可复用动作。

## 关键结论

- EEF 保留 RFT 的简单性，核心训练仍是 SFT loss。
- 它通过模拟少量 expert 中间状态，判断哪些失败轨迹片段对当前 policy 有帮助。
- 论文报告 WebShop 11k 上 EEF GPT-4 达到 58.4% win rate，EEF GPT-3&4 达到 62.0%，高于 RFT x6 的 53.6%。
- 这条路线对低成本改进 WebShop 可能比 HPL 更有性价比。

## 对 HPL 的启发

- HPL 的 negative group 主要用于偏好对；EEF 提醒我们失败轨迹中也可能有正向 group。
- 可以先从失败 expert trajectory 抽取 recovery group，再和 HPL group-level 结合。
- 如果目标是快速提高 WebShop 成功率，EEF/RFT 可能比完整 HPL 更值得先试。

## 资料来源

- paper: https://arxiv.org/abs/2504.13145
