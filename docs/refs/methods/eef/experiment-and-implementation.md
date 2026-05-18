# EEF 实验与实现

## 实验设置

| 项 | 内容 |
|---|---|
| 任务 | WebShop 11k, WebShop 3k, ScienceWorld 2k |
| 初始模型 | Llama3 8B Instruct；另有 Mistral-7B-v0.3 验证 |
| expert | WebShop 使用 GPT-4 生成 trajectories，也测试 GPT-3.5-Turbo 失败轨迹 |
| 训练 | SFT / RFT 风格，默认 3 次 fine-tune iteration |
| 默认参数 | `M=5`, `I=4`，论文表中 EEF GPT-4 为 3 个 FT iteration、6 次 simulation |
| 硬件 | 4 张 NVIDIA A6000 |

## 关键实验结果

| 方法 | WebShop 11k winrate | reward | WebShop 3k winrate | reward |
|---|---:|---:|---:|---:|
| GPT-4 | 35.6% | 0.66 | 35.6% | 0.66 |
| SFT POS | 46.4% | 0.75 | 39.6% | 0.67 |
| ETO | 42.0% | 0.68 | 37.8% | 0.67 |
| RFT | 52.0% | 0.75 | 38.8% | 0.66 |
| RFT x6 | 53.6% | 0.76 | 41.4% | 0.68 |
| EEF GPT-4 | 58.4% | 0.78 | 46.8% | 0.72 |
| EEF GPT-3&4 | 62.0% | 0.81 | 50.0% | 0.73 |

## 数据与 rollout 成本

EEF 仍然要探索，但不是 HPL/IPR 那种每个 step 做 MC reward：

- 每轮从所有初始 states 探索一次。
- 对每条 expert trajectory 只选 `M` 个中间状态模拟。
- 论文强调可降低 `M` 来换效率。
- Ablation 显示 `M=1` 时也能达到约 40% winrate，增加 simulation budget 继续提升。

性价比判断：

- 如果已有失败 expert trajectories，EEF 比 HPL 更直接。
- 它不需要 DPO 偏好对、不需要 reward gap curriculum、不需要 group MC。
- 主要难点是环境状态恢复和筛选 beneficial/recovery actions。

## 迁移到本仓库

WebShop 方向可以先做 EEF-light：

1. 用已有或新生成的失败 expert/base-agent trajectories。
2. 对每条失败轨迹选少量中间状态，例如 `M=1` 或 `M=2`。
3. 从这些状态继续 rollout，看当前 policy 是否能成功。
4. 只把能导向成功的后缀或 recovery segment 转成 SFT 数据。
5. 先用 LoRA SFT smoke 训练，不直接做 DPO。

## 最小复现建议

```text
200 条 WebShop trajectory
-> 区分 success/failure
-> 失败轨迹中每条取 1-2 个中间状态
-> 从中间状态 rollout
-> 只保留成功后缀
-> 加到 SFT 数据
-> 小规模 LoRA SFT
-> 用 100-200 eval task 看方向
```

## 待确认问题

- 当前 HPL WebShop env 是否方便恢复到 arbitrary intermediate state；如果不能，需要用 prefix replay 近似。
- EEF 是否有公开代码；当前只确认论文，未找到官方 GitHub。
- GPT-4/GPT-3.5 expert failure trajectories 是否能获取，否则需要自己生成失败轨迹。
