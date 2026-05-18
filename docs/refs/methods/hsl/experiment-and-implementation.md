# HSL 实验与实现

## 实验设置

| 项 | 内容 |
|---|---|
| 论文 | Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsight for Successful Demonstrations |
| 会议 | ICLR 2026 accepted |
| 任务 | ALFWorld, PlanCraft, WebShop |
| agent model | Llama3.2-1B |
| relabeler | Llama3.3-70B |
| 训练 | SFT+HSL, DPO+HSL |
| 运行资源 | 论文附录报告 ALFWorld 和 WebShop 实验使用 8 张 A100 |

## 关键实验结果

多随机种子结果中，WebShop 约为：

| 方法 | WebShop |
|---|---:|
| SFT | 62.61 |
| SFT+HSL | 66.48 |
| DPO | 69.27 |
| DPO+HSL | 70.28 |

主表中 WebShop 也显示：

| 方法 | WebShop |
|---|---:|
| SFT | 63.81 |
| SFT+HSL | 66.97 |
| DPO | 69.54 |
| DPO+HSL | 70.52 |

注意：这些数值和 EEF / HPL 的 WebShop 设置、模型大小、指标口径不完全一致，不能直接横向排序。

## 数据与成本

HSL 的成本来源和 HPL/IPR 不同：

- 需要收集 agent rollout。
- 需要强 relabeler 判断 trajectory 中实际完成的目标。
- 不需要对每个 step 做 MC reward。
- 不需要构造 step-level 或 group-level DPO pair。

论文附录报告每个 HSL training step 中，rollout collection、relabeling、loss/optimization 分别约为 11.53、62.03、24.03 秒；其中 relabeling 是主要额外成本。

## 迁移到本仓库

WebShop 方向可以先做 HSL-light：

1. 用当前 SFT agent 跑 100-200 条 WebShop trajectory。
2. 保留原始 instruction、observation、action、reward 和最终页面信息。
3. 用本地 7B/14B 或 API 模型判断轨迹实际完成了什么购物目标。
4. 对明显无关动作做简单 masking，例如重复 search、无效 click。
5. 生成 HSL SFT 数据。
6. 先用 LoRA SFT smoke 验证是否提升，再考虑接 DPO。

## 和当前 HPL 环境的关系

- 复用 WebShop 环境与 trajectory 采集。
- 复用 LLaMA-Factory SFT 训练入口。
- 不需要先跑 HPL 的 `run_negative_mc.sh`。
- 如果 HSL-light 有收益，再考虑把 HPL group 和 hindsight goal 结合。

## 待确认问题

- 当前 WebShop trajectory 是否保存了足够的页面状态，方便 relabeler 判断 achieved goal。
- 本地小模型 relabel 是否足够可靠，还是必须使用 70B/API。
- HSL 官方代码和数据是否可直接复用；当前只记录论文与会议信息。
