# WebShop GiGPO/SPEAR SFT 数据格式

## 结论

GiGPO / SPEAR 的 WebShop RL 阶段不把 ETO 的 ShareGPT expert trajectory 当作主输入。RL 主流程是从 WebShop 环境在线 rollout：

```text
task / env seed
-> group rollout
-> policy 每一步生成一个 action
-> 环境返回 observation / reward / done
-> 计算 GRPO / GiGPO / SPEAR advantage
```

因此 ETO 数据更适合做 cold-start SFT 或离线偏好数据，不适合直接作为 GiGPO/SPEAR RL 数据。

## SFT 应该切成 step-level

如果要用 ETO WebShop expert trajectory 做 cold-start，推荐切成一条条 step-level SFT 样本：

```text
input = 当前任务 + 当前 observation + admissible actions + 可选短历史
output = 当前 step 的专家 thought/action
```

不要把整条 trajectory 放进一个 assistant response 里监督模型一次性输出所有动作。GiGPO / SPEAR 的 WebShop prompt 是每一步只要求模型 `take one action for the current step`，所以 step-level SFT 更贴近后续 RL rollout 分布。

## 输出标签格式

面向 GiGPO / SPEAR / verl-agent 时，assistant 输出应使用：

```text
<think>reasoning</think>
<action>search[keywords]</action>
```

或：

```text
<think>reasoning</think>
<action>click[value]</action>
```

注意：`<action>` 内部不要再保留 `Action:` 前缀。verl-agent 的 WebShop projection 会从 `<action>...</action>` 里直接提取环境动作，因此应输出纯动作字符串：

```text
正确：<action>search[long clip-in hair extension]</action>
错误：<action>Action: search[long clip-in hair extension]</action>
```

如果目标是继续兼容 HPL 本仓库的 `envs/webshop_env.py`，才需要保留 `Action:` 行；但这会和 GiGPO/SPEAR 的 projection 口径不一致。

## ETO 原始格式

`agent-eto/eto-sft-trajectory` 的 WebShop 文件是 ShareGPT 多轮格式：

```json
{
  "id": 0,
  "conversations": [
    {"from": "human", "value": "You are web shopping..."},
    {"from": "gpt", "value": "OK"},
    {"from": "human", "value": "WebShop [SEP] Instruction: ... [SEP] Search"},
    {"from": "gpt", "value": "Thought: ...\nAction: search[...]"}
  ],
  "reward": 1.0,
  "source": "human"
}
```

转换时要做三件事：

1. 跳过开头 instruction / `OK`。
2. 按后续 `human -> gpt` pair 拆 step。
3. 把 `Thought:` / `Action:` 改写成 `<think>` / `<action>`。

## 推荐转换产物

第一版脚本产出两个文件：

```text
data/processed/gigpo/webshop_eto/webshop_step_sft.jsonl
data/processed/gigpo/webshop_eto/conversion_report.json
```

`webshop_step_sft.jsonl` 每行一个 step：

```json
{
  "id": "0_step_0",
  "traj_id": "0",
  "step_idx": 0,
  "messages": [
    {"role": "user", "content": "GiGPO/SPEAR 风格的当前 step prompt"},
    {"role": "assistant", "content": "<think>...</think>\n<action>search[...]</action>"}
  ],
  "raw_action": "search[...]",
  "reward": 1.0,
  "source": "human"
}
```

`conversion_report.json` 至少记录：

- 输入 trajectory 数。
- 输出 step 数。
- 重复 id。
- 解析失败样本数。
- 被过滤样本数。

## 当前本地转换结果

转换脚本：

```bash
.venv/bin/python tools/data/convert_eto_webshop_step_sft.py
```

实际输出：

```text
data/processed/gigpo/webshop_eto/webshop_step_sft_2048_raw.jsonl
data/processed/gigpo/webshop_eto/conversion_report.json
```

当前口径：

- WebShop only。
- 按 `id` 去重。
- 只保留 `reward >= 0.7`。
- 保留完整 history。
- 不做 observation cleaning，`clean_policy=none`。
- 使用 Qwen2.5-1.5B-Instruct tokenizer。
- 先用 Qwen chat template 渲染 `messages`，再 tokenize 统计长度。
- 原始 ETO `human` / `gpt` 已转换为标准 `user` / `assistant`。
- 只保留 token 长度 `<= 2048` 的 step 样本。

统计结果：

- 输入轨迹：1824。
- 去重后轨迹：1823。
- 重复 id：`4116`。
- `reward >= 0.7` 轨迹：1468。
- 原始 step：5730。
- 保留 step：5460。
- 超长过滤 step：270。
- 解析失败 step：0。
- 保留样本 token 长度：max 2048，p95 1717，p99 1955。

## RL prompt / manifest 的定位

可以额外生成 task-level manifest，但它不是 GiGPO/SPEAR 原生 RL 数据：

```text
task_id
initial_observation
reference_reward
source
expert_num_steps
```

这个 manifest 只适合做数据审计、任务筛选或后续迁移辅助。真正 RL 阶段仍应让 verl-agent/WebShop 环境在线 reset 和 rollout。

## 对本仓库的建议

- 当前 HPL 仓库最多用于 WebShop 环境验证、SFT、偏好学习和数据转换。
- 不建议在本仓库内实现 GiGPO/SPEAR 训练主循环。
- 后续 RL 主线应迁移到 `verl-agent` 或 SPEAR 的 fork。
- 本仓库转换脚本建议放在 `tools/data/`，不要放进 HPL 原始 `pipeline/`，避免和 HPL trajectory / step / group preference 链路混淆。

## 资料来源

- GiGPO / verl-agent: https://github.com/langfengQ/verl-agent
- verl-agent WebShop prompt: https://raw.githubusercontent.com/langfengQ/verl-agent/master/agent_system/environments/prompts/webshop.py
- verl-agent WebShop projection: https://raw.githubusercontent.com/langfengQ/verl-agent/master/agent_system/environments/env_package/webshop/projection.py
- verl-agent WebShop training script: https://raw.githubusercontent.com/langfengQ/verl-agent/master/examples/gigpo_trainer/run_webshop.sh
- SPEAR: https://github.com/TencentYoutuResearch/SPEAR
- SPEAR WebShop training script: https://raw.githubusercontent.com/TencentYoutuResearch/SPEAR/main/verl-agent/examples/gigpo_trainer/run_webshop_spear.sh
