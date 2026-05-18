# 架构说明

## 目标

当前目录承载 HPL 复现、环境配置和训练入口 smoke。核心目标不是改写 HPL，而是把第三方项目整理成可复现、可调试、可迁移的本地工作区。

核心设计目标：

1. 保留 upstream HPL 原始结构。
2. 本地 smoke、环境文档和必要配置与原始论文配置分开。
3. WebShop、ALFWorld、InterCode-SQL 三类环境按任务隔离。
4. 训练入口统一走 LLaMA-Factory。
5. 每个可执行入口都能从文档反查依赖、数据和命令。

## 顶层结构

```text
data/HPL/
  AGENTS.md
  README.md
  LOCAL_SETUP.md
  docs/
  .venv/
  LLaMA-Factory/
  envs/
  data/
  tools/
  scripts/
  train/
  smoke_data/
  saves/
  pipeline/
  evaluation.py
```

## 分层

### HPL 原始代码层

主要目录：

- `agents/`
- `tasks/`
- `envs/`
- `pipeline/`
- `prompt/`
- `configs/`
- `evaluation.py`

职责：

- agent 抽象与 OpenAI-compatible 调用。
- task/env 封装。
- trajectory、step、group preference 数据构造。
- evaluation 编排。

本地原则：

- 不随意改 upstream 逻辑。
- 如必须 patch，先定位原因，再记录到 `docs/ENVIRONMENT_SETUP.md` 或专门 patch 文档。

### 环境层

主要目录：

- `envs/webshop/`

职责：

- WebShop 源码。
- WebShop 数据。
- Lucene/pyserini 搜索索引。

当前状态：

- WebShop 源码来自 `princeton-nlp/WebShop`。
- 数据解压到 `envs/webshop/data`。
- 索引解压到 `envs/webshop/search_index/indexes`。
- 为兼容 WebShop 源码，建立软链 `envs/webshop/search_engine/indexes`。

后续 ALFWorld / InterCode-SQL 如果补全，应继续放在 HPL 项目内的稳定目录，不写入系统路径。

### 训练框架层

主要目录：

- `LLaMA-Factory/`
- `.venv/`
- `train/sft/`
- `train/sft_smoke/`

职责：

- LLaMA-Factory 提供 SFT/DPO 等训练 CLI。
- `.venv` 提供本项目隔离 Python 环境。
- `train/sft/` 保留 HPL 论文默认配置。
- `train/sft_smoke/` 是本地最小可运行配置。
- `train/sft/webshop_qwen2_5_1p5b_lora_step.yaml` 是当前 WebShop ETO step-level LoRA SFT 本地训练配置。

本地原则：

- 原始 `train/sft/*.yaml` 不直接覆盖。
- 本地验证和调试优先新建 `train/*_smoke/` 或明确命名的本地配置。
- 真实训练前必须把模型路径、数据 registry、Deepspeed 配置和 GPU 数量显式化。
- 四卡本地 DP 训练通过 LLaMA-Factory `FORCE_TORCHRUN=1` 启动；shell 里需要把 `.venv/bin` 加进 `PATH`，保证子进程能找到 `torchrun`。

### 数据层

主要目录：

- `data/`
- `tools/data/`
- `smoke_data/`
- `envs/webshop/data`
- `envs/webshop/search_index`

职责：

- `data/`：HPL 原始任务索引与未来任务数据。
- `tools/data/`：本地数据转换工具，例如 ETO WebShop trajectory 到 step-level SFT。
- `smoke_data/`：本地最小训练样本，只用于验证入口。
- `envs/webshop/data`：WebShop 商品与目标数据。
- `envs/webshop/search_index`：WebShop Lucene 搜索索引。

本地原则：

- 大数据和索引不做宽泛检索。
- 不把 smoke 数据误当论文训练数据。
- 数据来源、下载方式和路径必须写入文档。

### 本地运维脚本层

主要目录：

- `scripts/`

职责：

- 存放不属于 HPL 原始 pipeline 的本地手动脚本。
- 当前 `scripts/train_model.py` 是 GPU 占卡脚本，不是真实训练入口。
- `scripts/start_posttrain_vllm.sh` 复用相邻 `post_train` 项目的 vLLM 环境，在 tmux 中提供 OpenAI-compatible server。

本地原则：

- 会占用 GPU 的脚本必须手动运行，运行前确认目标 GPU 空闲。
- 不把 GPU 占卡、运维脚本放进 `pipeline/`，避免和 HPL 方法实现混淆。

### 文档层

主要目录：

- `AGENTS.md`
- `docs/STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/ENVIRONMENT_SETUP.md`
- `docs/refs/`
- `LOCAL_SETUP.md`

职责：

- `AGENTS.md`：长期协作规则。
- `STATE.md`：当前可执行状态。
- `ARCHITECTURE.md`：结构和职责。
- `ENVIRONMENT_SETUP.md`：环境配置经过和验证命令。
- `docs/refs/`：agent 后训练方法调研、横向对比和可迁移改进想法。
- `LOCAL_SETUP.md`：轻量快速入口。

## 主链路

### 1. SFT Smoke

输入：

- `smoke_data/*_sft_smoke.json`
- `train/sft_smoke/*.yaml`
- 本机已有 Qwen3-0.6B

输出：

- `saves/smoke/.../adapter_model.safetensors`

流程：

```text
smoke json
-> LLaMA-Factory dataset registry
-> llamafactory-cli train
-> Qwen3-0.6B LoRA 1 step
-> adapter checkpoint
```

用途：

- 验证 LLaMA-Factory、tokenizer、模型加载、训练和保存链路。
- 不用于判断 HPL 方法有效性。

### 2. WebShop Environment Smoke

输入：

- `envs/webshop/data`
- `envs/webshop/search_engine/indexes`
- Java 11
- `PYTHONPATH=envs:envs/webshop:.`

输出：

- `reset ok`
- `step ok`

流程：

```text
WebAgentTextEnv
-> load_products
-> load_goals
-> LuceneSearcher
-> reset
-> search action
```

用途：

- 验证 WebShop 数据和搜索索引可用。

### 3. WebShop ETO Step-Level SFT

输入：

- `data/processed/gigpo/webshop_eto/webshop_step_sft_2048_raw.jsonl`
- `LLaMA-Factory/data/dataset_info.json`
- `train/sft/webshop_qwen2_5_1p5b_lora_step.yaml`
- 本机已有 Qwen2.5-1.5B-Instruct

输出：

- `saves/qwen2_5_1p5b_lora_step_sft`

流程：

```text
ETO WebShop trajectory
-> step-level full-history SFT jsonl
-> LLaMA-Factory ShareGPT role/content registry
-> Qwen2.5-1.5B-Instruct LoRA SFT
-> adapter checkpoint
```

当前口径：

- 只训练最后一个 assistant turn：`train_on_prompt=false`、`mask_history=true`。
- `enable_liger_kernel=true`。
- 4 卡 DP：`per_device_train_batch_size=8`、`gradient_accumulation_steps=1`，总 batch size 为 32。

### 4. 原论文 SFT

输入：

- `train/sft/*.yaml`
- 真实 expert trajectory / SFT 数据
- Qwen2.5-7B-Instruct 或替换后的本地模型
- 多卡与 DeepSpeed 配置

输出：

- HPL base agent SFT checkpoint

当前状态：

- 原始 YAML 保留，但未按论文真实数据跑通。
- 需要先补数据和 registry，再考虑全量训练。

### 5. Preference Data Pipeline

主要入口：

- `pipeline/trajectory/`
- `pipeline/step/`
- `pipeline/group/`

职责：

- 生成 base agent trajectory。
- 构造 trajectory-level preference。
- 构造 step-level preference。
- 构造 group-level preference。

当前状态：

- 还未完成真实全链路 smoke。
- 后续应按 task 小样本逐段验证，不直接全量运行。

### 6. Evaluation

入口：

- `evaluation.py`

依赖：

- task config：`configs/task/*.json`
- model config：`configs/model/*.json`
- OpenAI-compatible API server 或真实 API
- 对应任务环境数据

当前状态：

- WebShop 环境本身已 reset/step smoke。
- evaluation 还未接本地模型 server 做完整 smoke。

## 维护原则

- 第三方源码优先保持可 pull。
- 本地新增配置、数据样例和文档必须明确标注 smoke / local。
- 大型数据下载和环境差异必须记录，避免下一台服务器重复排查。
- 若后续要系统学习 LLaMA-Factory、WebShop 或 HPL 算法，建议按 `docs/STATE.md` 中的下一步逐条推进。
