# 环境配置记录

本文记录 HPL 在本服务器上的配置经过、关键决策和验证结果，避免后续重复踩依赖、Java、WebShop 路径和 CUDA 版本问题。

## 基本信息

- 当前项目路径：`/mnt/dataset/fengshuwen/HPL`
- 实际挂载：`/mnt/dataset`
- HPL commit：`dc23ada initial release`
- WebShop commit：`64fa2a5 Update README.md`
- LLaMA-Factory：本地 clone 到 `LLaMA-Factory/`
- Python 环境：`.venv`
- 包管理：`uv`

## 克隆内容

```bash
git clone https://github.com/PatrickG1014/HPL.git ~/data/HPL
cd ~/data/HPL
git clone https://github.com/princeton-nlp/WebShop.git envs/webshop
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
```

说明：

- HPL README 里的 `setup.sh` 使用 conda，不适合当前项目的 uv 口径。
- HPL 原仓库没有包含 `envs/webshop` 源码目录，所以需要单独 clone WebShop。
- HPL 原仓库引用 `llamafactory-cli` 和 `LLaMA-Factory/examples/deepspeed/...`，所以需要单独 clone LLaMA-Factory。

## uv 环境

创建环境：

```bash
uv venv --python python3.11 .venv
```

使用 Python 3.11 的原因：

- 当前 LLaMA-Factory `pyproject.toml` 要求 `requires-python >=3.11`。
- HPL README 中的 Python 3.10 是原始 conda 口径，不适用于当前 LLaMA-Factory 版本。

核心安装：

```bash
uv pip install -e LLaMA-Factory deepspeed
uv pip install --python .venv/bin/python liger-kernel
uv pip install backoff==2.2.1 colorama==0.4.6 google_search_results==2.4.2 \
  gym==0.24.0 intercode==0.0.7 openai pytz scikit-learn gdown alfworld==0.3.5 \
  beautifulsoup4 cleantext Flask pyserini rank_bm25 selenium thefuzz spacy
uv pip install pyserini==0.17.0
uv pip install faiss-cpu
.venv/bin/python -m spacy download en_core_web_sm
```

没有照单安装 WebShop `requirements.txt` 的原因：

- WebShop 原始依赖固定了旧 `torch==1.11.0` 和 `transformers==4.19.2`。
- 这些版本会破坏当前 LLaMA-Factory 训练环境。
- 当前做法是保留 LLaMA-Factory 所需现代训练依赖，只补 WebShop import/runtime 需要的包。

## PyTorch / CUDA

LLaMA-Factory 首次安装时拉到了 `torch==2.12.0+cu130`，当前服务器 driver 只支持到 CUDA 12.8，表现为：

```text
CUDA initialization: The NVIDIA driver on your system is too old
```

因此改为安装 CUDA 12.6 wheel：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
```

验证结果：

```text
torch 2.7.1+cu126
cuda available True
device0 NVIDIA GeForce RTX 4090
```

## WebShop 数据与索引

下载数据：

```bash
.venv/bin/gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9 -O envs/webshop/data.zip
cd envs/webshop
unzip -q -o data.zip
```

下载索引：

```bash
.venv/bin/gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y -O envs/webshop/indexes.zip
cd envs/webshop
mkdir -p search_index
unzip -q -o indexes.zip -d search_index/
```

索引路径修正：

```bash
ln -s ../search_index/indexes envs/webshop/search_engine/indexes
```

原因：

- 下载包解压后路径为 `envs/webshop/search_index/indexes`。
- WebShop 源码默认查找 `envs/webshop/search_engine/indexes`。
- 当前用软链兼容，不移动大文件。

### Preview 与全量评测

WebShop 源码默认是 preview 配置：

```python
DEFAULT_ATTR_PATH = items_ins_v2_1000.json
DEFAULT_FILE_PATH = items_shuffle_1000.json
```

当前在 `human_goals=True` 下只加载到 13 个 human goals；这与 HPL 的 `data/webshop/test_indices.json` 不匹配，测试 id 可能越界。

本地新增：

- `configs/task/webshop_local10.json`
- `data/webshop_local10/test_indices.json`
- `configs/task/webshop_full10.json`

该配置只使用 session id `0..9`，用于确认 vLLM、eval、WebShop reset/step 和动作格式是否打通；它不是正式 WebShop test split。

`webshop_full10` 使用 `data/webshop/test_indices.json` 的前 10 个测试 id，并通过 task config 指向全量：

```python
file_path = envs/webshop/data/items_shuffle.json
attr_path = envs/webshop/data/items_ins_v2.json
num_products = None
```

并确认：

- 使用 full `search_engine/indexes`。
- `Loaded goals` 数量覆盖 `data/webshop/test_indices.json` 的最大 session id。
- search 候选集合与正式评测一致，否则 reward / success 不可比较。
- WebShop success rate 以 `reward == 1` 统计，不直接使用输出 JSON 中的 `state.success` 字段。

2026-05-18 全量前 10 题 smoke：

- base：`outputs/compare_full10/base`，`avg_reward=0.1950`，`success=1/10`。
- LoRA：`outputs/compare_full10/lora`，`avg_reward=0.6697`，`success=2/10`。

## Java / pyserini

系统默认 Java：

```text
openjdk version "11.0.30"
```

运行 WebShop 必须显式设置：

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:${LD_LIBRARY_PATH:-}
export PYTHONPATH=envs:envs/webshop:.
```

已踩问题：

- 新版 `pyserini==1.6.0` 会使用当前 Java 11 不支持的 JVM 参数。
- 降到 `pyserini==0.17.0` 后，又缺 `faiss`，已补 `faiss-cpu`。
- 如果不显式设置 Java 11，`pyjnius` 可能错误使用 Java 8，报 class file version 不兼容。

## Smoke 数据与配置

新增最小 SFT 数据：

- `smoke_data/webshop_sft_smoke.json`
- `smoke_data/alfworld_sft_smoke.json`
- `smoke_data/intercode_sql_sft_smoke.json`

新增最小 SFT 配置：

- `train/sft_smoke/webshop.yaml`
- `train/sft_smoke/alfworld.yaml`
- `train/sft_smoke/intercode_sql.yaml`

数据注册位置：

- `LLaMA-Factory/data/dataset_info.json`

注意：

- 这些数据只用于 smoke，不是论文训练数据。
- 原论文配置仍保留在 `train/sft/*.yaml`。

## ETO SFT Trajectory 数据

Hugging Face 数据集：

- `agent-eto/eto-sft-trajectory`

当前只计划在 WebShop 上使用。已下载 WebShop 单文件到项目内 HF cache：

- `data/hf_cache/datasets--agent-eto--eto-sft-trajectory/snapshots/a8f745ef799a5201d2288b9c2a47e40a0ff70a4c/data/webshop_sft.json`

同时存在一次完整 snapshot 下载：

- `data/eto/eto-sft-trajectory/data/webshop_sft.json`
- `data/eto/eto-sft-trajectory/data/sciworld_sft.json`
- `data/eto/eto-sft-trajectory/data/alfworld_sft.json`

统计结果：

- WebShop JSON 顶层 1824 条。
- WebShop unique id 为 1823，重复 id 为 `4116`。
- README metadata 中 WebShop split 标为 1823 examples。
- 当前训练和评估优先只使用 WebShop 文件，不使用 ScienceWorld / ALFWorld。

## WebShop ETO Step-Level SFT

训练配置：

- `train/sft/webshop_qwen2_5_1p5b_lora_step.yaml`

数据注册：

- `LLaMA-Factory/data/dataset_info.json`
- 数据集名：`webshop_step_sft_2048_raw`
- 数据文件：`data/processed/gigpo/webshop_eto/webshop_step_sft_2048_raw.jsonl`

注意：

- 该数据是标准 `role/content` 消息格式，不是 ShareGPT 默认 `from/value`。
- registry 里必须显式设置 `tags.role_tag=role`、`tags.content_tag=content`、`tags.user_tag=user`、`tags.assistant_tag=assistant`。

四卡 DP 启动命令：

```bash
env CUDA_VISIBLE_DEVICES=1,2,3,4 \
  FORCE_TORCHRUN=1 \
  PATH=/mnt/dataset/fengshuwen/HPL/.venv/bin:$PATH \
  .venv/bin/llamafactory-cli train train/sft/webshop_qwen2_5_1p5b_lora_step.yaml
```

当前已验证：

- `torchrun --help` 正常。
- `liger_kernel` import 正常。
- LLaMA-Factory 日志出现 `Liger kernel has been applied to the model.`
- 4 卡 DDP world size 为 4。
- `per_device_train_batch_size=8`、`gradient_accumulation_steps=1`，总 batch size 为 32。
- 数据 5460 条可完成格式转换和 tokenizer。
- 训练主循环出现 tqdm、step loss、grad norm 和 learning rate。
- 完整训练 342 step 已完成，用时 `0:11:59.72`，最终 `train_loss=0.323`。
- 最终 adapter 保存到 `saves/qwen2_5_1p5b_lora_step_sft/adapter_model.safetensors`。

## 本地 vLLM Server

HPL 不内嵌 vLLM，评测和 rollout 通过 OpenAI-compatible API 调用外部服务。当前优先复用相邻项目的 vLLM 环境：

- vLLM 环境：`../post_train/.venv`
- vLLM 版本：`0.11.2`
- PyTorch：`2.9.0+cu128`

启动脚本：

```bash
scripts/start_posttrain_vllm.sh --gpus 1
```

如需加载当前 WebShop SFT LoRA：

```bash
scripts/start_posttrain_vllm.sh --gpus 1 --adapter saves/qwen2_5_1p5b_lora_step_sft
```

脚本会创建 tmux 会话 `hpl_vllm`，日志写入 `logs/vllm/`。HPL 侧调用：

```text
--api_base http://127.0.0.1:8000/v1
--api_key EMPTY
```

WebShop eval 当前同时兼容 `Action: ...` 和 `<action>...</action>` 输出格式。小样本 sanity check 建议用：

```bash
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:${LD_LIBRARY_PATH:-} \
PYTHONPATH=envs:envs/webshop:. \
.venv/bin/python evaluation.py \
  --exp_config webshop_local10 \
  --agent_config agent \
  --split test \
  --max_tasks 10 \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model_name qwen2.5-1.5b-instruct \
  --override
```

## 验证命令

### LLaMA-Factory

```bash
.venv/bin/llamafactory-cli version
```

结果：正常输出 LLaMA Factory 版本。

### CUDA

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

结果：CUDA 可用。

### WebShop reset/step

```bash
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:${LD_LIBRARY_PATH:-} \
PYTHONPATH=envs:envs/webshop:. \
.venv/bin/python - <<'PY'
from webshop.web_agent_site.envs import WebAgentTextEnv

env = WebAgentTextEnv(observation_mode="text", session="fixed")
obs = env.reset("fixed")
print("reset ok", str(obs)[:120])
obs, reward, done, info = env.step("search[phone case]")
print("step ok", reward, done, str(obs)[:120])
PY
```

结果：`reset ok` 与 `step ok` 均正常。

### SFT smoke

```bash
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/llamafactory-cli train train/sft_smoke/webshop.yaml

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/llamafactory-cli train train/sft_smoke/alfworld.yaml

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/llamafactory-cli train train/sft_smoke/intercode_sql.yaml
```

结果：

- 三个任务均完成 `max_steps=1`。
- 均保存 LoRA adapter。
- WebShop 已确认在 GPU2 上运行，日志显示 `device: cuda:0`，这是 `CUDA_VISIBLE_DEVICES=2` 后的逻辑编号。

## 未完成项

- 未下载 ALFWorld 完整 game files。
- 未下载 InterCode-SQL 完整数据。
- 未启动 InterCode-SQL Docker/MySQL 环境。
- 未下载 HPL 论文 expert trajectories。
- 未对 `evaluation.py` 接本地 vLLM/OpenAI-compatible server 做完整 smoke。

## 维护建议

- 不要把 smoke 数据与真实 HPL 训练数据混用。
- 不要直接覆盖 `train/sft/*.yaml`。
- 若后续需要迁移服务器，优先 rsync：
  - `AGENTS.md`
  - `docs/`
  - `train/sft_smoke/`
  - `smoke_data/`
  - `.venv` 可重建，但大型 WebShop 数据和索引建议直接同步。
