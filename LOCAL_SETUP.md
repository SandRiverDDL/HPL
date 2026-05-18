# HPL 本地运行说明

详细环境配置经过见 `docs/ENVIRONMENT_SETUP.md`；当前状态见 `docs/STATE.md`；项目结构见 `docs/ARCHITECTURE.md`。

本目录由 `uv` 管理环境：

```bash
cd /home/fengshuwen/data/HPL
source .venv/bin/activate
```

## 训练 smoke

已新增三个最小 SFT smoke 配置，均使用本机已有 `Qwen3-0.6B`，LoRA rank 4，`max_steps=1`：

```bash
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/llamafactory-cli train train/sft_smoke/webshop.yaml

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/llamafactory-cli train train/sft_smoke/alfworld.yaml

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/llamafactory-cli train train/sft_smoke/intercode_sql.yaml
```

原论文配置仍在 `train/sft/*.yaml`，默认是 `Qwen/Qwen2.5-7B-Instruct` 全参 + DeepSpeed ZeRO-3，需要按实际机器改模型路径、GPU 数量和数据。

## WebShop 环境

WebShop 源码位于：

```text
envs/webshop
```

数据和索引已下载并解压：

```text
envs/webshop/data
envs/webshop/search_index/indexes
envs/webshop/search_engine/indexes -> ../search_index/indexes
```

WebShop 运行需要固定 Java 11 和 Python path：

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:${LD_LIBRARY_PATH:-}
export PYTHONPATH=envs:envs/webshop:.
```

最小环境 smoke：

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

## 已知口径

- 当前 `.venv` 使用 Python 3.11，因为当前 LLaMA-Factory 要求 `>=3.11`。
- PyTorch 已固定到 `torch==2.7.1+cu126`，当前 driver 可识别 CUDA。
- WebShop 原始依赖固定了很旧的 `torch/transformers`，没有照单安装，否则会破坏 LLaMA-Factory 训练环境。
- `pyserini` 使用 HPL/WebShop 口径的 `0.17.0`；新版 `pyserini` 会要求更新 JVM 参数。
