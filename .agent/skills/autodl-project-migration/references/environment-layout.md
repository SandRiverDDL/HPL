# AutoDL Environment Layout

## Directory Layout

不要写死项目名。项目、cache 和运行产物分开：

```bash
export WORKSPACE_ROOT=/root/autodl-tmp/projects
export PROJECT_NAME=<PROJECT_NAME>
export PROJECT_ROOT="$WORKSPACE_ROOT/$PROJECT_NAME"
export CACHE_ROOT=/root/autodl-tmp/cache
mkdir -p "$WORKSPACE_ROOT" "$CACHE_ROOT"
```

数据、checkpoint 和 outputs 默认遵循项目相对路径：

```text
$PROJECT_ROOT/data
$PROJECT_ROOT/saves
$PROJECT_ROOT/outputs
```

跨项目共享的内容只放 cache 或只读公共资产：

```text
/root/autodl-tmp/cache/uv
/root/autodl-tmp/cache/pip
/root/autodl-tmp/cache/huggingface
```

公开 HF base model 优先下载到 HF cache，不上传，也不复制到每个项目。只有 merged model、非 HF 格式模型、本地不可复现 checkpoint，才放项目 `saves/` 或明确共享目录。

## Environment File

在 AutoDL 上创建本地环境文件，权限收紧；不要提交：

```bash
cat > /root/autodl-tmp/autodl_env.sh <<'EOF'
export WORKSPACE_ROOT=/root/autodl-tmp/projects
export CACHE_ROOT=/root/autodl-tmp/cache

export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers

export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_CACHE_DIR=/root/autodl-tmp/cache/wandb
export WANDB_PROJECT=<PROJECT_NAME>
export WANDB_MODE=online

export TOKENIZERS_PARALLELISM=false
export PATH=/root/.local/bin:$PATH
EOF
chmod 600 /root/autodl-tmp/autodl_env.sh
source /root/autodl-tmp/autodl_env.sh
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"
```

`WANDB_API_KEY` 和 `HF_TOKEN` 只放服务器本地环境，或用 `wandb login` / `huggingface-cli login`。

## uv Strategy

不要使用系统 Python 训练环境。系统环境只用于 bootstrap：

```bash
source /root/autodl-tmp/autodl_env.sh
uv venv .venv --python 3.10
.venv/bin/python -V
```

BEACON 当前本地参考组合：

```text
Python 3.10.14
torch 2.6.0+cu124
CUDA runtime 12.4
vLLM 0.8.5.post1
transformers 4.51.1
peft 0.19.1
ray 2.49.2
```

AutoDL driver/CUDA 显示 12.8 时，仍可优先安装 `cu124` PyTorch wheel；driver 通常向下兼容 CUDA runtime。不要混用系统 torch 2.8 和项目 vLLM/flash-attn/flashinfer。

## HPL / BEACON Split

空间紧张时：

- BEACON 是主环境，负责 RL、vLLM、WebShop、ckpt。
- HPL 如果只做 eval，不建独立 `.venv`，显式调用 `BEACON/.venv/bin/python`。
- 不默认做 `HPL/.venv -> BEACON/.venv` 软链接；只有 legacy 脚本硬编码 `.venv/bin/python` 时才作为兜底。
- HPL 若要重新跑 SFT，再单独建 LLaMA-Factory 环境。

50GB 数据盘下优先做减法：

- 每个项目不迁移本地 `.venv`。
- 共享 `UV_CACHE_DIR` / `HF_HOME`。
- eval-only 项目只安装评测链路依赖。
- LoRA adapter 留在各项目 `saves/`。
- 安装后可执行 `uv cache prune` 清掉不用的构建缓存。

## Storage Notes

- `/root/autodl-tmp`：数据盘，快，适合训练热路径。
- `/root/autodl-fs`：文件存储，普通用户默认总容量约 200GB，20GB 以下免费；适合备份/中转，不适合高 IO 训练。
- `/root/autodl-nas`：地区网盘，适合中转，免费空间较小。

## Codex Bootstrap

AutoDL 无卡模式可以先装 Codex 和 auth，也可以用 `hf download` 下载公开 HF 模型 snapshot 到 cache。不要在无卡/低内存模式加载模型权重或跑 GPU smoke；只做 tokenizer/config 级别验证。

如果服务器 npm 安装卡住，优先从本地已安装 Codex 打包传输：

```bash
# 本地/实验室服务器
tar -czf /tmp/codex-cli-0.130.0-global.tgz \
  -C /home/fengshuwen/.nvm/versions/node/v24.14.0/lib/node_modules \
  @openai/codex
scp -P <PORT> /tmp/codex-cli-0.130.0-global.tgz ~/.codex/auth.json \
  root@<HOST>:/root/autodl-tmp/
```

AutoDL 上安装 Node 二进制和 Codex wrapper：

```bash
source /root/autodl-tmp/autodl_env.sh
NODE_VERSION=v24.14.0
NODE_ROOT=/root/autodl-tmp/tools/node-${NODE_VERSION}-linux-x64
mkdir -p /root/autodl-tmp/tools /root/.codex /root/.local/bin

if [ ! -x "$NODE_ROOT/bin/node" ]; then
  cd /root/autodl-tmp/tools
  curl -L --fail --retry 3 -O https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz
  tar -xf node-${NODE_VERSION}-linux-x64.tar.xz
fi

cp /root/autodl-tmp/auth.json /root/.codex/auth.json
chmod 600 /root/.codex/auth.json

rm -rf /root/autodl-tmp/tools/codex-global/@openai
mkdir -p /root/autodl-tmp/tools/codex-global
tar -xzf /root/autodl-tmp/codex-cli-0.130.0-global.tgz -C /root/autodl-tmp/tools/codex-global

cat > /root/.local/bin/codex <<'EOF'
#!/usr/bin/env bash
export PATH=/root/autodl-tmp/tools/node-v24.14.0-linux-x64/bin:/root/.local/bin:$PATH
exec /root/autodl-tmp/tools/node-v24.14.0-linux-x64/bin/node /root/autodl-tmp/tools/codex-global/@openai/codex/bin/codex.js "$@"
EOF
chmod +x /root/.local/bin/codex

/root/.local/bin/codex --version
```

`auth.json` 是敏感文件，只能放服务器本地 `/root/.codex/auth.json`，不要提交，不要写入日志。

能否启动 Codex 的判断标准：

```bash
command -v codex
codex --version
test -s /root/.codex/auth.json
```

如果 AutoDL 本机不能直连 OpenAI/Codex 后端，再处理代理或 `ssh -R`；训练环境、HF 下载、GitHub pull/push 不依赖 Codex 反向代理。
