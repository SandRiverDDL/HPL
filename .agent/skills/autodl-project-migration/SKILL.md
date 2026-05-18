---
name: autodl-project-migration
description: Use when migrating this project, datasets, HuggingFace assets, checkpoints, secrets, or W&B setup to or from an AutoDL server. Covers GitHub-based code sync, AutoDL academic acceleration, HF mirror/cache placement, uv/pip mirrors, dataset transfer decisions, checkpoint manifests, and validation.
---

# AutoDL 项目迁移

## 目标

把本项目迁移到 AutoDL 服务器，并支持后续双向同步：

- 代码：优先走 GitHub，不传 `.git` 目录压缩包。
- 小数据 / 本地加工数据 / 关键 LoRA adapter：直接传输。
- 大型 HuggingFace 数据集或基础模型：在 AutoDL 上通过学术加速和 HF 镜像重新下载。
- 重要 checkpoint：只传真正需要恢复或评测的 checkpoint，不搬整份缓存。
- 密钥和 token：只放服务器本地环境，不写入仓库。

## 先做判断

迁移前先列清单，不要直接全量打包：

```bash
pwd
git status --short
git remote -v
du -sh . data saves outputs logs 2>/dev/null
find saves -maxdepth 3 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_config.json' -o -name 'config.json' -o -name 'tokenizer*' -o -name '*.safetensors' \) -printf '%p\t%s\n' 2>/dev/null
```

分类规则：

- `<5GB` 且不是公开 HF 资源：可以直接传。
- 小文件特别多：压成 tar 流或 tar 包后传，避免逐文件 scp。
- 公开 HF dataset/model：不要传本地缓存，在 AutoDL 上下载。
- base model：除非本地有不可复现修改，否则在 AutoDL 上重新下载。
- LoRA adapter：通常值得传，体积小且和实验结果直接相关。
- optimizer / trainer state：只有需要 resume 才传；只做 eval 时不要传。

## AutoDL 目录约定

优先使用数据盘，不要把大文件放系统盘。项目名不要写死，由当前迁移任务指定：

```bash
export WORKSPACE_ROOT=/root/autodl-tmp/projects
export PROJECT_NAME=<PROJECT_NAME>
export PROJECT_ROOT="$WORKSPACE_ROOT/$PROJECT_NAME"
export CACHE_ROOT=/root/autodl-tmp/cache
mkdir -p "$WORKSPACE_ROOT" "$CACHE_ROOT"
```

数据、模型和 checkpoint 默认遵循各项目自己的相对路径，例如 `$PROJECT_ROOT/data`、`$PROJECT_ROOT/saves`、`$PROJECT_ROOT/outputs`。不要强行把所有项目拆到全局 `datasets/models/checkpoints`，否则配置、脚本和文档容易失配。

适合跨项目共享的只有缓存和只读公共资产：

- `UV_CACHE_DIR=/root/autodl-tmp/cache/uv`
- `PIP_CACHE_DIR=/root/autodl-tmp/cache/pip`
- `HF_HOME=/root/autodl-tmp/cache/huggingface`
- 公开 HF base model 优先重新下载到 HF cache，不从本地上传，也不在每个项目目录复制一份。
- 只有本地微调后的完整 merged model 或非 HF 格式模型，才考虑放到项目 `saves/` 或明确的共享只读目录。

可把重要但低 IO 的备份放 `/root/autodl-fs`，训练时再拷贝到 `/root/autodl-tmp`。AutoDL 文档中 `/root/autodl-fs` 是文件存储，可靠性更好但 IO 一般；`/root/autodl-nas` 是地区网盘，免费空间较小，适合中转。

## 学术加速

AutoDL 内置加速适合 GitHub / HuggingFace 下载。只在下载阶段开启，用完关闭：

```bash
source /etc/network_turbo
env | grep -i proxy

# 下载结束后关闭，避免影响正常网络
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

如果是 Notebook，需要把 `source /etc/network_turbo` 得到的 proxy 环境变量写入 Python `os.environ`。

## 服务器环境变量

在 AutoDL 上创建本地环境文件，权限收紧；不要提交到 GitHub：

```bash
cat > /root/autodl-tmp/autodl_env.sh <<'EOF'
# Python 包源：uv 优先使用 UV_DEFAULT_INDEX，pip 使用 PIP_INDEX_URL
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip

# HuggingFace：缓存放数据盘，公开资源优先走镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers

# W&B：正式训练/评测/RL 需要显式配置
export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_CACHE_DIR=/root/autodl-tmp/cache/wandb
export WANDB_PROJECT=<PROJECT_NAME>
export WANDB_MODE=online

export TOKENIZERS_PARALLELISM=false
EOF
chmod 600 /root/autodl-tmp/autodl_env.sh
source /root/autodl-tmp/autodl_env.sh
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"
```

`WANDB_API_KEY` 和 `HF_TOKEN` 不要写进仓库；可用交互式 `wandb login` / `huggingface-cli login`，或只写入服务器本地的 `autodl_env.sh`。

## GitHub 双向同步

本地先确认代码已 commit 并 push：

```bash
git status --short
git branch --show-current
git push
```

AutoDL 服务器上生成 GitHub 专用公钥，只把 `.pub` 内容添加到 GitHub 账号：

```bash
ssh-keygen -t ed25519 -C "autodl-$(hostname)-github-$(date +%Y%m%d)" -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub
```

用户把公钥加入 GitHub 后，在 AutoDL 测试并克隆：

```bash
ssh -T git@github.com
source /etc/network_turbo
git clone git@github.com:<USER>/<REPO>.git "$PROJECT_ROOT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd "$PROJECT_ROOT"
git checkout -b autodl/$(hostname)-$(date +%Y%m%d)
```

如果 GitHub SSH 22 端口不可用，可先测试 443：

```bash
ssh -T -p 443 git@ssh.github.com
```

只有确认需要时，才在服务器上写 `~/.ssh/config` 指向 `ssh.github.com:443`。

## 代码环境恢复

不要迁移本机 `.venv`。在 AutoDL 重新建环境：

```bash
cd "$PROJECT_ROOT"
source /root/autodl-tmp/autodl_env.sh
uv venv .venv --python 3.10
.venv/bin/python -V
uv pip install -r requirements.txt --default-index "$UV_DEFAULT_INDEX"
```

如果项目使用额外 CUDA / torch / flash-attn / vLLM 依赖，先读项目文档和当前镜像 CUDA 版本，再安装匹配 wheel；不要盲目升级整套环境。

50GB 数据盘下同时放两个项目时，优先做减法：

- 不迁移本地 `.venv`，每个项目在 AutoDL 重新建。
- 共享 `UV_CACHE_DIR` / `HF_HOME`，避免每个项目重复下载 wheel 和 HF cache。
- eval-only 项目只安装评测链路依赖，不安装训练、flash-attn、vLLM、WebShop 全套大依赖，除非评测实际需要。
- 如果两个项目都依赖同一个公开 HF base model，只保留一份 HF cache；LoRA adapter 留在各项目 `saves/`。
- 如果 HPL 只是为了 eval，而 BEACON 已有环境和 WebShop 小索引，优先把 eval 入口迁到 BEACON 或写轻量 wrapper，避免维护两套重环境。
- 安装后执行 `uv cache prune` 可以清掉不再需要的构建缓存；不要删除正在被环境引用的 `.venv` 内容。

## 数据迁移策略

### 小数据或本地加工数据

从本地机器传到 AutoDL，优先 `rsync`：

```bash
rsync -avhP -e "ssh -p <PORT>" data/processed/ root@<HOST>:/root/autodl-tmp/projects/<PROJECT_NAME>/data/processed/
```

如果小文件很多，用 tar 流：

```bash
cd <LOCAL_DATA_PARENT>
tar cf - <DATA_DIR> | ssh -p <PORT> root@<HOST> "mkdir -p /root/autodl-tmp/projects/<PROJECT_NAME>/data && cd /root/autodl-tmp/projects/<PROJECT_NAME>/data && tar xf -"
```

AutoDL 文档中的 scp 口径可作为兜底：

```bash
scp -rP <PORT> <LOCAL_PATH> root@<HOST>:/root/autodl-tmp/
```

### 大型 HF dataset / model

不要上传本地 HF cache；服务器网速通常比本地上传更快，公开 HF 数据集和模型优先在 AutoDL 上重新下载到 `HF_HOME`：

```bash
source /root/autodl-tmp/autodl_env.sh
source /etc/network_turbo

huggingface-cli download <ORG>/<DATASET> \
  --repo-type dataset

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

训练/评测配置里优先使用 HF repo id，例如 `Qwen/Qwen2.5-1.5B-Instruct`。如果框架必须填写本地路径，再用 `huggingface-cli download` 输出的 snapshot 路径，或显式下载到项目相对路径；不要同时保留 HF cache 和项目内完整副本。

私有 HF 资源需要先 `huggingface-cli login` 或设置服务器本地 `HF_TOKEN`。

## Checkpoint 迁移

只传关键 checkpoint。LoRA adapter 通常至少包含：

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json / tokenizer.model / tokenizer_config.json / special_tokens_map.json
training_args 或 trainer_state 可选
```

传输前生成 manifest：

```bash
cd <CKPT_DIR>
find . -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
du -sh .
```

传输：

```bash
rsync -avhP -e "ssh -p <PORT>" <CKPT_DIR>/ root@<HOST>:/root/autodl-tmp/projects/<PROJECT_NAME>/saves/<CKPT_NAME>/
```

到 AutoDL 后校验：

```bash
cd "$PROJECT_ROOT/saves/<CKPT_NAME>"
sha256sum -c MANIFEST.sha256
```

## 验证清单

迁移完成后至少检查：

```bash
cd "$PROJECT_ROOT"
source /root/autodl-tmp/autodl_env.sh
git rev-parse HEAD
git status --short
.venv/bin/python - <<'PY'
import os
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("HF_HOME:", os.environ.get("HF_HOME"))
print("WANDB_PROJECT:", os.environ.get("WANDB_PROJECT"))
PY
```

如果涉及 WebShop / RL / LLaMA-Factory，继续按项目文档跑最小 smoke；不要直接启动长训练。

## 最佳实践

- 先迁移代码和一个最小数据样本，smoke 通过后再传大数据。
- 每次迁移都记录：commit hash、数据版本、checkpoint 路径、manifest、AutoDL 镜像/CUDA/Python 版本。
- 大文件不要进 GitHub；用 `.gitignore` 排除 `data/`、`saves/`、`outputs/`、`wandb/`、`.env*`。
- 训练和评测用 `tmux`，避免 SSH 断开导致任务退出。
- AutoDL 本地数据盘不是可靠备份；重要结果及时同步到 GitHub release、HF private repo、对象存储、本地机器或 `/root/autodl-fs`。
- 学术加速不保证稳定，下载失败时先重试单个资源；不要把失败归因到代码。
- 不要把私钥、HF token、W&B key、GitHub token 打印到日志或写入提交。
- 双向同步只同步代码和轻量配置；实验产物通过 manifest 和明确路径单独同步。

## 参考

- AutoDL 学术资源加速：https://www.autodl.com/docs/network_turbo/
- AutoDL 实例目录与数据盘：https://www.autodl.com/docs/env/
- AutoDL HuggingFace 缓存建议：https://www.autodl.com/docs/huggingface/
- AutoDL 上传数据与 tar 流：https://www.autodl.com/docs/scp/
- AutoDL 文件存储实践：https://www.autodl.com/docs/fs/
- uv package index / environment variables：https://docs.astral.sh/uv/concepts/indexes/
