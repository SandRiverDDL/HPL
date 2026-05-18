---
name: autodl-project-migration
description: Use when migrating this project, datasets, HuggingFace assets, checkpoints, secrets, W&B setup, or Codex network access to an AutoDL server. Covers GitHub-based code sync, transfer-vs-download decisions, AutoDL directory layout, environment bootstrap, and when to use reverse SSH proxying.
---

# AutoDL 项目迁移

## Core Rule

先判断资源类型，再选择迁移方式：

- 代码：走 GitHub，不传 `.git` 压缩包。
- 小数据、本地加工数据、LoRA adapter：直接传输。
- 公开 HF dataset / base model：AutoDL 上重新下载到 HF cache，不上传本地 cache。
- optimizer / trainer state：只有要 resume 才传；只 eval 不传。
- token / 私钥 / W&B key：只放服务器本地环境，不写入仓库。

## When To Read References

- 需要配置 uv、HF cache、W&B、目录布局、BEACON/HPL 环境复用时：读 `references/environment-layout.md`。
- 需要让 AutoDL 访问 OpenAI/Codex 后端，或排查学术加速/代理/`ssh -R` 时：读 `references/network-proxy.md`。

## Preflight

迁移前先列清单，不要全量打包：

```bash
pwd
git status --short
git remote -v
du -sh . data saves outputs logs 2>/dev/null
find saves -maxdepth 3 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_config.json' -o -name 'config.json' -o -name 'tokenizer*' -o -name '*.safetensors' \) -printf '%p\t%s\n' 2>/dev/null
```

AutoDL 连接后至少检查：

```bash
hostname
df -h / /root/autodl-tmp
python3 -V
command -v uv || true
nvidia-smi || true
```

## GitHub Sync

AutoDL 上生成 GitHub 专用 key。只复制 `.pub` 公钥到 GitHub 账号，私钥留在 AutoDL 本机：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "autodl-$(hostname)-github-$(date +%Y%m%d)" -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub
```

把上一步输出的整行 `ssh-ed25519 ...` 加到 GitHub `SSH and GPG keys` 后，再测试和 clone：

```bash
ssh -i ~/.ssh/id_ed25519_github -T git@github.com
git clone git@github.com:<USER>/<REPO>.git /root/autodl-tmp/projects/<PROJECT_NAME>
cd /root/autodl-tmp/projects/<PROJECT_NAME>
git checkout -b autodl/$(hostname)-$(date +%Y%m%d)
```

如果 SSH 22 端口不可用，再测试 `ssh -T -p 443 git@ssh.github.com`；只有确认需要时才写 `~/.ssh/config`。

## Data And Checkpoint Transfer

小数据优先 `rsync`：

```bash
rsync -avhP -e "ssh -p <PORT>" data/processed/ root@<HOST>:/root/autodl-tmp/projects/<PROJECT_NAME>/data/processed/
```

小文件很多时用 tar 流：

```bash
cd <LOCAL_DATA_PARENT>
tar cf - <DATA_DIR> | ssh -p <PORT> root@<HOST> "mkdir -p /root/autodl-tmp/projects/<PROJECT_NAME>/data && cd /root/autodl-tmp/projects/<PROJECT_NAME>/data && tar xf -"
```

关键 checkpoint 先生成 manifest：

```bash
cd <CKPT_DIR>
find . -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
du -sh .
rsync -avhP -e "ssh -p <PORT>" <CKPT_DIR>/ root@<HOST>:/root/autodl-tmp/projects/<PROJECT_NAME>/saves/<CKPT_NAME>/
```

AutoDL 上校验：

```bash
cd /root/autodl-tmp/projects/<PROJECT_NAME>/saves/<CKPT_NAME>
sha256sum -c MANIFEST.sha256
```

## HF Download

不要上传本地 HF cache；公开 HF dataset / base model 在 AutoDL 上重新下载到 HF cache。无卡模式也可以下载模型 snapshot，但不要加载模型权重：

```bash
source /root/autodl-tmp/autodl_env.sh
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
hf download <ORG>/<DATASET> --repo-type dataset
hf download Qwen/Qwen2.5-1.5B-Instruct
```

如果直连慢或失败，再临时开启 AutoDL 学术加速：

```bash
source /etc/network_turbo
hf download <ORG>/<DATASET> --repo-type dataset
hf download Qwen/Qwen2.5-1.5B-Instruct
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

下载成功后，命令会输出 snapshot 路径。轻量验证只读 tokenizer/config：

```bash
MODEL_PATH=<HF_SNAPSHOT_PATH>
.venv/bin/python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
p = "<HF_SNAPSHOT_PATH>"
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
cfg = AutoConfig.from_pretrained(p, local_files_only=True)
print(tok.__class__.__name__, tok.vocab_size)
print(cfg.model_type, getattr(cfg, "hidden_size", None))
PY
```

训练/评测配置优先使用 HF repo id。若框架必须填写本地路径，再用 `hf download` 输出的 snapshot 路径；不要同时保留 HF cache 和项目内完整副本。

## BEACON / WebShop Runtime

BEACON 这类仓库可能把 WebShop 源码 vendored 进来，但主 lock/requirements 不一定覆盖 WebShop runtime。不要直接安装 WebShop 自带 `requirements.txt`，它可能降级 `torch/transformers/numpy`；只补运行缺口。

已验证的 AutoDL WebShop small/synthetic 最小清单：

```bash
uv pip install --python .venv/bin/python \
  gym==0.24.0 numpy==1.26.4 beautifulsoup4==4.11.1 cleantext==1.1.4 \
  Flask==2.1.2 Werkzeug==2.1.0 rank_bm25==0.2.2 thefuzz==0.19.0 \
  selenium==4.2.0 scikit-learn pyserini==0.17.0 spacy==3.7.2 \
  faiss-cpu requests_mock
uv pip install --python .venv/bin/python \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

`pyserini` 需要 Java 11。无系统 Java 时可装到数据盘，不动 `/usr`：

```bash
uv pip install --python .venv/bin/python install-jdk
.venv/bin/python - <<'PY'
import jdk
print(jdk.install("11", path="/root/autodl-tmp/cache/java"))
PY
export JAVA_HOME=/root/autodl-tmp/cache/java/jdk-11.0.31+11
export PATH="$JAVA_HOME/bin:$PATH"
```

small/synthetic WebShop 除 `indexes_1k` 外，通常还需要这些 data 文件：

```text
webshop/data/items_shuffle_1000.json
webshop/data/items_ins_v2_1000.json
webshop/data/items_human_ins.json
webshop/search_engine/indexes_1k
```

Ray / vLLM 环境里如果 dashboard agent 报：

```text
cannot import name 'OtelComponentTypeValues'
```

先检查 `opentelemetry-*` 版本。vLLM 0.8.5.post1 常见可用修复是：

```bash
uv pip install --python .venv/bin/python opentelemetry-exporter-prometheus==0.47b0
```

如果 `libgomp` 报 `Invalid value for environment variable OMP_NUM_THREADS`，检查是否被设成 `0`，改成正整数：

```bash
export OMP_NUM_THREADS=8
```

## Validation

迁移完成后至少检查：

```bash
cd /root/autodl-tmp/projects/<PROJECT_NAME>
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

## Best Practices

- 先迁移代码和最小样本，smoke 通过后再传大数据。
- 每次记录 commit hash、数据版本、checkpoint 路径、manifest、AutoDL 镜像/CUDA/Python 版本。
- 大文件不进 GitHub；排除 `data/`、`saves/`、`outputs/`、`wandb/`、`.env*`。
- 训练、评测、vLLM、RL 用 `tmux` / `screen`。
- `/root/autodl-tmp` 是训练热路径；`/root/autodl-fs` 适合备份/中转，不适合高 IO 训练。
- 不把私钥、HF token、W&B key、GitHub token 打印到日志或提交。

## References

- AutoDL 学术资源加速：https://www.autodl.com/docs/network_turbo/
- AutoDL 实例目录与数据盘：https://www.autodl.com/docs/env/
- AutoDL HuggingFace 缓存建议：https://www.autodl.com/docs/huggingface/
- AutoDL 上传数据与 tar 流：https://www.autodl.com/docs/scp/
- AutoDL 文件存储实践：https://www.autodl.com/docs/fs/
- uv package index：https://docs.astral.sh/uv/concepts/indexes/
