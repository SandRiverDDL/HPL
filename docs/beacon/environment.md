# BEACON 环境

## 路径

- BEACON 仓库：`/mnt/dataset/fengshuwen/agent-post-train/BEACON`
- BEACON Python：`/mnt/dataset/fengshuwen/agent-post-train/BEACON/.venv/bin/python`
- HPL 仓库：`/mnt/dataset/fengshuwen/HPL`
- HPL 编排脚本目录：`scripts/beacon/`

## Python 环境

BEACON 当前使用独立 `.venv`，不复用 HPL `.venv`。

已确认：

- `verl 0.3.1.dev`
- `ray 2.49.2`
- `vllm 0.8.5.post1`
- `torch 2.6.0+cu124`
- `gym==0.24.0`
- `gym_notices==0.1.0`
- `beautifulsoup4==4.11.1`
- `cleantext==1.1.4`
- `Flask==2.1.2`
- `Werkzeug==2.1.0`
- `pyserini==0.17.0`
- `faiss-cpu`
- `spacy==3.8.14`
- `en_core_web_sm==3.8.0`
- `thefuzz==0.19.0`
- `python-Levenshtein`
- `rank_bm25==0.2.2`
- `selenium==4.2.0`

说明：BEACON 当前代码直接 `import gym`，且 vendored WebShop 环境没有声明完整 `action_space/observation_space`。`gym==0.26.2` 会因环境检查失败阻塞 actor 创建，已降到 WebShop 原始 requirements 对应的 `gym==0.24.0`。

BEACON 使用独立 `.venv`，不能复用 HPL `.venv` 中已安装的 WebShop 依赖。WebShop 子项目 `requirements.txt` 会固定旧 `torch/transformers/numpy`，不要整文件安装；当前只补 BEACON WebShop import/runtime 必需依赖。

## Java

WebShop 的 `pyserini/anserini` 需要 Java 11。HPL 启动 BEACON 时必须显式设置：

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH:-}"
```

`scripts/beacon/run_webshop_lora_rl.sh` 已内置上述设置，避免 Ray WebShop actor 继承到 Java 8 后触发 `UnsupportedClassVersionError`。

## WebShop 数据与索引

BEACON vendored WebShop 目录不提交 full data/index。当前本机用软链复用 HPL 已准备好的 WebShop 资源：

```text
/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_package/webshop/webshop/data
  -> /mnt/dataset/fengshuwen/HPL/envs/webshop/data

/mnt/dataset/fengshuwen/agent-post-train/BEACON/agent_system/environments/env_package/webshop/webshop/search_engine/indexes
  -> /mnt/dataset/fengshuwen/HPL/envs/webshop/search_engine/indexes
```

这两个软链已加入 BEACON `.gitignore`，不要提交到 BEACON fork。

## Git Author

HPL 与 BEACON 两个仓库均已设置本地 git author：

```text
chy <290498984@qq.com>
```

这是仓库本地配置，不是全局 git 配置。

## 日志与监控

BEACON 使用 `verl.utils.tracking.Tracking`，支持：

- `console`
- `wandb`
- `swanlab`
- `mlflow`
- `tensorboard`
- `clearml`

当前 HPL 正式 RL 启动脚本默认：

```bash
trainer.logger=['console','wandb']
```

smoke test 可以临时覆盖为：

```bash
TRAINER_LOGGER="['console']"
```

正式训练、正式评测或长时间 RL 必须使用 W&B，并在启动前确认 project/name 配置。
