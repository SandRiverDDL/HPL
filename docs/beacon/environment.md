# BEACON 环境

## 路径

- BEACON 仓库：`/mnt/dataset/fengshuwen/agent-post-train/BEACON`
- BEACON fork origin：`git@github.com:SandRiverDDL/BEACON.git`
- BEACON upstream：`https://github.com/ZJU-REAL/BEACON.git`
- BEACON Python：`/mnt/dataset/fengshuwen/agent-post-train/BEACON/.venv/bin/python`
- HPL 仓库：`/mnt/dataset/fengshuwen/HPL`
- HPL fork origin：`git@github.com:SandRiverDDL/HPL.git`
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

## AutoDL A800 环境

AutoDL 远程项目路径：

```text
BEACON: /root/autodl-tmp/projects/BEACON
HPL:    /root/autodl-tmp/projects/HPL
cache:  /root/autodl-tmp/cache
```

已验证组合：

- GPU：A800-SXM4-80GB 单卡。
- Python：BEACON `.venv` Python 3.10。
- `torch==2.6.0+cu124`
- `vllm==0.8.5.post1`
- `ray==2.49.2`
- `transformers==4.51.1`
- `peft==0.19.1`
- `flash-attn==2.7.4.post1`
- `numpy==1.26.4`
- `gym==0.24.0`
- `pyserini==0.17.0`
- `spacy==3.7.2`
- `en_core_web_sm==3.7.1`

AutoDL 不要整装 WebShop `requirements.txt`。它会拉旧训练依赖；只补 WebShop runtime 缺口。已补过的关键缺口包括：

```text
beautifulsoup4 cleantext Flask Werkzeug rank_bm25 thefuzz selenium scikit-learn
pyserini spacy en_core_web_sm faiss-cpu requests_mock
```

`pyserini` 需要 Java 11；远程当前使用用户态 JDK，不改系统目录：

```text
JAVA_HOME=/root/autodl-tmp/cache/java/jdk-11.0.31+11
```

远程 `.env.autodl` 已追加：

```bash
export JAVA_HOME=/root/autodl-tmp/cache/java/jdk-11.0.31+11
export PATH="$JAVA_HOME/bin:$PATH"
export OMP_NUM_THREADS=8
```

Codex 命令路径和反向 SOCKS 代理默认写入远程 `/root/.bashrc`。HPL 提供两个脚本管理这段配置：

```bash
scripts/autodl/configure_codex_proxy.sh        # 默认端口 17891
scripts/autodl/remove_codex_proxy.sh
```

`configure_codex_proxy.sh` 会写入：

```bash
export PATH=/root/.local/bin:/root/autodl-tmp/tools/node-v24.14.0-linux-x64/bin:$PATH
export ALL_PROXY=socks5h://127.0.0.1:17891
export HTTPS_PROXY=socks5h://127.0.0.1:17891
export HTTP_PROXY=socks5h://127.0.0.1:17891
```

如果后续要用 AutoDL 直连或学术加速下载大文件，先执行 `scripts/autodl/remove_codex_proxy.sh` 或在当前 shell 临时 `unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy`。

AutoDL small/synthetic WebShop 已传入的必要小文件：

```text
agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json
agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json
agent_system/environments/env_package/webshop/webshop/data/items_human_ins.json
agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k
```

已定位并修复的 AutoDL 专有坑：

- Ray dashboard agent 因 `opentelemetry-exporter-prometheus==0.62b1` 与 vLLM 拉下来的 `opentelemetry-*==1.26/0.47b0` 不匹配崩溃；已对齐为 `opentelemetry-exporter-prometheus==0.47b0`。
- `OMP_NUM_THREADS=0` 会触发 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`；已改成 `8`。
- `spacy download en_core_web_sm` 在该环境会拼出错误 GitHub URL；改为直接安装 `en_core_web_sm-3.7.1` wheel。

已完成 smoke：

```text
命令：LOGGER="['console']" TOTAL_TRAINING_STEPS=1 TRAIN_DATA_SIZE=1 VAL_DATA_SIZE=1 GROUP_SIZE=1 scripts/autodl/run_webshop_smoke.sh
结果：Training Progress 1/1 完成
WebShop train goal count: 6410
WebShop val goal count: 500
episode/webshop_task_score (not success_rate): 0.714
episode/success_rate: 0.000
timing_s/step: 20.601
perf/max_memory_allocated_gb: 38.159
```

注意：SFT adapter 的 `adapter_config.json` 仍记录本地 base path `/home/fengshuwen/...`，PEFT 保存时会打印 warning；当前训练通过脚本显式传入远程 base model path，smoke 不受阻塞。

## Git Author

HPL 与 BEACON 两个仓库均已设置本地 git author：

```text
chy <290498984@qq.com>
```

这是仓库本地配置，不是全局 git 配置。

当前 GitHub SSH key 使用标准路径：

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
SHA256:PSkRb7o/rrSrl16DPjz+m9wHVujBHciwoPPSkvlaCYI
```

已验证 `ssh -T git@github.com` 认证到 `SandRiverDDL`。

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
