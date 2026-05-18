# State

本文件是 HPL 当前状态 dashboard，只保留当前可执行口径、已验证入口、主要风险和下一步。环境配置细节见 `docs/ENVIRONMENT_SETUP.md`。

BEACON/verl-agent 相关长期记录见 `docs/beacon/`，本文件只保留当前状态摘要。

## 当前定位

- 目标：在本机复现/学习 HPL（Hierarchical Preference Learning）项目，先保证 WebShop 与 LLaMA-Factory 训练入口可跑通。
- 仓库位置：`/home/fengshuwen/data/HPL`
- 真实挂载盘：`/mnt/dataset`，当前剩余约 `21T`
- HPL upstream：`https://github.com/PatrickG1014/HPL`
- WebShop upstream：`https://github.com/princeton-nlp/WebShop`
- 训练框架：本地 clone 的 `LLaMA-Factory`

## 当前有效环境

- Python：`.venv`，由 `uv` 创建。
- Python 版本：`3.11`，因为当前 LLaMA-Factory 要求 `>=3.11`。
- PyTorch：`torch==2.7.1+cu126`。
- Liger kernel：已安装并在 Qwen2.5-1.5B WebShop LoRA SFT 中验证可应用。
- CUDA 状态：已验证可识别 RTX 4090。
- Java：WebShop/pyserini 使用 `/usr/lib/jvm/java-11-openjdk-amd64`。
- WebShop 数据：`envs/webshop/data`
- WebShop 索引：`envs/webshop/search_index/indexes`
- WebShop 代码所需索引软链：`envs/webshop/search_engine/indexes -> ../search_index/indexes`
- ETO WebShop SFT trajectory：`data/hf_cache/datasets--agent-eto--eto-sft-trajectory/snapshots/a8f745ef799a5201d2288b9c2a47e40a0ff70a4c/data/webshop_sft.json`
- ETO WebShop step-level SFT：`data/processed/gigpo/webshop_eto/webshop_step_sft_2048_raw.jsonl`
- WebShop step-level SFT 训练配置：`train/sft/webshop_qwen2_5_1p5b_lora_step.yaml`

## 已验证入口

### 基础 import

- `torch` import 正常。
- `llamafactory-cli version` 正常。
- `deepspeed` import 正常。
- `alfworld` import 正常。
- `intercode` import 正常。
- WebShop import 正常，但需要 Java 与 PYTHONPATH 环境变量。

### WebShop smoke

已验证：

- `WebAgentTextEnv(observation_mode="text")` 可初始化。
- `reset("fixed")` 可加载任务。
- `step("search[phone case]")` 可调用搜索索引并返回 observation/reward/done。

### 训练 smoke

新增三份本地 smoke 配置，均为 Qwen3-0.6B + LoRA rank4 + `max_steps=1`：

- `train/sft_smoke/webshop.yaml`
- `train/sft_smoke/alfworld.yaml`
- `train/sft_smoke/intercode_sql.yaml`

已验证：

- 三个 smoke 都能加载数据、完成 1 step 训练并保存 adapter。
- WebShop smoke 已在 GPU2 上重新跑通，确认 CUDA 训练链路可用。
- WebShop ETO step-level LoRA SFT 已完成 4 卡 DDP 训练：`CUDA_VISIBLE_DEVICES=1,2,3,4`，`per_device_train_batch_size=8`，`gradient_accumulation_steps=1`，总 batch size 为 32。
- 上述训练日志已确认：Liger kernel applied、数据 5460 条可 tokenization、训练主循环出现 tqdm 和逐步 loss；342 step 用时 `0:11:59.72`，最终 `train_loss=0.323`。
- 最终 adapter：`saves/qwen2_5_1p5b_lora_step_sft/adapter_model.safetensors`。
- BEACON 本地 RL 环境：`/mnt/dataset/fengshuwen/agent-post-train/BEACON/.venv`，已验证可 `py_compile` 当前 `verl/workers/fsdp_workers.py`。
- BEACON 本地 patch：`actor_rollout_ref.model.path` 继续表示完整 HF base/merged model；新增 `actor_rollout_ref.model.lora_adapter_path` 用于加载已有 PEFT LoRA adapter，并从 `adapter_config.json` 自动推断 rank。
- BEACON `.venv` 已补 WebShop runtime 依赖：`gym`、`beautifulsoup4`、`pyserini==0.17.0`、`faiss-cpu`、`spacy/en_core_web_sm` 等；WebShop import smoke 已通过。若后续遇到 NumPy 2 兼容问题，再考虑 gymnasium patch。

## 当前目录占用

- `.venv`：约 `9.0G`
- `envs`：约 `13G`
- `envs/webshop/data`：约 `5.3G`
- `envs/webshop/search_index`：约 `3.4G`
- `saves/smoke`：约 `124M`

## 当前风险

- 原论文 `train/sft/*.yaml` 默认是 `Qwen/Qwen2.5-7B-Instruct` 全参 + DeepSpeed ZeRO-3，不适合直接在单卡上运行。
- HPL 原仓库没有附带 SFT 数据注册和 expert trajectories；当前只补了最小 smoke 数据，不能代表论文真实数据。
- WebShop 原始依赖非常旧，固定了旧 `torch/transformers`；HPL 与 BEACON 都不要照单安装，否则会破坏现有训练环境，只补 runtime 缺口。
- `pyserini` 必须使用与 Java 11 兼容的版本；当前固定为 `pyserini==0.17.0`。
- WebShop 运行依赖 `JAVA_HOME`、`LD_LIBRARY_PATH`、`PYTHONPATH`，不要只激活 `.venv` 就直接跑；BEACON 启动脚本已显式设置 Java 11。
- 当前只完成 WebShop 数据和索引；ALFWorld 游戏数据、InterCode-SQL 数据与 Docker/MySQL 环境尚未按论文 setup 完整下载/启动。
- ETO SFT trajectory 当前只打算使用 WebShop；不要把 ScienceWorld / ALFWorld 的下载文件误认为本项目已准备对应环境。
- ETO WebShop step-level SFT 当前是 raw observation、完整 history、`reward >= 0.7`、Qwen2.5-1.5B-Instruct chat template 计数后 `max_length <= 2048` 口径；原始 `human/gpt` 已转成 `user/assistant`，不包含 observation cleaning。
- `webshop_step_sft_2048_raw` 在 LLaMA-Factory registry 中必须显式声明 ShareGPT tags：`role/content/user/assistant`；否则会按默认 `from/value/human/gpt` 解析并报 `KeyError: 'from'`。
- 本机 shell 直接调用 `.venv/bin/llamafactory-cli` 时，分布式子进程需要能在 `PATH` 中找到 `torchrun`；启动四卡训练时显式加入 `PATH=/mnt/dataset/fengshuwen/HPL/.venv/bin:$PATH`。
- WebShop eval 已兼容两种动作格式：HPL 原生 `Action: ...` 和 GiGPO/SPEAR 风格 `<action>...</action>`。
- `evaluation.py` 支持 `--max_tasks N`，用于固定跑前 N 个未完成任务；不要再依赖当前 `--debug`，它历史实现实际只截断到 5 个任务。
- `envs/__init__.py` 与 `tasks/__init__.py` 对 InterCode import 做了可选导入，避免只跑 WebShop 时被缺失的 `intercode.envs` 阻塞。
- 当前本地 WebShop 环境只加载到 13 个 human goals；原 `data/webshop/test_indices.json` 的大 id 与本地 goals 不匹配，会触发 `IndexError: list index out of range`。
- 为 10 题 sanity check 新增 `configs/task/webshop_local10.json` 和 `data/webshop_local10/test_indices.json`，只使用本地可用 session id `0..9`，不代表论文 test split。
- `evaluation.py` 的 WebShop 指标分支已改为 `exp_config.startswith("webshop")`，避免 `webshop_local10` 被当作非 WebShop 任务导致 success rate 口径错误。
- WebShop 源码默认使用 preview 商品文件 `items_shuffle_1000.json` 和 `items_ins_v2_1000.json`；这会改变搜索候选集合，导致 reward / success 不可与正式 WebShop split 比较。
- `evaluation.py` 已支持从 task config 传入 WebShop `file_path`、`attr_path`、`num_products` 等参数；`configs/task/webshop_full10.json` 使用全量 `items_shuffle.json`、`items_ins_v2.json` 和 full `search_engine/indexes`。
- 全量 WebShop 前 10 个 `data/webshop/test_indices.json` 测试 id 已跑通：base `avg_reward=0.1950`、`success=1/10`；当前 LoRA `avg_reward=0.6697`、`success=2/10`。WebShop success 口径按 `reward == 1`，不要直接用输出 JSON 的 `state.success` 字段。
- BEACON 当前支持两种 RL 初始化口径：整模型直接传 `actor_rollout_ref.model.path=/path/to/full_hf_model`；LoRA 冷启动传 `actor_rollout_ref.model.path=/path/to/base_model actor_rollout_ref.model.lora_adapter_path=/path/to/lora_adapter`。详见 `docs/beacon/patches.md`。
- BEACON WebShop 本地 patch 支持 `env.webshop.train_start_idx`、`env.webshop.val_end_idx`、`env.webshop.exclude_goal_indices_path`；HPL 启动脚本默认 full human goals，并排除 `data/webshop/test_indices.json`，避免 RL 训练采样污染 HPL test split。详见 `docs/beacon/webshop-split.md`。
- `scripts/beacon/run_webshop_lora_rl.sh` 是 HPL 控制 BEACON 的 WebShop LoRA RL 编排入口；默认正式运行启用 W&B，smoke 可用 `TRAINER_LOGGER="['console']"` 覆盖；默认参数、训练量估计和待验证项见 `docs/beacon/rl-runbook.md`。

## 下一步

- 若要跑真实 SFT：先准备 HPL 论文所需 expert trajectory 数据，并注册到 LLaMA-Factory dataset registry。
- 若要跑 WebShop 正式评测：使用全量商品、属性和索引；`webshop_local10` 只用于 preview sanity check，`webshop_full10` 只用于全量小样本 sanity check。
- 若要复现完整 HPL：分阶段补齐 trajectory-level、step-level、group-level 数据构造链路，不要直接全量跑。
- 若要迁移到另一台服务器：优先同步 `AGENTS.md`、`docs/`、`train/sft_smoke/`、`smoke_data/` 和本说明；大型 WebShop 数据与索引按需 rsync。

## 本地脚本

- `tools/data/convert_eto_webshop_step_sft.py`：将 ETO WebShop trajectory 转成 step-level SFT 数据。
- `scripts/train_model.py`：手动 GPU 占卡脚本，支持显存和 GPU util 轻微波动；该脚本会占用 GPU，禁止在未确认 GPU 空闲和用户意图时自动运行。
- `scripts/start_posttrain_vllm.sh`：复用 `../post_train/.venv` 中的 vLLM，在 tmux 中启动 OpenAI-compatible server，供 HPL evaluation / rollout 调用。
- `scripts/beacon/run_webshop_lora_rl.sh`：在 HPL 侧编排相邻 BEACON 仓库的 WebShop LoRA RL。

## Documentation Map

- 稳定规则：`AGENTS.md`
- 架构分层：`docs/ARCHITECTURE.md`
- 当前状态：`docs/STATE.md`
- 环境配置记录：`docs/ENVIRONMENT_SETUP.md`
- BEACON 集成记录：`docs/beacon/README.md`
- 原始 HPL 说明：`README.md`
- 本地快速说明：`LOCAL_SETUP.md`
