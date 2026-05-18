# State

本文件是 HPL 当前状态 dashboard，只保留当前可执行口径、已验证入口、主要风险和下一步。环境配置细节见 `docs/ENVIRONMENT_SETUP.md`。

BEACON/verl-agent 相关长期记录见 `docs/beacon/`，本文件只保留当前状态摘要。

## 当前定位

- 目标：在本机复现/学习 HPL（Hierarchical Preference Learning）项目，先保证 WebShop 与 LLaMA-Factory 训练入口可跑通。
- 仓库位置：`/mnt/dataset/fengshuwen/HPL`
- 真实挂载盘：`/mnt/dataset`，当前剩余约 `21T`
- HPL fork origin：`git@github.com:SandRiverDDL/HPL.git`
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
- BEACON fork origin：`git@github.com:SandRiverDDL/BEACON.git`；BEACON upstream：`https://github.com/ZJU-REAL/BEACON.git`。
- BEACON 本地 patch：`actor_rollout_ref.model.path` 继续表示完整 HF base/merged model；新增 `actor_rollout_ref.model.lora_adapter_path` 用于加载已有 PEFT LoRA adapter，并从 `adapter_config.json` 自动推断 rank。
- BEACON `.venv` 已补 WebShop runtime 依赖：`gym`、`beautifulsoup4`、`pyserini==0.17.0`、`faiss-cpu`、`spacy/en_core_web_sm` 等；WebShop import smoke 已通过。若后续遇到 NumPy 2 兼容问题，再考虑 gymnasium patch。
- HPL 最新已推送到 fork commit `0800045`；BEACON 最新已推送到 fork commit `4a09564`。

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
- 当前 BEACON/SPEAR 对齐主线使用 small/synthetic WebShop：`items_shuffle_1000.json`、`items_ins_v2_1000.json`、`human_goals=False`。固定 split：test `0..199`，dev/validate `200..249`，train `500..6909`，`250..499` 暂不使用。该口径不可与 full human WebShop 直接混比。
- WebShop 源码默认使用 preview 商品文件 `items_shuffle_1000.json` 和 `items_ins_v2_1000.json`；这会改变搜索候选集合，导致 reward / success 不可与 full human WebShop split 比较。
- `evaluation.py` 已支持从 task config 传入 WebShop `file_path`、`attr_path`、`num_products` 等参数；`configs/task/webshop_full10.json` 使用全量 `items_shuffle.json`、`items_ins_v2.json` 和 full `search_engine/indexes`。
- 全量 WebShop 前 10 个 `data/webshop/test_indices.json` 测试 id 已跑通：base `avg_reward=0.1950`、`success=1/10`；当前 LoRA `avg_reward=0.6697`、`success=2/10`。WebShop success 口径按 `reward == 1`，不要直接用输出 JSON 的 `state.success` 字段。
- BEACON 当前支持两种 RL 初始化口径：整模型直接传 `actor_rollout_ref.model.path=/path/to/full_hf_model`；LoRA 冷启动传 `actor_rollout_ref.model.path=/path/to/base_model actor_rollout_ref.model.lora_adapter_path=/path/to/lora_adapter`。详见 `docs/beacon/patches.md`。
- BEACON WebShop 本地 patch 支持 `env.webshop.train_start_idx`、`env.webshop.val_start_idx`、`env.webshop.val_end_idx`、`env.webshop.exclude_goal_indices_path`；HPL 启动脚本默认 small/synthetic dev50 validate，避免再次创建大量 full WebShop actor。详见 `docs/beacon/webshop-split.md`。
- `scripts/beacon/run_webshop_lora_rl.sh` 是 HPL 控制 BEACON 的 WebShop LoRA RL 编排入口；三卡 smoke 已确认可完成 1 个 RL update、validate 和 checkpoint 保存。当前活跃 25 step milestone run 运行在 tmux `hpl_beacon_rl_to25_gpu125`，日志为 `logs/beacon_webshop_small_synth_b6n4_to25_seed1025.log`，使用 `CUDA_VISIBLE_DEVICES=1,2,5`、`ENV_SEED=1025`、`TRAIN_BATCH_SIZE=6`、`ROLLOUT_N=4`、`PPO_MINI_BATCH_SIZE=24`、`PPO_MICRO_BATCH_SIZE_PER_GPU=4`、`TOTAL_TRAINING_STEPS=25`、`SAVE_FREQ=25`、`TEST_FREQ=0`。该 run 用 `trainer.resume_mode=auto` 从同名 `global_step_3` calibration checkpoint 继续到 step 25。此前 `hpl_lora_migpo_qwen2_5_1p5b_webshop_small_50step` 误用 `ROLLOUT_N=2`，不作为正式结果。默认参数、训练量估计和待验证项见 `docs/beacon/rl-runbook.md`。
- 最近一次 BEACON smoke 与正式 n=4 run 已定位并修复已有 LoRA adapter 加载触发的 PEFT/Transformers tensor-parallel 兼容问题、MiGPO `milestone_achieved` numpy 数组 truth-value 判断问题，以及 LoRA adapter 保存时 `task_type/peft_type` 字符串兼容问题。下一步关注 `SAVE_FREQ=25` 的 checkpoint/独立 LoRA adapter 保存和训练后 dev/test 评测。
- BEACON 没有独立轻量 WebShop eval runner；`trainer.val_only=True` 仍会启动 Ray/vLLM/PPO trainer 体系。三组模型横向评测优先走 HPL `evaluation.py`。HPL small/synthetic 评测已对齐小索引：`configs/task/webshop_small_synth_*.json` 设置 `num_products=1000`，使用 `envs/webshop/search_engine/indexes_1k`，该索引已验证为 1000 docs。

## 下一步

- 若要跑真实 SFT：先准备 HPL 论文所需 expert trajectory 数据，并注册到 LLaMA-Factory dataset registry。
- 若要跑 BEACON small/synthetic 横向评测：使用 HPL `evaluation.py` + `configs/task/webshop_small_synth_test200.json`，评测 base、SFT LoRA、RL LoRA；该 config 已使用 `num_products=1000/indexes_1k` 真 1K 小索引口径。
- 若要跑 full human WebShop 正式评测：使用全量商品、属性和索引；`webshop_local10` 只用于 preview sanity check，`webshop_full10` 只用于全量小样本 sanity check。
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
