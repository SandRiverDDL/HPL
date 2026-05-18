# BEACON 集成文档

本目录记录 HPL 侧如何控制相邻 BEACON 仓库进行 WebShop agentic RL。BEACON 是外部执行引擎，HPL 当前只保留轻量编排脚本、测试 split 约束、本地 patch 说明和运行记录。

## 文档分工

- `environment.md`：BEACON 仓库、`.venv`、依赖和当前环境状态。
- `patches.md`：对 BEACON 源码做过的本地 patch、原因、影响和回滚口径。
- `webshop-split.md`：WebShop train/val/test 任务 id 采样、防泄漏规则和验证口径。
- `rl-runbook.md`：HPL 控制 BEACON 训练的启动脚本、默认超参、训练量估计和待验证项。

## 当前入口

HPL 编排脚本：

```bash
scripts/beacon/run_webshop_lora_rl.sh
```

BEACON 训练入口：

```bash
cd /mnt/dataset/fengshuwen/agent-post-train/BEACON
.venv/bin/python -m verl.trainer.main_ppo ...
```

## 当前状态摘要

- BEACON 提交 `7146bfa` 已支持已有 LoRA adapter 初始化和 WebShop split 过滤。
- HPL 提交 `519d300` 新增 WebShop LoRA RL 启动脚本。
- BEACON `.venv` 已补 `gym==0.26.2`，`import gym` 可通过。
- 当前 WebShop RL 还未跑通完整 1 个 update step；上一次失败点是缺少 `gym`，不是 NCCL/通信问题。
