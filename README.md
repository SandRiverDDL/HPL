# Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents

This repository contains the data and code for the paper "Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents" (ICLR 2026). [[paper]](https://openreview.net/forum?id=s8usvGHYlk)

## ⚙️ Installation

```bash
bash setup.sh
```

This setup script includes the following steps:

- Install Python dependencies for agent training and evaluation
- Donwload data and search engine indices for WebShop
- Download game files for ALFWorld
- Download data and create MySQL environment for InterCode-SQL
- Download expert trajectories for agent training

## 🚀 Quick Start

### 1. SFT training for base agent

```bash
# ALFWorld
FORCE_TORCHRUN=1 llamafactory-cli train train/alfworld.yaml
# WebShop
FORCE_TORCHRUN=1 llamafactory-cli train train/webshop.yaml
# InterCode-SQL
FORCE_TORCHRUN=1 llamafactory-cli train train/intercode_sql.yaml
```

### 2. Collect trajectory-level & step-level preference data

Generate base agent trajectories for tasks in train set:

```bash
bash pipeline/trajectory/run_trajectory.sh
```

MC sampling for step-level DPO:

```bash
bash pipeline/step/run_mc_sample.sh
```

Construct trajectory-level preference data:

```bash
python pipeline/trajectory/create_preference_traj.py --task alfworld --output_path OUTPUT_PATH --traj_path SAMPLED_DATA_PATH
```

Construct step-level preference data:

```bash
python pipeline/step/create_preference_step.py --task alfworld --output_path OUTPUT_PATH --traj_path SAMPLED_DATA_PATH
```

### 3. Collect group-level preference data

Group expert trajectories with some strategy:

```bash
python pipeline/group/group_semantic.py
```

Generate negative groups with base agent:

```bash
bash pipeline/group/run_negative.sh
```

MC sampling for group-level DPO:

```bash
bash pipeline/group/run_negative_mc.sh
```

Construct group-level preference data:

```bash
python pipeline/group/create_preference_group.py --task alfworld --output_path OUTPUT_PATH --traj_path SAMPLED_DATA_PATH
```

Construct curriculum learning data by bucketing:

```bash
python pipeline/group/create_curriculum.py
```

### 4. Hierarchical preference learning

Conduct hierarchical preference learning training with collected multi-granularity data.

## 🧪 Evaluation

```bash
# ALFWorld (seen)
python evaluation.py --exp_config alfworld --agent_config agent --model_name MODEL --split dev --api_base API_BASE --api_key API_KEY
# ALFWorld (unseen)
python evaluation.py --exp_config alfworld --agent_config agent --model_name MODEL --split test --api_base API_BASE --api_key API_KEY
# WebShop
python evaluation.py --exp_config webshop --agent_config agent --model_name MODEL --split test --api_base API_BASE --api_key API_KEY
# InterCode-SQL
python evaluation.py --exp_config intercode_sql --agent_config agent --model_name MODEL --split test --api_base API_BASE --api_key API_KEY
```


## 🙏 Acknowledgements

This repository is primarily adapted from [IPR](https://github.com/WeiminXiong/IPR) and [MPO](https://github.com/WeiminXiong/MPO). We sincerely thank the authors for making their code open-source.


## 📝 Citation

If you find this repository or our paper useful for your research, please consider citing:

```bibtex
@inproceedings{gao2026solving,
  title={Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon {LLM} Agents},
  author={Heyang Gao and Zexu Sun and Erxue Min and Hengyi Cai and Shuaiqiang Wang and Dawei Yin and Xu Chen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=s8usvGHYlk}
}
```

