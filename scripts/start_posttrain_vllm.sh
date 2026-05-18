#!/usr/bin/env bash
set -euo pipefail

# 使用 ../post_train/.venv 中的 vLLM 启动 OpenAI-compatible server。
# 默认进入 tmux 后台会话，避免长时间服务阻塞当前终端。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POST_TRAIN_ROOT="${POST_TRAIN_ROOT:-${PROJECT_ROOT}/../post_train}"
VLLM_BIN="${VLLM_BIN:-${POST_TRAIN_ROOT}/.venv/bin/vllm}"

MODEL_PATH="${MODEL_PATH:-/home/fengshuwen/data/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-1.5b-instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
LORA_NAME="${LORA_NAME:-qwen2.5-1.5b-webshop-sft}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
SESSION_NAME="${SESSION_NAME:-hpl_vllm}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
ATTACH="${ATTACH:-0}"

usage() {
  cat <<'EOF'
用法:
  scripts/start_posttrain_vllm.sh [选项]

常用:
  scripts/start_posttrain_vllm.sh --gpus 1
  scripts/start_posttrain_vllm.sh --gpus 1 --port 8000
  scripts/start_posttrain_vllm.sh --gpus 1 --adapter saves/qwen2_5_1p5b_lora_step_sft

选项:
  --model PATH              base 模型路径，默认 Qwen2.5-1.5B-Instruct 本地 HF snapshot
  --served-model-name NAME  OpenAI API 中使用的模型名，默认 qwen2.5-1.5b-instruct
  --adapter PATH            LoRA adapter 路径；传入后启用 vLLM LoRA
  --lora-name NAME          LoRA 请求模型名，默认 qwen2.5-1.5b-webshop-sft
  --host HOST               默认 127.0.0.1
  --port PORT               默认 8000
  --gpus IDS                例如 1 或 1,2；不传则继承当前 CUDA_VISIBLE_DEVICES
  --session NAME            tmux 会话名，默认 hpl_vllm
  --max-model-len N         默认 4096
  --gpu-memory-utilization  默认 0.85
  --attach                  启动后 attach 到 tmux
  -h, --help                显示帮助

HPL 评测时：
  --api_base http://127.0.0.1:8000/v1
  --api_key EMPTY
  --model_name qwen2.5-1.5b-instruct

如果启用了 --adapter，评测时 --model_name 通常使用 --lora-name 的值。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --adapter)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --lora-name)
      LORA_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --attach)
      ATTACH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "找不到 vLLM 可执行文件: ${VLLM_BIN}" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "找不到 tmux；请先安装 tmux 或手动在后台运行 vLLM。" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型路径不存在: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -n "${ADAPTER_PATH}" ]]; then
  if [[ "${ADAPTER_PATH}" != /* ]]; then
    ADAPTER_PATH="${PROJECT_ROOT}/${ADAPTER_PATH}"
  fi
  if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "LoRA adapter 路径不存在: ${ADAPTER_PATH}" >&2
    exit 1
  fi
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux 会话已存在: ${SESSION_NAME}"
  echo "查看日志: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

cmd=(
  "${VLLM_BIN}" serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --disable-log-requests
)

if [[ -n "${ADAPTER_PATH}" ]]; then
  cmd+=(
    --enable-lora
    --max-lora-rank 32
    --lora-modules "${LORA_NAME}=${ADAPTER_PATH}"
  )
fi

quoted_cmd="$(printf '%q ' "${cmd[@]}")"
tmux_cmd="cd $(printf '%q' "${PROJECT_ROOT}") && "
if [[ -n "${GPU_IDS}" ]]; then
  tmux_cmd+="export CUDA_VISIBLE_DEVICES=$(printf '%q' "${GPU_IDS}") && "
fi
tmux_cmd+="${quoted_cmd} 2>&1 | tee $(printf '%q' "${LOG_FILE}")"

tmux new-session -d -s "${SESSION_NAME}" "${tmux_cmd}"

echo "已启动 tmux 会话: ${SESSION_NAME}"
echo "API base: http://${HOST}:${PORT}/v1"
echo "base model_name: ${SERVED_MODEL_NAME}"
if [[ -n "${ADAPTER_PATH}" ]]; then
  echo "lora model_name: ${LORA_NAME}"
fi
echo "日志: ${LOG_FILE}"
echo "查看: tmux attach -t ${SESSION_NAME}"
echo "停止: tmux kill-session -t ${SESSION_NAME}"

if [[ "${ATTACH}" == "1" ]]; then
  tmux attach -t "${SESSION_NAME}"
fi
