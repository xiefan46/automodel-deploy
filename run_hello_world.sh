#!/bin/bash
# run_hello_world.sh — 跑 Automodel 最小示例验证整个 stack 通
#
# 默认:Gemma-3-270m SQuAD SFT(无 HF gating,单卡 ~1.5GB 显存)
# 用来验证:
#   - venv 激活
#   - HF 模型下载
#   - FSDP2 包装
#   - dataloader / loss / optimizer
#   - 至少跑完几十步训练 + 一次 val
#
# 用法:
#   bash run_hello_world.sh                              # 默认 Gemma-3-270m,单卡
#   NPROC=8 bash run_hello_world.sh                      # 8 卡
#   RECIPE=examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml bash run_hello_world.sh
#   AUTOMODEL_ROOT=/path/to/Automodel bash run_hello_world.sh
#   MAX_STEPS=20 bash run_hello_world.sh                 # 强制跑 20 步就停(快速冒烟)

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

# ─── 配置 ───
AUTOMODEL_ROOT="${AUTOMODEL_ROOT:-/root/Automodel}"
VENV_DIR="${AUTOMODEL_ROOT}/.venv"
RECIPE="${RECIPE:-examples/llm_finetune/gemma/gemma_3_270m_squad.yaml}"
NPROC="${NPROC:-1}"
MAX_STEPS="${MAX_STEPS:-}"

# ─── 检查 ───
[ -d "$AUTOMODEL_ROOT" ] || err "Automodel 不存在: $AUTOMODEL_ROOT"
[ -d "$VENV_DIR" ] || err ".venv 不存在: $VENV_DIR (先跑 setup_env.sh)"

RECIPE_PATH="${AUTOMODEL_ROOT}/${RECIPE}"
[ -f "$RECIPE_PATH" ] || err "Recipe 不存在: $RECIPE_PATH"

# ─── 激活 venv ───
log "激活 venv: ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "$AUTOMODEL_ROOT"

# ─── 信息 ───
echo -e "\n${BOLD}${GREEN}==================== Hello World ====================${RESET}"
echo -e "${BOLD}Recipe:${RESET}     ${RECIPE}"
echo -e "${BOLD}NPROC:${RESET}      ${NPROC}"
echo -e "${BOLD}MAX_STEPS:${RESET}  ${MAX_STEPS:-<from yaml>}"
echo -e "${BOLD}GPU:${RESET}        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo -e "${BOLD}cwd:${RESET}        $(pwd)"
echo -e "${GREEN}=====================================================${RESET}\n"

# ─── 跑训练 ───
CMD=("automodel" "$RECIPE")
if [ "$NPROC" != "1" ]; then
    CMD+=("--nproc-per-node" "$NPROC")
fi
if [ -n "$MAX_STEPS" ]; then
    CMD+=("--step_scheduler.max_steps" "$MAX_STEPS")
fi

log "执行: ${CMD[*]}"
"${CMD[@]}"

echo -e "\n${BOLD}${GREEN}==================== Hello World DONE ====================${RESET}"
echo -e "${YELLOW}尝试其他 recipe:${RESET}"
echo -e "  RECIPE=examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml bash $(basename $0)"
echo -e "  RECIPE=examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml NPROC=8 bash $(basename $0)"
echo -e "${YELLOW}LoRA 省显存:${RESET}"
echo -e "  RECIPE=examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml bash $(basename $0)\n"
