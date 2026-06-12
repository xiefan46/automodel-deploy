#!/bin/bash
# download_models.sh — 预下载 HF 模型到本地,避免训练时拉
#
# 用法:
#   bash download_models.sh                          # 默认下 Gemma-3-270m
#   bash download_models.sh google/gemma-3-270m
#   bash download_models.sh google/gemma-3-270m Qwen/Qwen3-0.6B
#
# 模型保存到 ${HF_HOME:-~/.cache/huggingface}/hub/
# Llama / Gemma2+ 等 gated 模型需要先在 HF 网页同意 license

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

# 默认模型(hello-world 用)
DEFAULT_MODELS=("google/gemma-3-270m")

# 参数 → 模型列表
if [ $# -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
else
    MODELS=("$@")
fi

# ─── 装 HF CLI ───
if ! command -v hf >/dev/null 2>&1; then
    log "安装 HF CLI..."
    pip3 install -qU "huggingface_hub[cli]" hf_transfer
fi

# ─── 登录检查(下 gated 模型必须)───
need_login=0
for m in "${MODELS[@]}"; do
    case "$m" in
        meta-llama/*|google/gemma-2*|google/gemma-3*|google/gemma-4*)
            need_login=1
            ;;
    esac
done
if [ $need_login -eq 1 ]; then
    if ! hf auth whoami >/dev/null 2>&1 && [ -z "${HF_TOKEN:-}" ]; then
        log "需要登录 HF(检测到 gated 模型)"
        log "  Token: ${BOLD}https://huggingface.co/settings/tokens${RESET}"
        hf auth login
    fi
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

# ─── 下载 ───
for m in "${MODELS[@]}"; do
    log "下载 $m ..."
    SECONDS=0
    if hf download "$m" 2>&1 | tail -3; then
        log "  完成 ($m, ${SECONDS}s)"
    else
        warn "  $m 下载失败(可能没同意 license 或 token 没权限)"
        warn "  Llama: https://huggingface.co/$m → 点 'Agree and access'"
    fi
done

echo -e "\n${BOLD}${GREEN}模型下载完毕${RESET}"
echo -e "本地缓存位置: ${HF_HOME:-$HOME/.cache/huggingface}/hub/"
