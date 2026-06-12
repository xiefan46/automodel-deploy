#!/bin/bash
# setup_env.sh — 从 HF Hub 缓存快速恢复 Automodel `.venv`
#
# 流程：
#   1. 本地 .venv 已存在 → 验证 torch + nemo_automodel 可 import → 直接复用
#   2. 本地不存在 → 从 HF Hub 下载缓存 → 解压到 ${AUTOMODEL_ROOT}/.venv
#
# HF cache 仓库: xiefan46/automodel-env-cache (private dataset)
# 如果 cache 丢失或要更新，运行: bash rebuild_env.sh
#
# 用法:
#   bash setup_env.sh [automodel_repo_path]      # 默认 /root/Automodel
#   FORCE_RESTORE=1 bash setup_env.sh            # 强制重新解压
#   HF_CACHE_REPO=user/repo bash setup_env.sh    # 自定义仓库

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
AUTOMODEL_ROOT="${1:-/root/Automodel}"
VENV_DIR="${AUTOMODEL_ROOT}/.venv"
HF_CACHE_REPO="${HF_CACHE_REPO:-xiefan46/automodel-env-cache}"
CACHE_DIR="/root/automodel_cache"
ENV_ARCHIVE="${CACHE_DIR}/automodel_venv.tar.zst"
PY="${VENV_DIR}/bin/python"

[ -d "$AUTOMODEL_ROOT" ] || err "Automodel 仓库不存在: $AUTOMODEL_ROOT (用法: bash setup_env.sh /path/to/Automodel)"
[ -f "${AUTOMODEL_ROOT}/pyproject.toml" ] || err "${AUTOMODEL_ROOT}/pyproject.toml 不存在,确认是 Automodel 仓库"

installed() { "$PY" -c "import $1" 2>/dev/null; }

# ─── 系统工具 ───
export DEBIAN_FRONTEND=noninteractive
NEED_PKGS=""
command -v zstd &>/dev/null || NEED_PKGS="${NEED_PKGS} zstd"
command -v tmux &>/dev/null || NEED_PKGS="${NEED_PKGS} tmux"
command -v curl &>/dev/null || NEED_PKGS="${NEED_PKGS} curl"
if [ -n "$NEED_PKGS" ]; then
    log "安装系统工具:${NEED_PKGS}..."
    apt-get install -y ${NEED_PKGS} 2>/dev/null || { apt-get update && apt-get install -y ${NEED_PKGS}; }
fi

# ─── 装 uv(Automodel 强制要求,不要用 pip)───
if ! command -v uv &>/dev/null; then
    log "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || err "uv 安装失败"
fi
# 让后续 shell 也能找到 uv
if ! grep -q "\.local/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# ─── HF helpers ───
ensure_hf_cli() {
    command -v hf >/dev/null 2>&1 && return
    log "安装 HF CLI..."
    # 用 uv 装到一个独立工具环境,不污染系统 python
    uv tool install "huggingface_hub[cli,hf_transfer]" 2>/dev/null \
        || pip3 install -qU "huggingface_hub[cli]" hf_transfer
    command -v hf >/dev/null 2>&1 || err "hf CLI 安装失败"
}

ensure_hf_login() {
    if hf auth whoami >/dev/null 2>&1; then
        log "HF 已登录: $(hf auth whoami 2>/dev/null | head -1)"
        return
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        log "使用 HF_TOKEN 环境变量"
        return
    fi
    log "HF 未登录"
    log "  Token: ${BOLD}https://huggingface.co/settings/tokens${RESET}"
    hf auth login
}

# ─── 主流程 ───

# 1. 本地 .venv 已存在 + 关键包能 import → 跳过解压
if [ -d "$VENV_DIR" ] && installed torch && installed nemo_automodel && [ "${FORCE_RESTORE:-}" != "1" ]; then
    log "本地 .venv 已存在($(du -sh $VENV_DIR | cut -f1)),跳过解压"
else
    if [ "${FORCE_RESTORE:-}" = "1" ] && [ -d "$VENV_DIR" ]; then
        log "FORCE_RESTORE=1: 删除本地 .venv"
        rm -rf "$VENV_DIR"
    fi

    # 2. 没有本地 archive → 从 HF 拉
    if [ ! -f "$ENV_ARCHIVE" ]; then
        ensure_hf_cli
        ensure_hf_login
        log "从 HF 下载 cache: ${HF_CACHE_REPO}"
        mkdir -p "$CACHE_DIR"
        export HF_HUB_ENABLE_HF_TRANSFER=1
        SECONDS=0
        if ! hf download --repo-type=dataset "$HF_CACHE_REPO" automodel_venv.tar.zst --local-dir "$CACHE_DIR"; then
            err "HF 下载失败。如果 cache 不存在或损坏,运行 'bash rebuild_env.sh' 完整重建"
        fi
        log "下载完成 (${SECONDS}s): $(du -sh $ENV_ARCHIVE | cut -f1)"
    else
        log "使用本地缓存: $ENV_ARCHIVE ($(du -sh $ENV_ARCHIVE | cut -f1))"
    fi

    # 3. 解压到 .venv
    log "解压到 ${VENV_DIR}..."
    SECONDS=0
    mkdir -p "$VENV_DIR"
    zstd -d "$ENV_ARCHIVE" --stdout | tar xf - -C "$VENV_DIR"
    log "解压完成 (${SECONDS}s)"
fi

# 4. 验证
log "验证环境..."
"$PY" -c "
import torch
print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import nemo_automodel
print(f'nemo_automodel: OK ({nemo_automodel.__file__})')
import transformers
print(f'transformers: {transformers.__version__}')
print('\n=== 验证通过 ===')
" || err "验证失败,cache 可能损坏。重建: bash rebuild_env.sh"

# 5. (可选)装一些训练时常用但不在 sync 里的工具
if ! installed wandb; then
    log "补装 wandb..."
    (cd "$AUTOMODEL_ROOT" && uv pip install wandb --quiet) 2>/dev/null || true
fi

echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  Automodel 环境就绪${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "\n${YELLOW}激活 venv:${RESET}"
echo -e "  source ${VENV_DIR}/bin/activate"
echo -e "\n${YELLOW}跑 hello-world:${RESET}"
echo -e "  bash $(dirname "$0")/run_hello_world.sh"
echo -e "\n${YELLOW}重建 cache (HF 丢失时):${RESET}"
echo -e "  bash $(dirname "$0")/rebuild_env.sh\n"
