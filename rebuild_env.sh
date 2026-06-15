#!/bin/bash
# rebuild_env.sh — 从零重建 Automodel `.venv` 并推送到 HF Hub
#
# 当 HF cache 丢失/损坏或要更新依赖时用。
#
# 流程:
#   [1/4] uv venv + uv sync --frozen --extra ${EXTRAS}  ~10-20 min
#         (含 flash-attn / transformer-engine 源码编译)
#   [2/4] 验证关键 import
#   [3/4] 打包压缩 (tar + zstd)                          ~2-3 min
#   [4/4] 推送到 HF Hub (xiefan46/automodel-env-cache)   ~2-3 min
#
# RunPod 基础镜像: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#
# 用法:
#   bash rebuild_env.sh [automodel_repo_path]    # 默认 /root/Automodel
#   SKIP_HF_UPLOAD=1 bash rebuild_env.sh         # 只构建本地,不推送
#   HF_CACHE_REPO=user/repo bash rebuild_env.sh  # 自定义仓库
#   EXTRAS=all bash rebuild_env.sh               # 默认 all,所有可选依赖
#   EXTRAS=vlm bash rebuild_env.sh               # 只装 vlm
#   EXTRAS= bash rebuild_env.sh                  # 只装基础(不带 --extra)

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
EXTRAS="${EXTRAS-all}"   # 默认装所有可选依赖
PY="${VENV_DIR}/bin/python"

[ -d "$AUTOMODEL_ROOT" ] || err "Automodel 仓库不存在: $AUTOMODEL_ROOT (用法: bash rebuild_env.sh /path/to/Automodel)"
[ -f "${AUTOMODEL_ROOT}/pyproject.toml" ] || err "${AUTOMODEL_ROOT}/pyproject.toml 不存在"
[ -f "${AUTOMODEL_ROOT}/uv.lock" ] || err "${AUTOMODEL_ROOT}/uv.lock 不存在 (Automodel 用 uv,要锁文件)"

installed() { "$PY" -c "import $1" 2>/dev/null; }

# ─── NVIDIA repo + CUDA forward-compat / toolkit 自动安装 ───
# Automodel 的 pyproject.toml 把 torch 锁死在 cu130 index。RunPod 暂时没
# 现成的 cu13 镜像,我们就自己装:
#   - cuda-compat-13-0 (~30 MB): 让 cu12.x driver 跑 cu130 程序 — 每个 pod 都装
#   - cuda-toolkit-13-0 (~4 GB): 提供 nvcc,编译 mamba/conv1d/TE 时需要 — 仅 rebuild

ensure_nvidia_cuda_repo() {
    if apt-cache policy 2>/dev/null | grep -q "developer.download.nvidia.com/compute/cuda"; then
        return
    fi
    log "添加 NVIDIA CUDA apt repo..."
    . /etc/os-release
    local DISTRO="${ID}${VERSION_ID//./}"  # ubuntu2204 / ubuntu2404
    local DEB_ARCH ARCH
    DEB_ARCH=$(dpkg --print-architecture)
    case "$DEB_ARCH" in
        amd64) ARCH="x86_64" ;;
        arm64) ARCH="sbsa" ;;
        *) err "未支持的架构: $DEB_ARCH" ;;
    esac
    local URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"
    wget -q "$URL" -O /tmp/cuda-keyring.deb \
        || err "下载 cuda-keyring 失败 (distro=${DISTRO} arch=${ARCH})"
    dpkg -i /tmp/cuda-keyring.deb >/dev/null
    apt-get update -qq
    rm -f /tmp/cuda-keyring.deb
}

ensure_cuda_compat() {
    # 参数: major-minor 比如 "13-0"
    local MM="$1"
    local PKG="cuda-compat-${MM}"
    if dpkg -s "$PKG" >/dev/null 2>&1; then
        log "${PKG} 已装"
    else
        ensure_nvidia_cuda_repo
        log "装 ${PKG} (~30 MB,让 driver 兼容 cu${MM/-/.})..."
        apt-get install -y "$PKG" >/dev/null
    fi
    local DOT="${MM/-/.}"
    local COMPAT_DIR="/usr/local/cuda-${DOT}/compat"
    if [ -d "$COMPAT_DIR" ]; then
        export LD_LIBRARY_PATH="${COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
        if ! grep -q "cuda-${DOT}/compat" ~/.bashrc 2>/dev/null; then
            echo "export LD_LIBRARY_PATH=${COMPAT_DIR}:\$LD_LIBRARY_PATH" >> ~/.bashrc
        fi
    fi
}

ensure_cuda_toolkit() {
    local MM="$1"
    local DOT="${MM/-/.}"
    if [ -x "/usr/local/cuda-${DOT}/bin/nvcc" ]; then
        log "CUDA ${DOT} toolkit 已装"
    else
        ensure_nvidia_cuda_repo
        log "装 cuda-toolkit-${MM} (~3-5 GB, ~10 min,首次)..."
        apt-get install -y "cuda-toolkit-${MM}" >/dev/null
    fi
    export CUDA_HOME="/usr/local/cuda-${DOT}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    if ! grep -q "CUDA_HOME=/usr/local/cuda-${DOT}" ~/.bashrc 2>/dev/null; then
        echo "export CUDA_HOME=/usr/local/cuda-${DOT}" >> ~/.bashrc
        echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    fi
}

# 检测并按需安装 CUDA 兼容层 + toolkit
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    CUDA_RUNTIME_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | cut -d. -f1)
    REQUIRED_CUDA_MAJOR=$(grep -oP "pytorch-cu\K[0-9]+" "${AUTOMODEL_ROOT}/pyproject.toml" | sort -u | tail -1 | cut -c1-2)
    REQUIRED_MM="${REQUIRED_CUDA_MAJOR:0:2}-0"   # "13-0"

    log "CUDA 状态:driver=${CUDA_DRIVER_VER} runtime=cu${CUDA_RUNTIME_VER:-?} 需要=cu${REQUIRED_CUDA_MAJOR}0"

    if [ -n "${CUDA_RUNTIME_VER:-}" ] && [ "$CUDA_RUNTIME_VER" -lt "$REQUIRED_CUDA_MAJOR" ]; then
        ensure_cuda_compat "$REQUIRED_MM"
    elif [ -n "${CUDA_RUNTIME_VER:-}" ]; then
        log "driver 已支持 cu${REQUIRED_CUDA_MAJOR}.x,不需要 compat"
    fi

    # --extra all / cuda / fa 会触发 native 编译,要 nvcc
    case "${EXTRAS}" in
        all|*cuda*|*fa*) ensure_cuda_toolkit "$REQUIRED_MM" ;;
    esac
fi

# ─── HF helpers ───
ensure_hf_cli() {
    command -v hf >/dev/null 2>&1 && return
    log "安装 HF CLI..."
    pip3 install -qU huggingface_hub hf_transfer
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
    log "HF 未登录,请粘贴 Write token (https://huggingface.co/settings/tokens):"
    hf auth login
}

# ─── 系统工具 ───
export DEBIAN_FRONTEND=noninteractive
NEED_PKGS=""
command -v tmux  &>/dev/null || NEED_PKGS="tmux"
command -v zstd  &>/dev/null || NEED_PKGS="${NEED_PKGS} zstd"
command -v cmake &>/dev/null || NEED_PKGS="${NEED_PKGS} cmake"
command -v ninja &>/dev/null || NEED_PKGS="${NEED_PKGS} ninja-build"
command -v curl  &>/dev/null || NEED_PKGS="${NEED_PKGS} curl"
if [ -n "$NEED_PKGS" ]; then
    log "安装系统工具:${NEED_PKGS}..."
    apt-get install -y ${NEED_PKGS} 2>/dev/null || { apt-get update && apt-get install -y ${NEED_PKGS}; }
fi

# ─── 装 uv ───
if ! command -v uv &>/dev/null; then
    log "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || err "uv 安装失败"
fi
if ! grep -q "\.local/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# ─── HF 登录(提前,避免后面卡在 token 输入)───
if [ "${SKIP_HF_UPLOAD:-}" != "1" ]; then
    ensure_hf_cli
    ensure_hf_login
fi

TOTAL_START=$SECONDS

# ──────────────────────────────────────────────────────────
# [1/4] uv venv + uv sync
# ──────────────────────────────────────────────────────────
log "[1/4] 用 uv 重建 .venv (EXTRAS=${EXTRAS:-<none>})..."

cd "$AUTOMODEL_ROOT"

# 清理旧 venv
if [ -d "$VENV_DIR" ]; then
    log "  删除旧 .venv..."
    rm -rf "$VENV_DIR"
fi

log "  创建 .venv..."
uv venv

SYNC_START=$SECONDS
if [ -z "${EXTRAS}" ]; then
    log "  uv sync --frozen (基础依赖,不带 --extra)..."
    uv sync --frozen
else
    log "  uv sync --frozen --extra ${EXTRAS} (含可选依赖,会编译 flash-attn / TE,耐心等)..."
    uv sync --frozen --extra "${EXTRAS}"
fi
log "    uv sync 完成 ($((SECONDS - SYNC_START))s)"

# ──────────────────────────────────────────────────────────
# [2/4] 验证
# ──────────────────────────────────────────────────────────
log "[2/4] 验证..."
"$PY" -c "
import torch; print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import nemo_automodel; print(f'nemo_automodel: OK ({nemo_automodel.__file__})')
import transformers; print(f'transformers: {transformers.__version__}')
# 可选:这些 import 失败不致命,只警告
try: import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
except ImportError: print('flash_attn: NOT installed (可选)')
try: import transformer_engine; print(f'transformer_engine: {transformer_engine.__version__}')
except ImportError: print('transformer_engine: NOT installed (可选)')
print('\n=== 验证通过 ===')
"

# ──────────────────────────────────────────────────────────
# [3/4] 打包
# ──────────────────────────────────────────────────────────
log "[3/4] 打包 .venv..."
mkdir -p "$CACHE_DIR"
PACK_START=$SECONDS
# 只打包 .venv 目录内容(不含外层 .venv 那一层),restore 时直接展开到 ${VENV_DIR}
tar cf - -C "$VENV_DIR" . | zstd -T0 -3 -f -o "$ENV_ARCHIVE"
ARCHIVE_SIZE=$(du -sh "$ENV_ARCHIVE" | cut -f1)
log "  打包完成: ${ARCHIVE_SIZE} ($((SECONDS - PACK_START))s)"

# ──────────────────────────────────────────────────────────
# [4/4] 推送 HF
# ──────────────────────────────────────────────────────────
if [ "${SKIP_HF_UPLOAD:-}" = "1" ]; then
    log "[4/4] SKIP_HF_UPLOAD=1,跳过 HF 上传"
    log "  手动上传:"
    log "    hf upload --repo-type=dataset --private ${HF_CACHE_REPO} ${ENV_ARCHIVE} automodel_venv.tar.zst"
else
    log "[4/4] 推送到 HF Hub: ${HF_CACHE_REPO}..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    UPLOAD_START=$SECONDS
    if hf upload --repo-type=dataset --private "$HF_CACHE_REPO" \
            "$ENV_ARCHIVE" automodel_venv.tar.zst; then
        log "  上传完成 ($((SECONDS - UPLOAD_START))s)"
    else
        warn "  HF 上传失败。本地 cache 仍可用: $ENV_ARCHIVE"
        warn "  手动重试: hf upload --repo-type=dataset --private $HF_CACHE_REPO $ENV_ARCHIVE automodel_venv.tar.zst"
    fi
fi

TOTAL_TIME=$((SECONDS - TOTAL_START))
echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  环境重建完成! (${TOTAL_TIME}s)${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "\n${YELLOW}激活: source ${VENV_DIR}/bin/activate${RESET}"
echo -e "${YELLOW}下次新 pod: bash setup_env.sh (~2-5 min 从 HF 恢复)${RESET}\n"
