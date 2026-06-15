# automodel-deploy

RunPod 一键部署 [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) 训练环境。

## RunPod 基础镜像

使用 `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

创建 Pod 时在 Docker Image 处填入此镜像名（已包含 PyTorch 2.8 + CUDA 12.8 + cuDNN）。

> Automodel 要求 Python 3.10+ 和 PyTorch 2.6+，上述镜像满足。

## 策略

环境缓存的 source of truth 是 **HF Hub dataset** (`xiefan46/automodel-env-cache`，private)。

| 场景 | 脚本 | 行为 | 耗时 |
|------|------|------|------|
| 重建 cache | `rebuild_env.sh` | `uv sync --frozen --extra all` → 打包 → 推送 HF Hub | ~15-25 min |
| 日常启动 | `setup_env.sh` | 从 HF Hub 下载 cache → 解压 → 链接 | ~2-5 min |
| 重入容器 | `setup_env.sh` | 检测到本地 `.venv` 已存在 → 直接复用 | ~5 s |
| 跑 hello-world | `run_hello_world.sh` | 拉小模型 + 跑 Gemma-3-270m SQuAD SFT | ~5 min |

> 跟 verl-deploy 的差异：Automodel 用 `uv` 不是 conda；缓存对象是项目级 `.venv` 而不是全局 conda env。

## 快速开始

> **所有命令包在 `tmux` 里 — SSH 断了训练不停。末尾 `; bash` 让 tmux 在训练完后保留 shell。**
> **`Ctrl-b d` 脱离 tmux,`tmux attach -t auto` 重新进。**

### Qwen2.5-7B PEFT(1× H100 80GB,推荐首跑)

最快从零到看 loss(~30 min,首次):

```bash
tmux new -s auto "git clone https://github.com/xiefan46/Automodel.git /root/Automodel && git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && SKIP_HF_UPLOAD=1 EXTRAS= bash /root/automodel-deploy/rebuild_env.sh && source /root/Automodel/.venv/bin/activate && cd /root/Automodel && automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml; bash"
```

`SKIP_HF_UPLOAD=1` 不推 HF cache;`EXTRAS=` 跳过 TE/flash-attn 编译。

### 首次 + 想种 HF cache(下次开 pod 才能秒拉)

```bash
export HF_TOKEN=hf_xxxxxxxxx
tmux new -s auto "git clone https://github.com/xiefan46/Automodel.git /root/Automodel && git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && bash /root/automodel-deploy/rebuild_env.sh && source /root/Automodel/.venv/bin/activate && cd /root/Automodel && automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml; bash"
```

含 `--extra all`(装 TE + flash-attn)+ 推 HF Hub。首次 ~25 min setup + 训练。

### 后续 pod(HF cache 已存在,2-5 min 拉缓存)

```bash
export HF_TOKEN=hf_xxxxxxxxx
tmux new -s auto "git clone https://github.com/xiefan46/Automodel.git /root/Automodel && git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && bash /root/automodel-deploy/setup_env.sh && source /root/Automodel/.venv/bin/activate && cd /root/Automodel && automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml; bash"
```

### Hello World(Gemma-3-270m,任何 GPU,~10 min)

```bash
tmux new -s auto "git clone https://github.com/xiefan46/Automodel.git /root/Automodel && git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && SKIP_HF_UPLOAD=1 EXTRAS= bash /root/automodel-deploy/rebuild_env.sh && bash /root/automodel-deploy/run_hello_world.sh; bash"
```

### 换 recipe(只改最后一段)

```bash
# 小模型快测
automodel examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml

# Llama-1B(需 HF login + 同意 Llama license)
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node 8

# LoRA 省显存
automodel examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml

# 冒烟模式(20 步停)
automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml --step_scheduler.max_steps 20
```

### 推荐的 hello-world 模型(按门槛由低到高)

| 模型 | yaml | 大小 | 显存 | HF gating |
|---|---|---|---|---|
| **Gemma-3-270m** ⭐ | `examples/llm_finetune/gemma/gemma_3_270m_squad.yaml` | 270M | ~1.5GB | 无 |
| Qwen3-0.6B | `examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml` | 600M | ~3GB | 无 |
| Llama-3.2-1B | `examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml` | 1B | ~5GB | 需要登录 + 同意 |

## 缓存重建（HF cache 损坏或要更新依赖时）

```bash
# 在任一 pod 上跑（出口带宽快,HF 上传 2-3 min）
bash /root/automodel-deploy/rebuild_env.sh
```

完整跑：`uv venv` → `uv sync --frozen --extra all`（含 transformer_engine、deepep 等所有可选依赖）→ 打包 → 推送到 `xiefan46/automodel-env-cache`。完成后所有新 pod 跑 `setup_env.sh` 拉新版本。

```bash
# 只重建本地，不推送 HF
SKIP_HF_UPLOAD=1 bash rebuild_env.sh

# 只装基础依赖(不要 vlm / cuda 大包,省时间)
EXTRAS= bash rebuild_env.sh
```

首次运行会要求 `hf auth login` — 准备好 [Write token](https://huggingface.co/settings/tokens) 粘贴。

## 选项

```bash
# 自定义 Automodel 路径（默认 /root/Automodel）
bash setup_env.sh /path/to/Automodel

# 强制重新解压（删本地 .venv,重新从 HF 拉）
FORCE_RESTORE=1 bash setup_env.sh

# 自定义缓存仓库
HF_CACHE_REPO=user/repo bash setup_env.sh

# 自定义 uv extras（重建时）
EXTRAS=all bash rebuild_env.sh        # 默认 all
EXTRAS=vlm bash rebuild_env.sh        # 只装 vlm
EXTRAS= bash rebuild_env.sh           # 只装基础（不带 --extra）
```

## 监控

### Wandb

```bash
# 远程机器上一次性登录
wandb login

# yaml 里加 wandb 段或者 CLI override:
automodel <yaml> --wandb.project=automodel-hello --wandb.name=gemma-270m-squad
```

### GPU 占用

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 环境说明

- **Python**: 3.11(RunPod 镜像自带)
- **包管理**: uv(Automodel 强制要求)
- **venv 位置**: `${AUTOMODEL_ROOT}/.venv`(uv 默认)
- **缓存 source of truth**: HF Hub dataset `xiefan46/automodel-env-cache`(private)
- **本地缓存路径**: `/root/automodel_cache/automodel_venv.tar.zst`(~3-5 GB)
- **Automodel 安装方式**: `uv sync --frozen` editable,改代码不用重装

## 常见问题

### `uv: command not found`

setup_env.sh 第一次会自动装 uv。如果手动想装：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 想跑 Llama 但报 401 access denied

需要 HF token 且**已同意 Llama license**:
1. 访问 https://huggingface.co/meta-llama/Llama-3.2-1B 点同意
2. `huggingface-cli login` 粘贴 token

不想用 Llama 就跑 Gemma 或 Qwen 例子,无 gating。

### MoE 模型跑 DeepEP 报缺依赖

Automodel 的 `--extra all` 不包含 DeepEP(需要 H100 + 编译)。如果要 DeepEP:
```bash
uv pip install deep-ep   # 在 venv 激活状态下
```

### FlashAttention 编译慢

`uv sync` 大头是 flash-attn 源码编译(~5-10 min on Hopper)。这就是为啥要 HF cache —— 后续 pod 就不用重编了。
