# automodel-deploy

RunPod 一键部署 NVIDIA-NeMo/Automodel 训练环境的脚本集合。

## 一键复制粘贴(常用场景)

> 所有训练都在 `tmux` 里跑 — SSH 断了训练不停。
> 中途按 `Ctrl-b d` 脱离 tmux,`tmux attach -t auto` 重新进。

### A. 首次 pod,跑 Qwen2.5-7B PEFT(最常用 — 最快从零到训练)

进 tmux:
```bash
tmux new -s auto
```

在 tmux 里跑(一坨复制):
```bash
git clone https://github.com/xiefan46/Automodel.git /root/Automodel && \
git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && \
SKIP_HF_UPLOAD=1 EXTRAS= bash /root/automodel-deploy/rebuild_env.sh && \
source /root/Automodel/.venv/bin/activate && \
cd /root/Automodel && \
automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml
```

`SKIP_HF_UPLOAD=1` 不推 HF cache,`EXTRAS=` 跳过 TE/flash-attn 编译 → 总耗时 ~5 min setup + ~20 min 训练 = **30 min 内见 loss**。

### B. 首次 pod + 想种 HF cache(下次开 pod 才能秒拉)

进 tmux:
```bash
tmux new -s auto
```

在 tmux 里跑(脚本会交互式问 HF token,粘贴你的 [Write token](https://huggingface.co/settings/tokens)):
```bash
git clone https://github.com/xiefan46/Automodel.git /root/Automodel && \
git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && \
bash /root/automodel-deploy/rebuild_env.sh && \
source /root/Automodel/.venv/bin/activate && \
cd /root/Automodel && \
automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml
```

含 `--extra all`(装 TE + flash-attn)+ 推 HF Hub。**首次 ~25 min setup + 训练**。

### C. 后续 pod(HF cache 已存在,2-5 min 拉缓存)

```bash
tmux new -s auto
```

脚本会交互式问 HF token(从 cache 拉也要 read 权限):
```bash
git clone https://github.com/xiefan46/Automodel.git /root/Automodel && \
git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && \
bash /root/automodel-deploy/setup_env.sh && \
source /root/Automodel/.venv/bin/activate && \
cd /root/Automodel && \
automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml
```

### D. Hello world(Gemma-3-270m,任何 GPU 都行)

```bash
tmux new -s auto
```

```bash
git clone https://github.com/xiefan46/Automodel.git /root/Automodel && \
git clone https://github.com/xiefan46/automodel-deploy.git /root/automodel-deploy && \
SKIP_HF_UPLOAD=1 EXTRAS= bash /root/automodel-deploy/rebuild_env.sh && \
bash /root/automodel-deploy/run_hello_world.sh
```

### 换 recipe 只改最后一段

把 `automodel examples/...yaml` 替换成想跑的 yaml,或加参数:
```bash
automodel examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml             # 小模型快测
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml            # Llama (需 HF login + 同意 license)
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node 8   # 多卡
automodel examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml --step_scheduler.max_steps 20  # 冒烟模式
```

### 监控

新开 SSH 窗口(或 tmux 分屏 `Ctrl-b "`):
```bash
watch -n 1 nvidia-smi             # GPU 占用
tmux attach -t auto                # 看训练日志
```

## 脚本

| 脚本 | 用途 |
|------|------|
| `setup_env.sh` | 从 HF Hub 缓存快速恢复 Automodel `.venv`(~2-5 min) |
| `rebuild_env.sh` | `uv sync --frozen --extra all` 完整重建并推送到 HF Hub(~15-25 min) |
| `run_hello_world.sh` | 跑最小示例(Gemma-3-270m SFT on SQuAD,无 HF gating) |
| `download_models.sh` | 预下载 HF 模型到本地 |

## 环境配置

- **基础镜像**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **包管理**: uv(Automodel 强制要求,不要用 pip install)
- **venv 路径**: `${AUTOMODEL_ROOT}/.venv`,默认 `/root/Automodel/.venv`
- **环境缓存** (source of truth): HF Hub dataset `xiefan46/automodel-env-cache`(private)
- **本地解压目录**: `/root/automodel_cache/automodel_venv.tar.zst`
- **Automodel 仓库**: 默认 `/root/Automodel`

## setup_env.sh

三步:

1. **装 cuda-compat-13-0**(~30 MB,~1 min):cache 里的 venv 用 cu130 PyTorch,如果当前 pod driver 是 cu12.x 需要这个兼容层才能跑 GPU
2. 拉 venv:本地有 archive 直接用,否则从 HF 下载 → 解压
3. 验证 import 通

不需要装 cuda-toolkit(没编译需求,wheel 都 ready)。

如果 HF cache 损坏或不存在,会提示运行 `rebuild_env.sh`。

## rebuild_env.sh

五步:

1. **CUDA 13 自动装**:从 NVIDIA apt repo 装 `cuda-compat-13-0` + `cuda-toolkit-13-0`(如果 driver < cu13 或 `--extra all` 需要 nvcc 编译)~10 min 首次
2. 安装 uv + 创建 venv(~30s)
3. `uv sync --frozen --extra all` ~10-20 min(含 flash-attn / transformer-engine 源码编译)
4. 打包压缩 **.venv** 成 `tar.zst` ~2-3 min(**不打包系统 CUDA**,系统包靠 apt 单独装)
5. 推送到 HF Hub ~2-3 min

`SKIP_HF_UPLOAD=1 bash rebuild_env.sh` 只构建本地不推送。
`EXTRAS=` 跳过所有可选依赖,只装基础(快 5×,但功能少,而且不会装 toolkit)。

## run_hello_world.sh

默认跑 `examples/llm_finetune/gemma/gemma_3_270m_squad.yaml`:
- Gemma-3-270m 模型(无需 HF gating)
- SQuAD 数据集
- 单卡(`--nproc-per-node 1`),~5 分钟跑一个完整 epoch 的几百步
- 用来验证整个 stack(模型加载 / FSDP2 / dataloader / optimizer / checkpoint)能跑通

通过环境变量切其他例子:
```bash
RECIPE=examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml bash run_hello_world.sh
NPROC=8 bash run_hello_world.sh
```

## download_models.sh

下载 HF 模型到本地 `${HOME}/models/` 以便离线训练。

```bash
bash download_models.sh                                # 默认下 Gemma-3-270m
bash download_models.sh Qwen/Qwen3-0.6B
bash download_models.sh google/gemma-3-270m Qwen/Qwen3-0.6B
```

## 重要约定

- **不要在脚本里用 `pip install`** — Automodel 的 CLAUDE.md 明确要求 uv,我们这里也保持一致
- **venv 路径硬编码到缓存里**:缓存是用 `${AUTOMODEL_ROOT}/.venv` 路径打的包,restore 时必须放回同一路径(脚本里默认 `/root/Automodel`)
- **HF token 共享**:`setup_env.sh` 里 login 一次,token 写入 `~/.cache/huggingface/token`,后续 `hf` / `huggingface-cli` 都直接用
- **不假设有 DeepEP / UCCL-EP**:这两个需要 H100/H200 + 额外编译,不进默认 cache;需要的话激活 venv 后 `uv pip install deep-ep`
- **系统 CUDA 不进缓存**:`cuda-compat-13-0` / `cuda-toolkit-13-0` 装在 `/usr/local/cuda-13.0/` 下,**每个新 pod 都重新 apt 装**,不打包到 HF cache。理由:(1) 各 pod 系统环境不一样不能直接搬,(2) apt 装 30 MB compat 包只要 1 min 不慢,(3) 缓存只放可移植的 venv 简单干净
- **LD_LIBRARY_PATH 持久化**:脚本会写 `export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH` 到 `~/.bashrc`,所有新 shell 自动加载

## 跟 verl-deploy 的差异

| 项 | verl-deploy | automodel-deploy |
|---|---|---|
| 包管理 | conda + pip | **uv**(Automodel 要求) |
| env 路径 | `/opt/conda/envs/verl`(系统级) | `/root/Automodel/.venv`(项目级) |
| editable install | `pip install -e .` 单独装 | `uv sync` 自带 editable |
| Megatron 依赖 | 必须(TE/Apex/mbridge/megatron-core) | 可选(Automodel 默认 PyTorch DTensor) |
| MoE kernel | 不涉及 | DeepEP / TE 可选 |

## 维护

修改依赖前:
- 看 [Automodel pyproject.toml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/pyproject.toml) 对照
- 看 [Automodel docker/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/docker) 看官方镜像怎么装
