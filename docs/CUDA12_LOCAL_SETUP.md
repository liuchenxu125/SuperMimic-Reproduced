# 本机 CUDA 12 / Casbot02 环境

本项目可以使用 CUDA 12 训练 Casbot02，不需要与 README 中的 CUDA 13 环境完全一致。

项目已迁移到系统分区的 `/srv/liuchenxu/SuperMimic-Reproduced`。
原桌面路径 `~/Desktop/SuperMimic-Reproduced` 是指向该目录的软链接。
可用 `cd /srv/liuchenxu/SuperMimic-Reproduced` 进入项目；虚拟环境路径已修复，
并验证 CUDA 前向反向、Warp GPU 内存和 Casbot02 checkpoint 推理通过。
这属于本机兼容性适配；训练吞吐和收敛结果需要独立测量。

2026-11-06 检查到的硬件与软件：

| 项目 | 本机配置 |
| --- | --- |
| 系统 | Ubuntu 22.04.5，x86_64 |
| GPU | NVIDIA RTX5880-Ada-48Q，48 GiB 虚拟显存配置，计算能力 8.9 |
| 驱动 | 570.172.18，`nvidia-smi` 显示最高 CUDA 12.8 |
| 内存 | 约 62 GiB |
| Python | 项目 `.venv` 内的 Python 3.12 |
| PyTorch | 2.14.1+cu126 |
| Warp | 1.18.0+cu12，使用 NVIDIA 官方兼容 wheel |
| MuJoCo / MuJoCo Warp | 3.16.0 / 3.16.0 |
| mjlab | 保持项目原来的 `8bb2bc9e187797689ec82c78c35019afd2fd0376` |

`nvidia-smi` 的 CUDA 版本表示驱动支持能力，不表示系统安装了对应版本的 Toolkit。
本配置使用 Python 包提供的运行库。无需为运行这些预编译包另装系统 CUDA 13。
NVIDIA 的兼容表说明 CUDA 13.x 通常需要 580 或更新驱动；CUDA 12.x 的最低驱动属于 525 系列，
但 PTX/JIT 等功能还需要匹配具体构建版本，因此必须实际测试 Warp 和训练链路。

- 驱动兼容说明：https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- Warp CUDA 12 兼容包：https://github.com/NVIDIA/warp/releases/tag/v1.18.0

## 安装与日常使用

在项目根目录执行。`pyproject.toml` 已将 Linux x86_64 的 Warp 固定为本地 CUDA 12 wheel，
避免后续同步时装回默认 CUDA 13 构建。首次安装或换机器时先下载完整 wheel：

```bash
mkdir -p .wheels
curl -fL -C - --retry 2 \
  -o .wheels/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl'
env -u PYTHONPATH uv sync --group dev
```

本次下载的 wheel 为 164,625,100 字节，SHA-256：
`6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656`。

不要只把 `warp-lang` 从直接依赖里注释掉：mjlab / mujoco-warp 仍会间接安装它。
如果默认构建需要更高版本驱动，PyTorch 能用 GPU，Warp 仍然可能无法启动。

## 单动作数据与训练入口

本机存在 `~/Downloads/lafan1_casbot02/walk4_subject1.csv`，前两行均为 34 列，
符合 7 维根部位姿加 27 个关节的格式。以下帧率采用项目 LAFAN1 的 30 Hz 约定；
若自己的重定向文件使用其他帧率，应相应修改 `--input-fps`。
转换不会清洗超限动作，也不能替代数据质量检查。

本次已完成该动作转换：8,196 帧、27 个关节；全部数值有限，按 `1e-5 rad`
容差检查，没有帧超过当前模型的关节硬限位。

```bash
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --robot casbot02 --input-dir ~/Downloads/lafan1_casbot02 \
  --output-dir data/npz/casbot02_local \
  --pattern walk4_subject1.csv --input-fps 30 --output-fps 50 --workers 1
```

先用较小配置跑通真实数据，再扩大并行度。下面是待验证的起步配置，不是已证明收敛的超参数：

```bash
env -u PYTHONPATH uv run sm-train \
  --robot casbot02 --run-name casbot02_local_cu126_walk4 \
  --data.npz-dir data/npz/casbot02_local --data.clips walk4_subject1 \
  --env.no-anchor-obs --env.no-anchor-ori-obs --env.actuator-delay \
  --env.collision-model primitive --env.num-envs 512 \
  --env.replay-capacity 256 \
  --policy-trainer.batch-size 256 --wm-trainer.batch-size 256 \
  --wm-trainer.compile None --policy-trainer.compile None \
  --schedule.max-hours 0.1
```

这保留默认 H=32 想象、k=8 世界模型窗口及机器人控制参数。
减少并行环境数、batch 和 replay 容量会改变数据与优化统计；兼容性验证通过后，
再按显存和实际吞吐逐步提高，不能直接沿用 RTX 5090 的训练时长估计。
更换实验使用新的 `--run-name`，不要用 G1 权重续训 Casbot02。

## 本机实测结果

2026-11-06 已完成：

- PyTorch CUDA 矩阵乘前向、反向，梯度有限。
- Warp CUDA 初始化、GPU 内存分配，mjlab / mujoco-warp 导入。
- `uv sync --locked --group dev` 同步成功；`uv pip check` 检查 156 个已安装包，无依赖冲突。
- 真实 `walk4_subject1` 数据训练：32 环境、replay 64、WM batch 32、
  每次更新 1 个 WM batch、策略 batch 32，默认完整网络，H=32、k=8，
  870 维本体历史 + 690 维参考 → 27 动作；关闭两处可选 `torch.compile`。
- 完成 5 次世界模型 / actor / critic 外层更新、52 个环境控制步（含 40 步 warmup），
  生成 `logs/casbot02_local_cu126_walk4_smoke/ckpt_latest.pt`
  和 `ckpt_0000005.pt`；世界模型、策略、critic 的全部保存参数均有限。
- 用项目的 checkpoint loader 将策略重新加载到 GPU，`870 + 690` 输入得到
  `(1, 27)` 输出，全部输出有限。

训练命令可由上方示例调整为实测配置：`--env.num-envs 32 --env.replay-capacity 64`
加 `--wm-trainer.batch-size 32 --wm-trainer.batches-per-update 1`
和 `--policy-trainer.batch-size 32 --schedule.total-iterations 5`
以及 `--schedule.log-interval 1 --schedule.checkpoint-interval 5`；
移除 `--schedule.max-hours`，并使用全新的 run-name。

短测的 `nse8` 尚未到定期评测时点，因此日志显示 `nan` 占位；没有完成回合的单次区间，
`train_ret` 也会显示 `nan` 占位。它们不能直接当作参数非有限的证据。
本次另行运行的带编译合成数据测试，在首次 PyTorch 编译持续数分钟后主动中断，
未取得通过结果。因此当前建议先关闭 `torch.compile`；这保留训练算法，但影响性能。

以上仅证明本机训练链路可运行，不证明动作已经学会、长期数值稳定或可直接部署真机。

本机项目分区初次检查时仅余约 6 GiB，依赖同步后复查约 20 GiB；可用空间会变化，
数据、编译缓存与多份 checkpoint 需要额外空间。
长训前应整理磁盘或将数据和 `--log-root` 放到空间充足的持久目录。
沙箱内 GPU 探测可能失败；判断本机状态应使用主机终端中的 `nvidia-smi` 和实际 CUDA 测试。
