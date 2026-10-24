# 论文对齐：评测、更新次数、策略目标与 TF32

2026-10-24。依据本地 `SuperMimic.pdf` 的 §4、Eq. 2、Algorithm A.1、Table A.5 和 Fig. 5，修正下面四项。G1 和 Casbot02 共用这些实现。

## 1. 完整动作评分

`eval/benchmark.py` 的主分数改为：

```text
norm_return = 已获得的奖励和 / (0.1 × 完整参考帧数)
```

提前失败后尚未执行的部分按零奖励计入。例如 1000 帧动作只存活 100 步，获得奖励 8，主分数为 0.08，存活段分数为 0.8。保留 `norm_return_survived` 作辅助诊断；汇总的帧加权分数也使用完整帧数。当前环境的 L 帧动作执行 L 步，包含 clip-end timeout 的最后一步；成功动作若不满足此长度契约会报错。

新评测 JSON 标记 `schema_version=2`、`normalization=full_clip`。历史 JSON 不会被自动重写，旧分数应重评后再横向比较。训练日志的 `train_ret` 仍是随机起点训练回合的存活段均值，不是整段 benchmark 主分数。

## 2. 更新计数与学习率

Algorithm A.1 每轮都更新世界模型和策略。现在 `iteration` 表示完成一次外层更新：采集数据 → 世界模型训练（默认 10 个 minibatch）→ 一次 actor 更新及其 critic 拟合。`num_updates` 单独记录 actor 更新次数；默认每外层一次，两者相等。

| 配置 / 记录 | 当前含义 |
|---|---|
| `schedule.total_iterations=320000` | 320,000 次外层更新 |
| `schedule.lr_decay_iterations=250000` | 固定 250,000 次更新的余弦衰减，之后保持 0.03 倍 |
| `schedule.updates_per_env_step=0.4` | 每更新平均采集 2.5 个向量化环境步，交替采集 2、3 步 |
| `schedule.warmup_iterations=40` | 仅采集 40 个环境步填充 replay，不计外层更新；续训需重新填充 |
| `schedule.log_interval=50` | 每 50 次外层更新输出日志 |
| `schedule.checkpoint_interval=4000` | 每 4000 次外层更新保存 latest 和编号权重 |
| `env_steps` | 每个并行环境执行的累计仿真步数，包含 replay 填充 |
| `replay_refill_env_steps` | 其中仅用于初始/恢复填充的步数 |

从头完成默认计划约需 800,040 个向量化仿真步，不能再把旧版的 320,000 仿真步当成同等预算。`max_hours` 仍限制本次启动的墙钟时间；缩短总更新数或设置三小时上限，都不会加速 250k 更新的学习率周期。保存间隔已对齐更新单位；周期性整段评测仍需单独执行 benchmark，不会自动在训练进程内运行。

新 checkpoint 带 `iteration_unit=update`，保存采集余数、环境步数、优化器状态、累计训练时间及精度设置，使用临时文件加原子替换保存。续训保留更新数、学习率阶段及 UTD 余数；replay/仿真状态和随机数状态不保存，因此不承诺逐位重现连续训练轨迹。

**旧 checkpoint 可以继续评测、播放和导出，但默认不能直接续训。** 原版 `iteration` 是环境步、`num_updates` 是外层更新数；显式加 `--legacy-resume` 并使用新的 run-name 才允许迁移。迁移用原 `num_updates` 恢复新更新数，采用新的 LR、策略目标尺度与精度，并写入 `resume_migration`。这不是按新协议从头训练的结果；论文对照实验建议重新开始。

## 3. Eq. 2 策略目标

actor 使用 `loss = -J`，删除额外的 `/H`。`J` 本身保留 batch 平均、时间折扣奖励和、软存活权重及 critic 自举。全局梯度范数 1.0、随后逐元素 0.1 的裁剪顺序保持不变。常数缩放会影响裁剪及 Adam 数值行为，不代表学习速度必然提高 H 倍。

本次对齐指定的 Eq. 2 actor 目标。WM 的窗口平均与 critic 的 minibatch MSE 平均仍是原有实现约定；论文没有给出所有代码级 reduction 细节，没有在本次顺带改变这两项。

## 4. 显式 TF32

`TrainConfig.tf32=True` 默认开启 CUDA float32 矩阵乘法和 cuDNN 的 TF32 许可，在创建/编译网络前设置；权重、状态仍为 float32，不启用 bf16/autocast。`--no-tf32` 仅供独立对照。与请求冲突的 NVIDIA/PyTorch 环境覆盖变量会在启动时拒绝运行。

核查补充：旧 `make_env()` 已经通过 mjlab 默认参数隐式开启 TF32，因此不能仅凭入口缺少设置或模块导入时的默认值，断言旧训练使用完整 FP32。本次统一入口与环境创建的设置，消除隐式覆盖，并补上精度记录；旧 checkpoint 没有完整精度元数据。

`logs/<run>/runtime.json`、checkpoint 的 `precision` 及启动日志记录请求值、后端许可、PyTorch 版本和覆盖变量。这里记录的是后端配置，不表示所有算子都使用 Tensor Core。当前固定版本的 Inductor 仍调用旧精度查询接口，因此统一使用兼容的旧 API；避免混用新旧设置（参见 [PyTorch CUDA 精度说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)）。

## 训练与验证

使用新的实验名，不覆盖历史 `_fixed` 训练及其部署包：

```bash
env -u PYTHONPATH uv run sm-train \
  --robot casbot02 --run-name casbot02_lafan1_proprio_paper_aligned \
  --data.npz-dir data/npz/lafan1_casbot02_codex \
  --env.no-anchor-obs --env.num-envs 4096 \
  --policy-trainer.batch-size 2048 --schedule.max-hours 3
```

上述三小时到达时可能尚未完成 320k 更新；去掉 `--schedule.max-hours 3` 才按总更新数运行到底。Casbot02 的 batch 2048、机器人与 LAFAN1 数据仍是本地适配差异，不等于论文 G1/BONES-SEED 全规模实验。

回归测试在 `tests/test_paper_alignment.py`，覆盖提前失败评分、固定 LR 周期、UTD 分数累积、断点恢复及 actor 时间求和梯度。真实 GPU 短程检查使用：

```bash
env -u PYTHONPATH uv run python scripts/validate_training_numerics.py \
  --run-name casbot02_paper_alignment_smoke \
  --npz-dir data/npz/lafan1_casbot02_codex --iterations 300
```

目录已存在时更换 run-name。短程验证只证明流程和有限值检查通过，不能证明长期收敛或动作质量改善。策略输入输出不变：Casbot02 870 维历史 + 640 维参考 → 27 维动作；新训练完成后需重新导出部署权重。

本机验证结果：CPU 全套 **71 passed、17 skipped**；G1 保存/续训与 Casbot02 兼容测试在实际 GPU 上 **2 passed**。4096 环境 / batch 2048 / H=32 的 Casbot02 完成 **300 次外层更新、790 个环境步**，全部参数及已记录标量有限。TF32 开关分别做了 2 次真实 GPU 更新，确认环境创建后仍保持所选精度。另用临时 50 帧样本确认成功时执行 50 步；用短程权重确认提前失败时主分数按 8196 帧而非 66 个存活步归一化。详见 [验证报告](../outputs/casbot02_hardware/paper_alignment_validation.json)。
