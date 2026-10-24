# Casbot02 无 anchor 训练的数值修复

2026-10-23：旧原始数据与清洗数据实验，均在 iteration 250 的日志首次记录到 NaN。训练日志每 50 步汇总一次，因此首个异常更新在 201–250 步之间；仅凭日志不能锁定具体算子。

## 修复内容

`envs/mdp/rewards.py` 的姿态误差原先调用 mjlab `quat_box_minus`。其四元数对数在零角度时，`torch.where` 未选中的分支仍计算 `0/0`，导致前向误差为 0、反向梯度为 NaN。普通执行及编译执行均能复现。

现在改用本项目的 `rotvec_from_quat`，在小角度分支保留有限分母及级数近似。正常姿态下的前向误差与原定义一致，姿态完全相同和反号四元数的梯度有限。没有删除根部奖励，也没有加回 actor 的根部观测。

世界模型、策略及 critic 的更新共用 `training/numerics.py`：

- loss 非有限时，在 backward 前报错。
- 平均梯度后，以 `error_if_nonfinite=True` 检查梯度范数，在 optimizer.step 前报错。
- 若梯度元素全部有限、仅 FP32 平方求和溢出，则用 FP64 重算全局范数并缩放，再转回原精度；保持同一裁剪阈值。真正的 NaN/Inf 梯度仍立即报错。
- 策略仍按原顺序执行全局范数裁剪，再执行逐元素裁剪；不将异常梯度替换成 0，也不静默跳过异常更新。
- critic 的回报目标也检查有限性。

回归测试覆盖零角度、反号四元数、小角度梯度检查、32 步旋转反传，以及异常 loss/梯度下 Adam 权重和动量保持不变。

首次修复后的 GPU 实验 `casbot02_noanchor_numerics_raw_v1` 越过旧的 250 步发散区间，但在 729 步遇到 critic 的 FP32 范数溢出，被保护逻辑停止；后续版本补充了上述 FP64 回退。该失败记录保留，不能将其记为 1000 步通过。

## 短程 GPU 验证

执行脚本：`scripts/validate_training_numerics.py`。独立实验目录内保存 `numerics_validation.json`。此测试检查数值稳定性，不评估真机可部署性。

```bash
env -u PYTHONPATH uv run python scripts/validate_training_numerics.py \
  --run-name casbot02_noanchor_numerics_raw_v1 \
  --npz-dir data/npz/lafan1_casbot02 --iterations 1000
```

对照实验加 `--anchor-obs` 并更换 run-name。脚本默认 seed=0、4096 个环境、策略 batch 2048，使用目录内全部动作，保持噪声、域随机化和回报采样。学习率仍按 250,000 步衰减，不随短程验证长度加速。

修复后的三组均已完成，模型参数与日志中的关键 loss/梯度均有限：

| 实验 | 数据 | anchor | 完成步数 | 结果 |
|---|---|---|---:|---|
| `casbot02_noanchor_numerics_raw_v2` | 原数据 | 关闭 | 1000 | 通过 |
| `casbot02_anchor_numerics_raw_v2` | 原数据 | 开启 | 1000 | 通过 |
| `casbot02_noanchor_numerics_clean_v2` | 清洗数据 | 关闭 | 1000 | 通过 |

CPU 回归测试：55 项通过、17 项跳过。汇总见 [验证报告](../outputs/casbot02_hardware/training_numerics_validation.json)。短程仍出现过很大的有限想象回报/critic loss 尖峰；没有对长训收敛、整段动作成功率或真机效果作出结论。

## 重新正式训练

旧 NaN 权重仍然无效，必须从头训练，并使用新 run-name：

```bash
env -u PYTHONPATH uv run sm-train \
  --robot casbot02 --run-name casbot02_lafan1_noanchor_fixed \
  --data.npz-dir data/npz/lafan1_casbot02 \
  --env.no-anchor-obs --env.num-envs 4096 \
  --policy-trainer.batch-size 2048 --schedule.max-hours 3
```

若使用清洗数据，改为 `data/npz/lafan1_casbot02_codex`，并更换 run-name，例如 `casbot02_lafan1_proprio_codex_fixed`。清洗数据无需重复裁头或限位。

不要恢复旧的损坏权重。短程数值验证权重也不代表完成训练；常规训练使用默认 320,000 步计划及默认学习率衰减。后续仍需完整动作评测、sim2sim 和独立部署验证。
