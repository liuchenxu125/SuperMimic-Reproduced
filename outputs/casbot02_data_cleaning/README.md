# Casbot02 训练副本数据清洗报告

处理目录：`/home/casbot/Desktop/SuperMimic-Reproduced/data/npz/lafan1_casbot02_codex`
备份目录：`/home/casbot/Desktop/SuperMimic-Reproduced/data/npz/lafan1_casbot02_codex_backup_20261023_135229`

仅修改指定的 NPZ 副本；原版 NPZ、原始 CSV 和旧模型均未修改。40 条动作均保留。

## 结果

| 指标 | 处理前 | 处理后 |
|---|---:|---:|
| 动作数 | 40 | 40 |
| 总帧数（50 FPS） | 441,138 | 441,093 |
| 超出模型硬限位的帧数（1e-6 rad 容差） | 116,810（26.48%） | 0 |

9 条动作各去掉开头 0.1 秒（5 帧），总共删除 45 帧。其余动作长度不变。限位处理修改 165,198 个关节角采样值，最大修正 0.39250004 rad。

## 裁剪起始帧的 9 条动作

| 动作 | 旧评测失败时间（秒） | 删除帧数 |
|---|---:|---:|
| `dance2_subject4` | 0.10 | 5 |
| `fallAndGetUp1_subject1` | 0.52 | 5 |
| `run2_subject1` | 0.08 | 5 |
| `sprint1_subject4` | 0.10 | 5 |
| `walk1_subject1` | 0.16 | 5 |
| `walk1_subject2` | 0.22 | 5 |
| `walk3_subject1` | 0.10 | 5 |
| `walk3_subject2` | 0.26 | 5 |
| `walk3_subject4` | 0.14 | 5 |

失败时间来自旧权重、原数据的历史评测，用于选择需要裁头的动作；不是清洗后或新权重的成绩。

## 关节修正

| 关节 | 修改采样值数 | 硬限位（rad） |
|---|---:|---|
| `leg_l4_joint` | 752 | [0.0, 2.5307] |
| `leg_r4_joint` | 1,463 | [0.0, 2.5307] |
| `upper_left_4_joint` | 88,576 | [-1.8675, 0.0] |
| `upper_right_4_joint` | 74,407 | [-1.8675, 0.0] |

按模型硬限位逐点裁剪。为避免 float32 保存时向外舍入，使用限位内可表示的端点；没有另加平滑、动作删选或电机速度裁剪。训练的 0.9 软限位与这里的物理硬限位不同，软限位惩罚不保证为零。

## 数据一致性与验证

- 裁头后重算根部线速度、SO(3) 角速度；关节裁剪后重算关节速度。
- 重算全部刚体位置、姿态、线速度、角速度，避免裁剪后的关节姿态与身体目标矛盾。
- 全部帧的关节角均检查在精确硬限位内；全部关节及根部导数检查通过。
- 每条动作独立抽查 8 帧，以 MuJoCo FK 和 mj_objectVelocity 核对全部 30 个刚体的位姿/速度。
- 写入后 40 个文件 SHA256 与已验证的暂存文件一致；备份与原版 NPZ 的 SHA256 一致。
- 保留裁剪后原始根部位置和所有非时序元数据；根部四元数进行数值归一化。
- 两个数据修复单元测试通过，覆盖裁头、限位、导数、FK 和浮点边界。

原来约 26.6% 的统计来自 30 FPS CSV；本报告的 26.48% 来自重采样后的 50 FPS NPZ。副本里的 source_audit.json 仍描述原始 CSV，清洗后以 cleaning_report.json 为准。

本次未训练或重新评测策略。裁剪能消除关节超限和已选开头的数据，但不保证长动作全部成功，也不是重新运动重定向或动力学可行性优化。

## 使用清洗数据从头训练无根部位置策略

```bash
env -u PYTHONPATH uv run sm-train \
  --robot casbot02 --run-name casbot02_lafan1_proprio_codex \
  --data.npz-dir data/npz/lafan1_casbot02_codex \
  --env.no-anchor-obs --env.num-envs 4096 \
  --policy-trainer.batch-size 2048 --schedule.max-hours 3
```

不要使用旧 649 维权重的 --resume，也不要继续使用旧的 packed library 或导出包。新训练应重新读取本目录的 NPZ。
