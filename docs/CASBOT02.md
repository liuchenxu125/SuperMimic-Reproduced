# Casbot02 27 DOF 适配

## 模型和控制参数

模型与 meshes 从本机 `~/Desktop/ProtoMotions/protomotions/data/assets/casbot02/`
复制到 `src/supermimic/robots/assets/casbot02/`；源项目无需运行，也无需安装其依赖。
`copied_files_sha256.json` 用于核对复制资产与源文件完全一致。
电机参数和默认站姿从 `ProtoMotions/protomotions/robot_configs/casbot02.py` 提取到
`motor_parameters.json`，内含源文件路径及 SHA-256。

- 27 个动作、30 个刚体、总质量 50.653931 kg；头部保持模型中的固定连接。
- root、anchor、COM 随机化身体均为 `base_link`；14 个跟踪身体沿用 ProtoMotions 配置。
- 每个电机使用原 stiffness、damping、effort limit 和 armature；被动阻尼、摩擦为零。
- 隐式 PD：`目标关节角 = 默认站姿 + action × (0.25 × effort_limit / stiffness)`。
- 200 Hz 物理仿真、4 个子步、50 Hz 控制，无电机延迟。
- 原模型的碰撞几何和关节硬限位保留，软限位系数 0.9。
- `velocity_limit` 保留在参数表并用于动作数据审计。MuJoCo 的隐式位置执行器没有
  Isaac Lab 对应的硬速度约束，这些速度值未额外作为速度钳位或力矩速度曲线；
  PD、力矩限位、armature 则逐执行器映射并在测试中核对。

配置名为 `casbot02`。G1 仍为默认机器人，原 `e7` 权重可继续加载。
Casbot02 的关节和网络尺寸不同，需要从头训练；跨机器人 checkpoint 恢复会报错。

| 数量 | G1 | Casbot02 |
| --- | ---: | ---: |
| 关节/动作 | 29 | 27 |
| qpos / qvel | 36 / 35 | 34 / 33 |
| 世界模型状态 | 71 | 67 |
| 单帧观测 / 10 帧历史 | 93 / 930 | 87 / 870 |
| 单帧参考 / 10 帧目标 | 68 / 680 | 64 / 640 |
| 仿真参数 xi | 33 | 31 |

`--env.anchor-obs` 给目标追加 9 维根部相对位姿，Casbot02 的目标变为 649 维。
这是显式启用的跟踪辅助观测，偏离论文的纯本体观测口径。

## LAFAN1 数据

源目录为 `~/Downloads/lafan1_casbot02_27dof_csv/csv/`：无表头，列为 xyz（米）、
根四元数 xyzw、按 MJCF 关节顺序排列的 27 个关节角（弧度）。
用户已确认导出保持 LAFAN1 原始帧率 **30 FPS**，当前转换设置与之相符。

```bash
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --robot casbot02 \
  --input-dir ~/Downloads/lafan1_casbot02_27dof_csv/csv \
  --output-dir data/npz/lafan1_casbot02 \
  --input-fps 30 --output-fps 50 --workers 4
```

本地已生成 40 条 NPZ，共 441,138 帧，约 2.45077 小时。
原始 CSV 没有修改；重采样使用位置/关节线性插值和旋转 SLERP，速度通过差分计算。

质量报告为 `data/npz/lafan1_casbot02/source_audit.json`：

- 264,704 个原始帧中，70,342 帧（约 26.6%）至少一个关节超出模型硬限位，最大超出
  0.3925 rad。明显超限来自左右肘 `upper_left_4_joint` / `upper_right_4_joint`：
  复用模型下限 -1.8675 rad，数据最低 -2.26 rad；膝部另有 0.00003 rad 的细小差异。
  报告列出每条动作、每个关节的超限情况。
- 按 30 FPS 计算，7 条动作的第一段根位移对应速度超过 5 m/s。
- 当前转换保留所有帧和关节角，不自动裁剪、平滑或删除异常帧。
  源参考的超限动作无法完全由有限位机器人复现，训练质量需结合这些数据问题判断。

可重新审计：

```bash
env -u PYTHONPATH uv run python scripts/data/audit_motion_csv.py \
  --robot casbot02 --input-dir ~/Downloads/lafan1_casbot02_27dof_csv/csv \
  --input-fps 30 --out data/npz/lafan1_casbot02/source_audit.json
```

## 训练

```bash
env -u PYTHONPATH uv run sm-train \
  --robot casbot02 --run-name casbot02_lafan1 \
  --data.npz-dir data/npz/lafan1_casbot02 \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 \
  --env.anchor-obs --schedule.max-hours 3
```

上述规模需检查可用显存；`num-envs`、策略批大小和 replay 容量都可独立调小。
不添加 `--resume logs/e7/ckpt_latest.pt`：e7 是 G1 的模型。

## 播放与评测

机器人类型、网络尺寸和 anchor 设置会从 checkpoint 的配置自动恢复。

```bash
env -u PYTHONPATH MUJOCO_GL=glfw uv run python scripts/play.py \
  --ckpt logs/casbot02_lafan1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1_casbot02 --clips walk1_subject1 --viewer native

env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/casbot02_lafan1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1_casbot02
```

## 验证

```bash
env -u PYTHONPATH uv run pytest tests/test_casbot02.py -q
```

测试核对资产 SHA-256、27 个执行器的增益/力矩/armature、与 MuJoCo 的 FK 和速度对拍、
固定头部传播、梯度、完整/紧凑动作库一致性、67/870/640 维输入输出、
启用 anchor 和 torch.compile 的 GPU 短训练及 checkpoint 加载。
短训练仅验证流程，不代表已获得可用的动作跟踪策略。

本机 RTX 4090 D 的完整 GPU 回归测试为 **38 passed**（221 秒），包括上述 Casbot02
测试、真实 LAFAN1 仿真/采样、G1 训练/恢复和世界模型拟合测试。
