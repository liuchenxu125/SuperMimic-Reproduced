# 只有 IMU 和关节状态时的部署路线

用户已确认真机目前没有根部定位。现有 `casbot02_lafan1` checkpoint 开启了 `anchor_obs=True`，其 649 维目标包含实测根部相对位置/姿态，仍不能用于这个传感器配置。

下一步训练 **anchor_obs=False、640 维目标**的新策略。单帧本体观测仍是关节位置/速度、基座角速度、单位重力方向及上一动作，共 87 维；10 帧历史为 870 维。参考动作仍有 10 帧未来目标。推理不需要机器人实际根部位置或线速度，也不需要额外的世界航向反馈。

这条路线不能直接纠正全局位置/航向漂移，实际动作精度需要重新评测。仿真评测可用根部真值计分，但该值不输入策略；真机只保留可测量的倾斜、关节、传感器和时序保护，无法做根部高度/位置误差保护。

## 已准备的代码

- `scripts/train_casbot02_proprio.py`：只训练 `walk4_subject1`，其余噪声、域随机化等沿用当前训练默认设置。默认 4096 环境、策略 batch 2048、最多一小时；训练效果以实际评测为准。
- `rl_controller/supermimic/`：按 checkpoint manifest 决定读取 640 或 649 维目标。只有 640 维权重走 IMU 姿态输入路径；649 维权重仍强制要求根部位姿，不会因改 YAML 而绕过。
- `rl_controller/config/supermimic_walk4_proprio.yaml`：不订阅根部位姿，用 IMU 四元数及安装外参算 base_link 重力方向；没有根部位置估计或伪造的 anchor 输入。
- 导出与离线验证脚本已支持两种目标尺寸。旧权重、旧 ONNX 和旧部署包保留。

当前**尚未训练新的 640 维可用权重**。接口测试使用 `/tmp` 中的随机小网络，仅核对 C++/Python 输入输出、位置/全局 yaw 不变性及 ROS 无定位输入路径，不能用来证明动作效果。

## 开始单动作训练

在主项目根目录运行：

```bash
env -u PYTHONPATH uv run python scripts/train_casbot02_proprio.py
```

先查看配置而不启动训练：

```bash
env -u PYTHONPATH uv run python scripts/train_casbot02_proprio.py --print-config
```

输出写入 `logs/casbot02_walk4_proprio/`，不会覆盖旧训练。脚本拒绝把原来的 649 维权重当作 640 维 checkpoint 恢复。若需要续训，应提供本次新训练产生的 checkpoint：

```bash
env -u PYTHONPATH uv run python scripts/train_casbot02_proprio.py \
  --resume logs/casbot02_walk4_proprio/ckpt_latest.pt
```

## 训练后的仿真与导出

先评测这一条完整动作：

```bash
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/casbot02_walk4_proprio/ckpt_latest.pt \
  --npz-dir data/npz/lafan1_casbot02 --clips walk4_subject1

env -u PYTHONPATH uv run python scripts/export_sim2sim.py \
  --ckpt logs/casbot02_walk4_proprio/ckpt_latest.pt \
  --clips walk4_subject1 --output outputs/casbot02_proprio_sim2sim

env -u PYTHONPATH uv run python scripts/export_hardware.py \
  --bundle outputs/casbot02_proprio_sim2sim \
  --output rl_controller/config/model/supermimic_walk4_proprio

bash rl_controller/scripts/build_supermimic.sh

env -u PYTHONPATH uv run python scripts/validate_hardware.py \
  --sim-bundle outputs/casbot02_proprio_sim2sim \
  --hardware-bundle rl_controller/config/model/supermimic_walk4_proprio \
  --probe rl_controller/build/supermimic_ros/supermimic_probe \
  --from-rest --explicit-pd \
  --report outputs/casbot02_hardware/proprio_walk4_full.json
```

这些动作测试应重新运行；原 649 维模型的成功结果不能算作新策略的验证。

## 新权重验证后的 dry-run

```bash
SUPERMIMIC_CONFIG="$PWD/rl_controller/config/supermimic_walk4_proprio.yaml" \
  bash rl_controller/scripts/run_supermimic.sh
```

启动日志应显示 `goal=640 root_pose=not_required dry_run=true`。IMU 必须提供有效姿态四元数和角速度；四元数为 IMU 到世界的旋转，安装外参 `imu_to_base_quat_wxyz` 为 IMU 到 base_link 的旋转。只使用姿态算重力，不使用把 covariance 当 RPY 的旧逻辑。若 IMU 装在会相对 base_link 转动的躯干上，还需按腰部运动处理外参。

真实命令仍使用有力矩限制的 effort-only 路径，其底层语义和保护要求见本地 `rl_controller/SUPERMIMIC_DEPLOYMENT.md`。`rl_controller` 是独立内网工程，不随本公开仓库分发，相关构建/ROS 命令需要先具备该工程和 SDK。在新权重完成训练和验证前，proprio 配置指向的模型目录为空，启动会报模型不存在，而不会回退使用旧模型。
