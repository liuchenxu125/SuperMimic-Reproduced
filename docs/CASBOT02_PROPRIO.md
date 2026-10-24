# 只有 IMU 和关节状态时的部署路线

用户已确认真机目前没有根部定位。现有 `casbot02_lafan1` checkpoint 开启了 `anchor_obs=True`，其 649 维目标包含实测根部相对位置/姿态，仍不能用于这个传感器配置。

部署使用 **anchor_obs=False、640 维目标**的策略。单帧本体观测是关节位置/速度、基座角速度、单位重力方向及上一动作，共 87 维；10 帧历史为 870 维。参考动作有 10 帧未来目标。推理不需要机器人实际根部位置或线速度，也不需要额外的世界航向反馈。640 维目标中的参考高度与速度来自离线动作文件，仍予以保留。

这条路线不能直接纠正全局位置/航向漂移，实际动作精度需要重新评测。仿真评测可用根部真值计分，但该值不输入策略；真机只保留可测量的倾斜、关节、传感器和时序保护，无法做根部高度/位置误差保护。

## 当前已同步的部署包（2026-10-24）

- 训练：`casbot02_lafan1_proprio_codex_fixed`，固定使用已评测的 `ckpt_0280000.pt`，不是随训练变化的 latest。
- 权重 SHA256：`05c847907b66affdd578c10b8c2e557d9a781d90f4ed072d4ea47cc78253485d`。
- 模型目录：`rl_controller/config/model/supermimic_walk4_proprio/`，包含真实训练权重导出的 `actor.onnx`、清洗数据参考 `motion.bin` 和电机/接口 `manifest.json`。
- ONNX 输入：`obs_hist[1,870]`、`goal[1,640]`；输出：`action[1,27]`。总输入 1510 维；目标先编码为 32 维 latent，再与历史观测组成 902 维 decoder 输入。

| 输入/输出 | 单帧组成 | 维度 |
|---|---|---|
| 本体历史 | q−q0 27、dq 27、基座角速度 3、投影重力 3、上一原始动作 27；按训练的 term-major 顺序打包 10 帧 | 87×10=870 |
| 未来参考 | 参考 q−q0 27、dq 27、投影重力 3、body 角速度 3、heading 线速度 3、高度 1；目标从 f+1 开始取 10 帧 | 64×10=640 |
| actor 动作 | 27 个归一化动作，转为 `q_des=q0+scale*action` | 27 |
| 电机命令 | 复用训练电机参数，以 1 kHz 计算并逐电机限幅 `tau=Kp*(q_des-q)-Kd*dq`；ROS effort-only | 27 |

真实权重验证结果：

- C++ ONNX / Python 输入对照：870 维历史与 640 维目标的最大绝对差均为 0。
- C++ ONNX + 原生 MuJoCo implicit PD：完整 8196 步、163.92 秒，无高度/倾斜失败。
- 从零初速度起步 + C++ ONNX + 1 kHz 显式限矩 PD：完整 8196 步、163.92 秒，无失败；actor 最大差 `1.34e-5`，本机往返延迟 p99 约 2.36 ms。
- 使用真实 ONNX 的隔离 ROS 测试已通过 dry-run 与合成 effort 路由两条路径，检查无根部位姿订阅、错误接口拦截、缺失/失效传感器拦截和超时阻尼；没有向真实 `/motion/*` 话题发命令。

闭环报告：[implicit PD](../outputs/casbot02_hardware/proprio_280000_walk4_onnx_full.json)、[从静止显式 PD](../outputs/casbot02_hardware/proprio_280000_walk4_effort_from_rest.json)。这两个报告的 MuJoCo 根部真值仅用于初始化和评分，送入 C++ 的实际根部位置为 NaN。

长时全局路径仍有明显漂移：显式 PD 报告中的平均根位置误差约 9.89 m，关节 RMSE 约 0.1006 rad。完成动作不代表路径准确；以上是本机 CPU 仿真和隔离 ROS 验证，尚未进行真机或 ARM 计算板实测。

当前包可直接构建并以默认 dry-run 启动：

```bash
bash rl_controller/scripts/build_supermimic.sh
bash rl_controller/scripts/run_supermimic.sh
```

启动日志会打印 `obs=870 goal=640 total_input=1510 actions=27`、`checkpoint_iteration=280000`、权重 SHA256、`root_pose=not_required` 和 `dry_run=true`。构建和启动不会自动切换机器人模式或开始动作。

## 已准备的代码

- `scripts/train_casbot02_proprio.py`：只训练 `walk4_subject1`，其余噪声、域随机化等沿用当前训练默认设置。默认 4096 环境、策略 batch 2048、最多一小时；训练效果以实际评测为准。
- `rl_controller/supermimic/`：按 checkpoint manifest 读取 640 或 649 维目标。640 维路径只用 IMU 姿态/角速度和关节状态，根部位置标记为 NaN，不参与验证、参考对齐、策略或 PD 计算。
- `rl_controller/config/supermimic_walk4_proprio.yaml`：`require_anchor_obs: false`；不订阅根部位姿，用 IMU 四元数及安装外参算 base_link 重力方向。配置与模型的 anchor 设置不匹配时拒绝启动。
- `rl_controller/scripts/run_supermimic.sh`：默认选择上述无定位配置，仍默认 dry-run。
- 导出前检查策略全部参数和缓冲区、零输入及随机输入输出是否有限；C++ 启动预热 ONNX 并检查输出，早于电机命令发布者的创建。
- `scripts/validate_hardware.py`：无定位模型的 C++ 闭环输入将根部位置替换为 NaN；MuJoCo 真值仅用于仿真初始化和评分。

## 历史损坏权重（2026-10-23 检查）

`logs/casbot02_lafan1_proprio_codex/` 已有 640 维训练结果，但从 iteration 4000 到 36000 的全部 9 个定期 checkpoint，以及 `ckpt_latest.pt`，策略均有 **20 个非有限张量、8,685,883 个非有限值**。本次尝试导出被拦截，没有生成可部署的新模型包。完整检查见 [权重检查报告](../outputs/casbot02_hardware/proprio_checkpoint_audit.json)。不能从这些损坏 checkpoint 继续恢复以期待自动修复，需定位训练发散原因并取得有效权重。

训练数值修复及重新开始的命令见 [TRAINING_NUMERICS.md](TRAINING_NUMERICS.md)。旧 NaN checkpoint 不会因源码修复而恢复；数值验证产生的短程模型也不代表动作质量已通过部署评测。

早期接口验证使用 `/tmp` 中的随机小网络，不保存为真机模型。结果见 [接口测试报告](../outputs/casbot02_hardware/proprio_deployment_interface_test.json)：检查 C++/Python 输入输出、根部位置为 NaN、全局 yaw 不变性、无根部订阅、传感器超时阻尼和错误模型拦截。当前真实权重的独立验证见上节，不能与历史随机网络测试混淆。

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

## 更新部署权重时的仿真与导出

以下以清洗数据重新训练的 `casbot02_lafan1_proprio_codex_fixed` 目录为例；旧 `casbot02_lafan1_proprio_codex` 的 NaN 权重仍会被拒绝。必须先取得有效 checkpoint，再执行导出和真实权重闭环验证。

```bash
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/casbot02_lafan1_proprio_codex_fixed/ckpt_latest.pt \
  --npz-dir data/npz/lafan1_casbot02_codex --clips walk4_subject1

env -u PYTHONPATH uv run python scripts/export_sim2sim.py \
  --ckpt logs/casbot02_lafan1_proprio_codex_fixed/ckpt_latest.pt \
  --npz-dir data/npz/lafan1_casbot02_codex \
  --clips walk4_subject1 --output outputs/casbot02_proprio_codex_sim2sim

env -u PYTHONPATH uv run python scripts/export_hardware.py \
  --bundle outputs/casbot02_proprio_codex_sim2sim \
  --output rl_controller/config/model/supermimic_walk4_proprio

bash rl_controller/scripts/build_supermimic.sh

env -u PYTHONPATH uv run python scripts/validate_hardware.py \
  --sim-bundle outputs/casbot02_proprio_codex_sim2sim \
  --hardware-bundle rl_controller/config/model/supermimic_walk4_proprio \
  --probe rl_controller/build/supermimic_ros/supermimic_probe \
  --from-rest --explicit-pd \
  --report outputs/casbot02_hardware/proprio_walk4_full.json
```

更新为其他 checkpoint 后应重新运行这些动作测试；原 649 维模型的成功结果不能算作新策略的验证。当前已验证包固定为上节的 28 万步版本，导出新权重时应保留明确的版本记录。

## 当前默认 dry-run

```bash
bash rl_controller/scripts/run_supermimic.sh
```

启动日志应同时显示 `goal=640`、`root_pose=not_required` 和 `dry_run=true`。IMU 必须提供有效姿态四元数和角速度；四元数为 IMU 到世界的旋转，安装外参 `imu_to_base_quat_wxyz` 为 IMU 到 base_link 的旋转。只使用姿态算重力，不使用把 covariance 当 RPY 的旧逻辑。若 IMU 装在会相对 base_link 转动的躯干上，还需按腰部运动处理外参。

控制仍为 50 Hz 策略 + 1 kHz 限矩 PD，默认先 6 秒准备到参考首帧，再由 `/supermimic_controller/start` 启动 10 秒预览。有效模型完成验证后，才使用 `--ros-args -p dry_run:=false` 开启真实命令。停止、模式 7 门控、关节限位、倾斜和传感器时序保护继续生效；无定位路径没有实测根部高度/位置误差保护。

真实命令仍使用有力矩限制的 effort-only 路径，其底层语义和保护要求见本地 `rl_controller/SUPERMIMIC_DEPLOYMENT.md`。`rl_controller` 是独立内网工程，不随本公开仓库分发，相关构建/ROS 命令需要先具备该工程和 SDK。当前 proprio 模型目录已生成；若目录缺失或接口不匹配，启动拒绝运行，不会回退使用旧模型。
