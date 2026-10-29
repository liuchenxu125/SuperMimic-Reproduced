# 只有 IMU 和关节状态时的部署路线

> **SDK 2.2.37 接口已修正：** 真机 runner 采用目标位置＋整数 Kp/Kd（effort=0），并通过模式服务和实际增益反馈检查启动条件。下文的显式 PD 仿真结果是历史模型验证，不是新 SDK 控制链路验证。当前操作以 [SDK 部署说明](../rl_controller/SUPERMIMIC_DEPLOYMENT.md) 开头为准。

> 2026-10-27 更新：默认 walk1 部署配置已开启自动启动。先启动控制器，再从非 7 模式切入模式 7，
> 自动等待新鲜数据 0.5 秒、准备 2 秒、平滑接管 1 秒，然后推进参考动作；每次进程只尝试一次。
> 启动时已经在模式 7 不会直接触发，需先退出再重新切入。stop、fault、退出模式或正常完成后
> 不会自动重新启动。默认 dry-run 仍不发布真实命令；用 `--ros-args -p dry_run:=false` 启用输出。
> 如需使用本文手动 prepare/start 流程，启动时加 `-p auto_start:=false`。完整规则见
> [部署说明](../rl_controller/SUPERMIMIC_DEPLOYMENT.md)。


用户已确认真机目前没有根部定位。现有 `casbot02_lafan1` checkpoint 开启了 `anchor_obs=True`，其 649 维目标包含实测根部相对位置/姿态，仍不能用于这个传感器配置。

部署使用 **anchor_obs=False、640 维目标**的策略。单帧本体观测是关节位置/速度、基座角速度、单位重力方向及上一动作，共 87 维；10 帧历史为 870 维。参考动作有 10 帧未来目标。推理不需要机器人实际根部位置或线速度，也不需要额外的世界航向反馈。640 维目标中的参考高度与速度来自离线动作文件，仍予以保留。

这条路线不能直接纠正全局位置/航向漂移，实际动作精度需要重新评测。仿真评测可用根部真值计分，但该值不输入策略；真机只保留可测量的倾斜、关节、传感器和时序保护，无法做根部高度/位置误差保护。

## 当前部署：walk1_subject2（2026-10-26）

**当前已替换为新训练模型：** `casbot02_lafan1_proprio_paper_kpkd/ckpt_0048000.pt`，对应 48,000 次外层更新。Kp、Kd、力矩上限、action_scale 和 q0 与此次新训练一致。40 条动作各从第 0 帧进行一次整段 GPU 仿真评测，30 条未触发高度/倾斜失败终止，成功率为 75%；整段归一化回报均值为 0.4265。`walk1_subject2` 完成全部 13061 帧（261.22 秒），归一化回报为 0.5219。此处成功指完成片段，不能等同于跟踪误差达标或多次试验成功率。

新模型另已通过从动作首帧姿态、零初速度开始的 C++ ONNX / 1 kHz 显式限矩 PD 前 10 秒闭环，观测/目标输入差为 0，actor 最大绝对差约 `5.96e-6`。隔离 ROS 的接收、模式门控和过渡流程检查通过，ARM64 安装包与源部署包的模型、动作、manifest 和配置哈希一致。完整动作的原生 C++ 闭环、完整起步回位物理闭环及真机尚未验证。部署为固定快照，不会自动热更新。

结果：[整段评测](../outputs/casbot02_kpkd_048000/benchmark.json)、[10 秒闭环报告](../outputs/casbot02_kpkd_048000/walk1_subject2_cpp_effort_10s.json)、[同步与备份记录](../outputs/casbot02_kpkd_048000/deployment_sync.json)。此次替换前的 24,000 次更新部署包已保存在 `outputs/casbot02_deployment_backups/walk1_before_kpkd_048000_20261026_223111/`。旧 26 万步策略的验证结果属于历史记录，不用于评价当前模型。

| 电机组 | 关节（左右相同） | Kp | Kd | action_scale | 力矩上限 Nm |
|---|---|---:|---:|---:|---:|
| 腿重型 | 1、2、4 | 276.348923 | 17.592919 | 0.108558411 | 120 |
| 腿轻型 | 3、5、6 | 114.487411 | 7.288495 | 0.174691696 | 80 |
| 腰 | yaw | 96.722123 | 6.157522 | 0.155083444 | 60 |
| 臂重型 | 1、2、4 | 130.278778 | 8.293805 | 0.076758472 | 40 |
| 臂轻型 | 3、5、6、7 | 96.722123 | 6.157522 | 0.077541722 | 30 |

源文件的 articulation 仅列出 22 个关节。本项目保持 27 DOF，腰采用该文件单独定义的腰电机，腕部第 6、7 轴按轻型臂电机补齐。除控制参数与推导 action_scale 所需的力矩上限外，其余惯量、摩擦和延迟配置保持原项目设置。当前新模型的 q0 也已同步为训练使用的新默认位置，没有修改网络权重来转换旧 q0。

新参数闭环报告：[HANDOFF 参数](../outputs/casbot02_hardware/proprio_260000_walk1_subject2_handoff_effort_10s.json)。原部署包备份在本地 `outputs/casbot02_handoff_parameters/before/hardware_walk1/`，来源 SHA256、映射和参数对照在 `outputs/casbot02_handoff_parameters/parameter_sync.json`。当前 ARM64 安装包已包含新参数。

## 动作前后过渡（2026-10-26）

参考 [MorphMimic 的 CuedClipReference](https://github.com/lzyang2000/morphmimic/blob/main/deploy/sim2sim/policy_node.py)：该项目在 `--from-stand` 模式中对参考序列做站立到动作、动作到站立的插值；姿态由 `teleop.blend_states` 做四元数球面插值。

本地采用控制器目标层的过渡，使用现有 27 关节接口：

1. `prepare`：`prepare_pose: default`，用 2 秒五次缓动从当前实测姿态到训练默认位置。当前 `prepare_joint_pos` 显式记录 27 个模型关节的默认位置；不配置此表时使用部署 manifest 的 q0。目标按软限位约束；期间参考时钟不前进。
2. `start`：进入 `HANDOFF`，用 `entry_blend_seconds: 1.0` 将准备目标逐渐交给策略。所有未来参考帧固定为动作第 `start_frame` 帧姿态，参考关节/根部速度置零；策略仍按 50 Hz 使用实时观测、更新历史和上一动作。接管完成后进入 `RUN`，不重置历史，按训练的 f+1 时序推进动作参考。
3. 预演超时、片段结束或调用 `finish`：进入 `RETURN`，用 `return_seconds: 3.0` 从最后实际使用的位置目标回到 **prepare 前记录的实测姿态（按软限位约束）**；回位期间停止策略推理，随后进入 `HOLD`，用 PD 保持该姿态。
4. `stop` 立即进入阻尼；故障及模式退出也立即中断过渡。`HOLD` 的静态 PD 保持不能视为站立策略或平衡保证。

这与 MorphMimic 的参考序列插值方式不同，不会给当前策略输入其 240 维参考格式，也不是动作之间的动态无缝切换。接管时长不消耗 `preview_seconds` 的动作播放预算。过渡权重在固定端点处具有零一阶、二阶导数；策略端点仍按 50 Hz 更新，不能宣称整段执行目标处处连续可微。不配置 entry/return 时长时保持即时接管/结束行为，未指定 prepare_pose 时兼容动作首帧准备姿态。

手动平滑结束和保持：

```bash
ros2 service call /supermimic_controller/finish std_srvs/srv/Trigger '{}'
```

立即停止策略、进入阻尼仍使用：

```bash
ros2 service call /supermimic_controller/stop std_srvs/srv/Trigger '{}'
```

已通过插值端点/单调性/时长校验、50 步冻结参考和历史连续性检查，以及隔离 ROS 域 186 中的 dry-run 和合成 effort 话题测试：默认准备目标、HANDOFF 保持第 0 帧、接管时长独立于播放预算、正常 RETURN/HOLD、保持姿态的恢复力矩、手动 finish、RETURN 中 stop、传感器故障、模式退出。本机与 ARM64 编译安装通过；这些流程检查没有执行真机控制，也不能证明完整起步/接管/回位过程的物理平衡。

## walk1 策略与更新前参数的历史验证记录

根据旧版训练曲线，20–24 万环境步进入平台期，24–26 万步处于较好的区间。26 万步后训练回报轻微波动，没有持续退化。本次按用户要求停止逐个模型精确筛选，选择 `casbot02_lafan1_proprio_codex_fixed/ckpt_0260000.pt` 作为平台期代表，不宣称它是精确最优。曲线与窗口统计见本地 `outputs/casbot02_old_selection/convergence.md`。

- 默认配置：`rl_controller/config/supermimic_walk1_subject2_proprio.yaml`；默认启动脚本已切换到此配置。
- 部署包：`rl_controller/config/model/supermimic_walk1_subject2_proprio/`，包含重新导出的 `actor.onnx`、`motion.bin`、`manifest.json`。
- 动作：完整 `walk1_subject2`，13061 帧、50 Hz、261.22 秒；数据未裁剪。现有 `preview_seconds: 10.0` 保持不变。
- 接口：870 维历史 + 640 维目标 → 27 动作，根部位置不输入策略。
- 已通过从静止起步、C++ ONNX、1 kHz 显式限矩 PD 的前 10 秒闭环；没有高度/倾斜失败。C++/Python 的观测、目标输入差均为 0，actor 最大差约 `8.58e-6`。这是 10 秒验证，不是完整 261.22 秒或真机验证。

验证报告：[walk1_subject2 前 10 秒](../outputs/casbot02_hardware/proprio_260000_walk1_subject2_effort_10s.json)。构建并以默认 dry-run 启动：

```bash
bash rl_controller/scripts/build_supermimic.sh
bash rl_controller/scripts/run_supermimic.sh
```

## 历史部署包：walk4_subject1（2026-10-24）

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

历史 walk4 包可通过显式配置以 dry-run 启动：

```bash
bash rl_controller/scripts/build_supermimic.sh
SUPERMIMIC_CONFIG="$PWD/rl_controller/config/supermimic_walk4_proprio.yaml" \
  bash rl_controller/scripts/run_supermimic.sh
```

启动日志会打印 `obs=870 goal=640 total_input=1510 actions=27`、`checkpoint_iteration=280000`、权重 SHA256、`root_pose=not_required` 和 `dry_run=true`。构建和启动不会自动切换机器人模式或开始动作。

## 已准备的代码

- `scripts/train_casbot02_proprio.py`：只训练 `walk4_subject1`，其余噪声、域随机化等沿用当前训练默认设置。默认 4096 环境、策略 batch 2048、最多一小时；训练效果以实际评测为准。
- `rl_controller/supermimic/`：按 checkpoint manifest 读取 640 或 649 维目标。640 维路径只用 IMU 姿态/角速度和关节状态，根部位置标记为 NaN，不参与验证、参考对齐、策略或 PD 计算。
- `rl_controller/config/supermimic_walk4_proprio.yaml`：`require_anchor_obs: false`；不订阅根部位姿，用 IMU 四元数及安装外参算 base_link 重力方向。配置与模型的 anchor 设置不匹配时拒绝启动。
- `rl_controller/scripts/run_supermimic.sh`：默认选择 `supermimic_walk1_subject2_proprio.yaml`，仍默认 dry-run。
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

更新为其他 checkpoint 后应重新运行这些动作测试；原 649 维模型的成功结果不能算作新策略的验证。以上是历史 walk4 导出示例；当前 walk1_subject2 的 26 万步包及其验证范围见本文首节。

## 当前默认 dry-run

```bash
bash rl_controller/scripts/run_supermimic.sh
```

启动日志应同时显示 `goal=640`、`root_pose=not_required` 和 `dry_run=true`。IMU 必须提供有效姿态四元数和角速度；四元数为 IMU 到世界的旋转，安装外参 `imu_to_base_quat_wxyz` 为 IMU 到 base_link 的旋转。只使用姿态算重力，不使用把 covariance 当 RPY 的旧逻辑。若 IMU 装在会相对 base_link 转动的躯干上，还需按腰部运动处理外参。

控制仍为 50 Hz 策略 + 1 kHz 限矩 PD。当前默认配置在调用 `prepare` 后，用 2 秒到默认准备姿态；READY 后调用 `/supermimic_controller/start`，用 1 秒固定参考接管，再播放 10 秒。有效模型完成验证后，才使用 `--ros-args -p dry_run:=false` 开启真实命令。停止、模式 7 门控、关节限位、倾斜和传感器时序保护继续生效；无定位路径没有实测根部高度/位置误差保护。

### AMP SDK 接收路径与手柄模式

部署接收同时订阅 `/motion/joint_state`（`sensor_msgs/msg/JointState`）与 `/motion/debug/joint_state`（`crb_ros_msg/msg/JointStateData`），分别由 `standard_joint_topic` 和 `joint_topic` 配置。与 AMP 一样按最近接收时间选择来源；每一帧仍须完整包含模型控制的 27 个关节，且时间戳有效。两路独立校验和缓存，不拼接不完整帧；某一路损坏或过期时可使用另一路的完整新鲜帧。IMU 和模式仍使用 `/motion/imu` 与 `/motion/current_mode`。

使用 4 线程 ROS 执行器，传感器订阅和 1 kHz 控制定时器属于独立回调组。接收计数在所有机器人模式下更新；状态日志分别列出 `joint_debug_rx`、`joint_standard_rx`、`imu_rx`、`mode_rx` 和 `joint_source`。ROS 端接收不以模式 7 为条件；若底层 SDK 在调试模式才实际发布数据，仍须由手柄启用该底层状态，话题存在不能证明数据已开始发送。

手柄切到 SDK 的模式 7，含义是允许 RL 控制。AMP 随模式进入其站立/行走流程；SuperMimic 保留分步启动：程序启动 → 手柄进入模式 7 → `prepare` → READY → `start`。进入模式 7 本身不会自动播放动作。`dry_run=true` 不向电机话题发布命令；`dry_run=false` 在模式 7 的 IDLE 阶段发送有力矩上限的阻尼，prepare 后才发送姿态跟踪力矩。退出模式 7 立即停止发布，重新进入不会自动重播。

实际命令复用 AMP 已有的 `/motion/debug/joint_cmd` 接口，输出为 27 关节的限矩 PD 力矩，消息 kp/kd 为零。没有同时向 `/motion/joint_cmd` 下发第二份命令，避免底层重复执行；也不订阅 AMP 的速度指令 `/motion/cmd_vel`，当前策略跟踪包内参考动作。

真实命令仍使用有力矩限制的 effort-only 路径，其底层语义和保护要求见本地 `rl_controller/SUPERMIMIC_DEPLOYMENT.md`。`rl_controller` 是独立内网工程，不随本公开仓库分发，相关构建/ROS 命令需要先具备该工程和 SDK。当前 proprio 模型目录已生成；若目录缺失或接口不匹配，启动拒绝运行，不会回退使用旧模型。
