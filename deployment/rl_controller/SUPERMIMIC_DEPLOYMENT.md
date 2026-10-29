# Casbot02 SuperMimic 真机部署

## 当前部署权重（2026-10-29）

已切换为 `casbot02_lafan1_proprio_paper_kpkd_primitive_data/ckpt_0068000.pt` 的固定快照。
输入仍为 870 维本体历史 + 640 维参考，输出 27 维；首次自动启动为 `walk1_subject2`。
主包与 15 条动作库共用新 actor，保留整段播放；`prepare_seconds: 0.0` 跳过准备姿态，从当前实测关节位置开始 1 秒固定参考接管、X 的 0.8 秒连续策略切换和 B 回位。
入口 YAML 和主包目录继续叫 `supermimic_walk1_subject2_proprio`，包内 `manifest.json` 的默认 `clip` 为 `walk1_subject2`，不用修改原启动命令。

68000 次权重已通过默认踏步从静止起步的 10 秒 C++ ONNX / 原生隐式 PD 闭环，
观测和参考与 Python 一致；无需提供实测根部位置。
菜单中 15 组相邻切换均通过短测（先运行 5 秒、桥接 0.8 秒、再运行最多 10 秒；短动作到末帧结束），
使用 1 kHz 整数增益显式限矩 PD，保留 0/5/10 ms 延迟采样。
该固定检查点的 CPU 整段评测为 40/48 完成，其中新增 SDK 动作 8/8，原 LAFAN1 为 32/40；菜单中的 15 条均完整完成。
整段报告位于 `outputs/casbot02_primitive_data_068000/evaluation.json`。完成仅表示未触发高度/倾斜终止，不代表路径精确或真机成功。

导出记录、快照和 32000 次旧包备份位于 `outputs/casbot02_primitive_data_068000_15motions/`，
总记录为 `deployment.json`。未上传或启动真机，训练进程未中断。
手动更新脚本和命令见 [训练检查点到部署包](../docs/CASBOT02_DEPLOY_EXPORT.md)。

历史 20000 次权重的 CPU 整段评测为 26/40 完成（65%），其中当前菜单 7 条均完成；
该历史权重的 walk1 平均根部位置误差 18.26 m、关节 RMSE 0.164 rad，说明完成不代表全局路径准确。
这些历史数字不是当前 68000 次权重的评测结果，报告保留于 `outputs/casbot02_primitive_020000/`。

## 手柄直接切换下一条动作（2026-10-28）

默认配置已开启 `joystick_enabled: true`，直接订阅 SDK 2.2.37 的
`/joystick_events`（`crb_ros_msg/msg/JoystickCmdReport`）。无需额外启动终端选择脚本。

- X：每次单独按下再松开，切换到下一条。RUN 中使用 `switch_blend_seconds: 0.8` 秒的参考轨迹桥接，进入 `SWITCH` 后恢复 `RUN`；策略一直计算，不经过 RETURN、默认姿态 PREPARE 或再次 HANDOFF。保留实测历史和上一帧策略动作。
- 桥接从当前参考姿态/速度到下一条第 0 帧，使用平滑权重并重算关节速度、单位重力及根部角/线速度；参考仍为原来的 10×64 编码。新动作时钟在桥接到第 0 帧后从第 1 帧正常推进。当前仅支持无根部位置观测策略。
- IDLE/HOLD/DONE 中按 X 与首次自动启动均跳过准备姿态，直接从当前实测关节位置进入 1 秒接管。RUN 内的 0.8 秒动作切换不变。
- 十字键不再用于 SuperMimic 动作选择。终端参数/脚本仍可按名称选择或重播指定动作。
- B：平滑结束当前动作并回位 HOLD，取消待播放请求。它是回位操作；底层阻尼/急停仍使用原 SDK 操作。
- 长按不会重复触发；开机时已经按住的键必须先完全松开。LB/RB/Back/Start/扳机等组合均被屏蔽，原 SDK 的模式切换和急停操作保留。

循环顺序由 `joystick_motion_order` 配置，当前为：
`walk1_subject2 → 原地左转 → X1_1000HZ → 后退 → 直行2步态 → 踏步 → 原地右转 → X1_Xiao360_100HZ → 直行1步态 → dance1_subject2 → walk1_subject1 → walk2_subject4 → walk3_subject1 → walk4_subject1 → dance1_subject3 → walk1_subject2`。
默认当前动作是 `walk1_subject2`，第一次有效 X 松开后会进入“原地左转”；在最后一条 `dance1_subject3` 松开 X 会回到 `walk1_subject2`。步进以已加载的 `motion` 为准；被取消/拒绝的切换不会额外跳过动作。
控制器会立即打印 `Joystick X: walk1_subject2 -> 原地左转; ...`。状态中的 `joystick_choice` 是请求目标，
`motion` 是当前已加载动作，`joystick_pending` 是待执行请求，`joystick_rx` 为消息计数。

只有健康的模式 7 接受手柄动作命令。准备/接管/桥接/回位期间重复 X 被忽略；B 可以中断桥接并按原流程回位，动作正常结束也仍回位；FAULT 不会被手柄自动清除。
手柄消息中断超过 0.5 秒时，尚未执行的切换请求会取消，恢复后需先松开再重新操作；
已在运行的策略仍遵循原有传感器、模式和推理检查。手动 stop/prepare/start/finish/play 或参数切换也会取消之前排队的手柄请求。

首次切入调试模式仍沿用 `auto_start_on_mode7: true`，自动播放默认动作。
如果将该项改为 `false`，进入模式 7 后会等待操作；X 仍表示进入当前默认动作的下一条，因此第一次 X 会播放“原地左转”。要播放指定名称，可使用下文终端 `/play` 接口。

本地 ARM64 包已构建；更新功能后上传并重启一次控制器即可，后续切换不再重启：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced/rl_controller
./scripts/install_remote.sh
```

Orin 按原流程启动底层 SDK，然后运行：

```bash
source /workspace/rl_controller/setup.bash
bash /workspace/rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false
```

本次没有连接真机或上传 Orin。手柄定义来自 SDK 锁定的 crb_ros_msg 子模块，
验证范围见主项目 `outputs/casbot02_hardware/x_next_motion_validation.json`。

**X 短过渡的历史验证（20000 次权重）：** 当前顺序的 7 组相邻切换（包括末尾回到第一条），每组先播放 5 秒、桥接 0.8 秒、再播放 10 秒，均通过原生 MuJoCo 隐式 PD 和 1 kHz 整数增益显式限矩 PD 短测；保留 0/5/10 ms 延迟采样。另通过历史/上一动作连续性、桥接终点第 0 帧与第 1 帧时序检查，以及隔离 ROS 的重复 X 屏蔽、桥接中 B 回位、正常结束与终端切换回归。结果见 `outputs/casbot02_hardware/x_fast_switch_validation.json`；当前 68000 次权重的更新结果见本文顶部。固定切换时刻、标称仿真参数的短测不验证任意相位或真机完整 SDK/电机链路；该模型并未专门训练任意动作切换。

## 终端日志显示

默认 `console_full_status: false`，终端仅在动作、运行阶段、模式或传感器可用性/故障状态变化时打印简短状态，避免滚动淹没 X 切换提示。完整诊断仍以 1 Hz 发布到 `/supermimic_controller/status`，手柄和控制频率不变。

需要完整日志时，可在 YAML 改为 `console_full_status: true`，或启动时添加 `-p console_full_status:=true`。仅查看一条完整状态：

```bash
ros2 topic echo /supermimic_controller/status std_msgs/msg/String --once
```

## 整段播放和免重启切换动作（2026-10-28）

默认 `supermimic_walk1_subject2_proprio.yaml` 现为 `preview_seconds: 0.0`，并启用 `hold_last_frame_on_finish: true`：完整播放后进入 **END_HOLD**，固定最后一帧的关节、姿态和高度参考，将整个未来参考窗口的速度置零，继续按 50 Hz 更新观测历史并运行策略。不是冻结最后一次电机指令，也不会因自然结束自动回位或进入阻尼。末帧姿态的真实平衡能力仍取决于策略和该姿态，短测试不能保证长时间稳定。

END_HOLD 中松开 X，沿用 0.8 秒连续策略参考过渡切到下一动作；B 或 `~/finish` 仍执行 3 秒回位后固定 PD 的 **HOLD**。`~/stop`、模式退出或保护故障保持原有行为；若日志是 **FAULT**，需要依据 `fault=` 原因排查，开启末帧保持不会绕过该故障。设置 `hold_last_frame_on_finish: false` 可恢复自然结束后的回位路径，旧配置缺少此项时也使用旧行为。

正数 `preview_seconds` 仍可设置 RUN 阶段的秒数上限，触发时沿用回位路径；该时间上限与自然播放结束的 END_HOLD 分开。HANDOFF 不计入预览时间预算，END_HOLD 中传感器、模式、倾斜、推理和目标过期检查继续生效。

默认在启动时预加载菜单中的 15 条动作，共用当前 68000 次更新 actor；手柄顺序以 YAML 的 `joystick_motion_order` 为准，完整顺序见上文。
动作库在 `config/model/supermimic_motion_bank/`，配置的 `motions` 将名称映射到各 bundle。
启动时检查所有包的模型/观测/电机契约和实际 actor 文件一致；切换时不读磁盘、不加载新 ONNX。
默认从 `walk1_subject2` 开始，输入仍为 870+640、27 输出。

本地需要重新构建和上传（只编辑 YAML 无法让旧二进制支持新功能）：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced/rl_controller
./build_in_docker.sh arm64
./scripts/install_remote.sh
```

Orin 终端一，按原 SDK 2.2.37 流程启动控制器，再用手柄切调试模式：

```bash
source /workspace/rl_controller/setup.bash
bash /workspace/rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false
```

终端二可列出或选择并播放动作，不必重启终端一：

```bash
source /opt/ros/humble/setup.bash
source /workspace/rl_controller/setup.bash
python3 /workspace/rl_controller/scripts/supermimic_motion.py --list
python3 /workspace/rl_controller/scripts/supermimic_motion.py dance1_subject3
python3 /workspace/rl_controller/scripts/supermimic_motion.py walk1_subject2
```

该脚本会明确请求结束当前 PREPARE/READY/HANDOFF/RUN，等待 RETURN 完成到 HOLD，再选择新动作并调用 `/play`：跳过准备姿态 → 从实测关节位置开始固定起始参考 HANDOFF 1 秒 → 完整动作 → 回位 3 秒 → HOLD。如果当前已经 HOLD/IDLE/DONE，直接选择并准备；若 FAULT 则报错，不自动清故障。每次新动作从第 0 帧开始准备，策略从 f+1 开始推进，历史和上一动作重新初始化。重复播放同一个名称也支持。

终端脚本仍通过回位和准备切换；手柄 X 在 RUN 中采用上文的短参考桥接。回位沿用 prepare 前实测姿态的 PD 插值；`/play` 仍要求新鲜模式 7、传感器和 SDK 增益反馈。

仅选择、不启动：

```bash
python3 /workspace/rl_controller/scripts/supermimic_motion.py walk3_subject1 --select-only
# HOLD/IDLE/DONE 下也可直接设置；RUN/RETURN 等阶段会拒绝。
ros2 param set /supermimic_controller motion walk3_subject1
# 以后显式启动选中动作：自动完成 prepare 和 start。
ros2 service call /supermimic_controller/play std_srvs/srv/Trigger '{}'
```

配置 `bundle` 仍决定基础权重和默认动作，命令行 `-p motion:=walk3_subject1` 可指定本次进程的起始动作。
可用动作参数只读：`ros2 param get /supermimic_controller available_motions`。
运行期间不覆盖已加载的动作包文件。增加动作需要离线导出、加入 `motions` 后再重启一次；已预加载的动作可反复切换。

需要准备更多动作时，在主项目目录执行：

```bash
env -u PYTHONPATH uv run python scripts/prepare_deployment_motions.py \
  --clips 踏步 直行1步态 后退 直行2步态 原地左转 原地右转 X1_Xiao360_100HZ X1_1000HZ \
  walk1_subject2 dance1_subject2 walk1_subject1 walk2_subject4 walk3_subject1 walk4_subject1 dance1_subject3
```

MorphMimic 的 `deploy/sim2sim/policy_node.py` 中没有 `preview_seconds` 上限；`CuedClipReference` 支持一次 cue 完整播放“站立引入 → 单条 clip → 返回站立”，结束后可再次 cue，播放期间 cue 被忽略。该文件启动时只接收一个 `--clip`，没有预录多动作库选择接口。硬件 `g1_node.py` 另有可选的 `--max-steps N`（50 步/秒），默认 `None` 不限步数；它与固定开启的预览时限不同。源码：[policy_node.py](https://github.com/lzyang2000/morphmimic/blob/main/deploy/sim2sim/policy_node.py)。硬件入口：[g1_node.py](https://github.com/lzyang2000/morphmimic/blob/main/deploy/hardware/g1_node.py)。本地沿用现有控制目标渐变，而非直接移植其参考状态插值。

本次验证使用离线 C++ 推理与独立 ROS 域内的合成 SDK；没有启动真机或上传 Orin。验证记录见主项目 `outputs/casbot02_hardware/full_clip_motion_switch_validation.json`。

## 策略计算时间容限

策略目标频率仍是 50 Hz（20 ms 周期）。偶发一次 21.85 ms 不再触发故障：

```yaml
max_inference_seconds: 0.04       # 单次计算硬阈值 40 ms
late_inference_seconds: 0.025     # 连续超时的判断阈值 25 ms
max_consecutive_late_updates: 20  # 连续 20 次计算超过 25 ms 才故障
max_target_age: 0.06              # 目标或计算结果超过 60 ms 仍故障
```

在新目标提交前，SDK 继续跟随上一目标；调度落后不连续追帧。不超过 25 ms 的计算
会清零连续超时计数；每次 start 也会清零。状态新增 `late_updates`（进程累计超过标称 20 ms 次数，仅统计）、
`consecutive_late`（连续超过 25 ms 的次数）和 `max_policy_compute_ms`（进程累计最大计算时间）。
计算时间包含该段锁等待及线程调度，不能等同于 ONNX 的纯 CPU 运算时间。

对照 `/home/casbot/Desktop/rl_controller/rl_controller` 中的旧 AMP：它没有单次推理超时故障；
50 Hz 线程记录实际频率，无新结果时沿用已有目标；2000 次调用没有新结果仅打印错误。
SuperMimic 保留单次严重超时、持续超时、目标过期、传感器和模式故障检查。

## 当前 SDK 2.2.37 部署接口（2026-10-27）

本节替代下文历史记录中的“外部显式 PD / effort 透传”真机接口说明。
SDK 2.2.37 会覆盖腿部 effort，手臂/腰部接收位置插值目标；当前 runner 因此发送
`position + kp/kd`、`velocity=0`、`effort=0`，由 SDK 执行控制。策略仍是 870+640 输入、27 动作，权重和参考编码没有改变。

- 命令包含 27 个策略关节和两个头部关节（共 29 个）；头部保持 prepare 时的实测位置，Kp/Kd 为 SDK 头部默认 30/2。
- 策略关节增益来自 manifest，并显式截断为 SDK 实际使用的整数；例如 276.3489/17.5929 变为 276/17。没有改写训练参数或模型 manifest。
- 目标位置根据采样状态限制 PD 请求幅度并限制到模型硬关节范围。它不是硬件力矩限幅保证：SDK 插值、状态延迟、控制库和电机固件在下游。
- stop、FAULT、IDLE、DONE 使用 Kp=0、Kd=对应整数阻尼、effort=0。阻尼依赖 SDK 的实时测量，不再在上位机使用陈旧速度算力矩。
- 正常 Ctrl+C/SIGTERM 会在 ROS 退出前留出 150 ms 发送阻尼。SIGKILL、崩溃、断网无法靠上位机保证停机；这版 SDK 缓存命令且没有接收超时看门狗。
- `/motion/current_mode` 提供真实模式号，收到新鲜的 `7` 才认定为调试模式。每 50 ms 调用 `/get_robot_mode` 仅获取分类：2=ZERO、3=STAND、4=WALK。SDK 把调试模式也归为 4，不能把分类 4 当成原始模式 7。
- 自动启动要求先确认分类 2/3（阻尼/准备），随后收到新鲜的真实模式 7 话题。启动时已经在调试模式，或只查询到分类 4，不会自动动作。先用手柄回到阻尼/准备，再切入调试模式。原始模式话题超过 max_sensor_age（默认 100 ms）未更新，或查询超过 250 ms 未更新，会阻止启动/触发运行故障。
- 真实模式还要求 `/motion/debug/joint_state` 返回的实际 Kp/Kd 与已发送增益匹配。首次 prepare 前等待匹配；运行中增益切换最多允许 250 ms 反馈时间，失配会锁存故障。支持 SDK 的 49 名称、29 个非手指增益布局。

Orin 上先检查实际 SDK 配置（默认路径可在部署 YAML 的 `sdk_config_file` 修改）：

```yaml
# /workspace/hl_motion/hl_config/motion_params.yaml
# 修改已有配置项，保留其他参数；修改后按原 SDK 启动流程重新加载。
debug_mode_params:
  use_default_kp_kd: false
```

真实启动会读取该文件；缺失或开关仍为 true 时直接拒绝启动，不创建电机命令发布者。
文件内容不代表进程已重新加载，所以另外核验实时增益反馈。不要只改文件而继续使用未重载的 SDK。

本地构建和上传：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced/rl_controller
./build_in_docker.sh arm64
./scripts/install_remote.sh
```

Orin 先启动控制器，确认处于 SDK 阻尼/准备模式（状态 mode_class=2/3），再用手柄切入调试模式：

```bash
source /workspace/rl_controller/setup.bash
# 先观察接收和模式查询；默认不发布电机命令，pd_feedback=dry_run_unverified。
bash /workspace/rl_controller/scripts/run_supermimic.sh
# 结束上一条进程后，启用真实 SDK 位置/PD 命令：
bash /workspace/rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false
```

自动流程：新鲜传感器/模式/增益反馈 → 连续稳定 0.5 秒 → 跳过默认姿态准备 →
从实测关节位置开始固定起始参考 HANDOFF 1 秒 → 完整播放所选动作 → 3 秒回位并 HOLD。
正常真实运行状态应显示 `command_mode=sdk237_position_pd`、`pd_feedback=matched`，
`mode_query_rx` 持续增加；调试模式时应同时显示 `mode=7 mode_class=4`，这是 SDK 的两套编号。不要同时运行 AMP 或其他电机命令发布者。

本次验证为本机编译、SDK 接口模拟和隔离 ROS 测试；旧显式限矩 PD 的 MuJoCo 成绩
不代表 SDK 上肢控制库、EtherCAT 或真机已经验证。训练电机参数、观测、模型和参考动作保留原值。



## 自动启动方式（SDK 逻辑沿用 2026-10-27，播放时长见文首）

默认 `supermimic_walk1_subject2_proprio.yaml` 已开启 `auto_start_on_mode7: true`。
先启动控制器，确认 SDK 处于阻尼/准备（mode_class=2/3），再用手柄切入真实模式 7：传感器与模式消息连续新鲜 0.5 秒后，
从当前实测关节位置自动开始 1 秒 HANDOFF，随后完整播放参考动作，
最后 3 秒回到准备前实测姿态并 HOLD。无需调用 prepare/start 服务。

如果启动时已经处于模式 7，控制器保持 IDLE，须先退出再重新切入；只有先确认 SDK 阻尼/准备，随后收到新鲜的真实 mode=7 才会自动执行。
启动时已在 mode=7 不会直接引发动作。自动启动每次进程只尝试一次；stop、fault、模式退出、正常结束或手动
prepare/start/finish 后不会再次自动触发。等待初始新鲜数据最多 5 秒；启用正数准备时间时，准备后等待到位也最多 5 秒；
超时进入 FAULT。原有传感器、目标年龄、姿态及模式检查继续生效。

Orin 启动（默认仍为 dry-run，无真实命令发布）：

```bash
source /workspace/rl_controller/setup.bash
bash /workspace/rl_controller/scripts/run_supermimic.sh
```

启用真实命令输出：

```bash
bash /workspace/rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false
```

启动后切模式 7 即自动执行。退出自动启动模式、继续用手动服务：

```bash
bash /workspace/rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false -p auto_start:=false
```

状态包含 `auto_start=waiting_mode_edge/pending/consumed/disabled` 和 `auto_reason`。
`consumed` 表示本次自动机会已使用或被手动操作取消。故障后可显式 prepare/start 恢复，
不会仅因数据恢复或再次切模式而重启。下面的手动启动说明用于关闭自动启动后的操作或历史配置。

**历史 48000 次更新权重（已被当前 primitive/20000 快照替换）：** `casbot02_lafan1_proprio_paper_kpkd/ckpt_0048000.pt`。当时 40 条动作整段 GPU 仿真评测完成 30 条（75%，按未触发高度/倾斜失败终止统计），`walk1_subject2` 完成全部 261.22 秒；另通过前 10 秒 C++ ONNX / 1 kHz 显式限矩 PD 原生 MuJoCo 闭环及隔离 ROS 流程检查。历史评测和验证记录见主项目 `outputs/casbot02_kpkd_048000/deployment_sync.json`。部署是固定快照，后续不会自动热更新。

**当前准备/接管流程：** `prepare_seconds: 0.0`，跳过准备姿态，配置中的默认姿态表不参与此次启动；从实测关节位置进入 `HANDOFF`，固定第 0 帧的整个未来参考窗口、参考速度为零，用 1 秒交接策略。观测历史始终更新；HANDOFF 完成后不重置历史，从 f+1 正常推进动作，动作播放预算从这时开始；当前 `preview_seconds: 0` 不截断参考。自然结束进入 END_HOLD，只有 B / `~/finish` 才按 3 秒回位并 HOLD。默认位置表只影响准备目标，不会擅自改变 actor q0。跳过准备和冻结参考流程已通过隔离 ROS 测试；尚未做完整启动/回位物理闭环和真机验证。

**SDK 接收已对齐 AMP：** 同时订阅标准 `/motion/joint_state` 与 debug `/motion/debug/joint_state`，从独立校验的完整新鲜帧中按接收时间选取最近来源。使用 4 线程执行器和独立的传感器/控制回调组。手柄模式仍监听 `/motion/current_mode`，真实命令只在 7 内发布；默认 walk1 配置会在确认 ZERO/STAND 分类、随后收到真实模式话题 7 后自动准备并启动；每次进程只尝试一次，手动 prepare/start 仍可用。状态分别显示双关节来源、IMU 和模式的接收计数。完整流程见主项目 `docs/CASBOT02_PROPRIO.md` 的 AMP SDK 小节。

以下较早的 26 万步/0.5 秒接管说明为历史背景，当前行为以本段和主项目部署文档为准。

**历史动作前后过渡配置。** 早期版本使用 `prepare_seconds: 6.0`、`entry_blend_seconds: 0.5`，并在预演/片段正常结束或调用 `~/finish` 后以 `return_seconds: 3.0` 回位。当前准备/末帧保持流程以上文和 YAML 为准；显式 B / `~/finish` 的回位仍是控制器目标插值，不保证站立平衡。`~/stop` 和故障仍进入阻尼。

**当前电机参数已按 HANDOFF 的 casbot02_constants.py 更新。** Kp、Kd、action_scale 和公式所需的力矩上限已同步到训练配置及 walk1 部署 manifest。腕部 6/7 轴按源文件轻型臂电机补齐，腰使用其单独定义的腰电机。旧 26 万步 actor 未按这些新参数训练，新参数下显式 PD 闭环在 9.56 秒失败，因此下文旧参数的 10 秒成功记录不再代表当前配置验证通过。需使用新训练权重重新验证。原包备份位于主项目 `outputs/casbot02_handoff_parameters/before/hardware_walk1/`；参数对照和失败报告见主项目部署文档。此次只生成本地配置和 ARM64 构建产物，没有启动真机控制。

**当前默认入口已切换为只需 IMU 和关节状态的 640 维策略。** 完整配置、导出及验证命令见主项目 [CASBOT02_PROPRIO.md](../docs/CASBOT02_PROPRIO.md)。`config/supermimic_walk4_proprio.yaml` 强制 `require_anchor_obs: false`，不订阅根部位姿，不估计根部位置；实际位置在内存中标记为 NaN。默认 dry-run 不创建电机命令发布者。

**2026-10-26：默认动作切换为 `walk1_subject2`，权重使用旧版平台期代表 `casbot02_lafan1_proprio_codex_fixed/ckpt_0260000.pt`。** 根据训练曲线选择 24–26 万步区间，未做完逐模型精确筛选，不宣称精确最优。默认配置为 `config/supermimic_walk1_subject2_proprio.yaml`，模型与动作位于 `config/model/supermimic_walk1_subject2_proprio/`。ONNX 接口仍为 `obs_hist[1,870] + goal[1,640] -> action[1,27]`，无需根部位置。

`walk1_subject2` 保留完整 13061 帧（261.22 秒）；配置仍按原流程预演 10 秒。本次已通过从静止起步、C++ ONNX、1 kHz 显式限矩 PD 的前 10 秒闭环，actor 最大差约 `8.58e-6`，p99 往返约 1.08 ms。报告见主项目 `outputs/casbot02_hardware/proprio_260000_walk1_subject2_effort_10s.json`；没有进行完整动作或真机验证。`scripts/run_supermimic.sh` 已默认选择新配置，旧 walk4 配置保留。

**以下 walk4_subject1 的 28 万步结果为历史验证记录。** C++ 启动会打印实际 checkpoint 迭代与 SHA256。

2026-10-26 ARM64 Docker 构建已验证：`./build_in_docker.sh arm64` 编译、安装成功，输出在 `build/install_arm64/`，默认配置为上述 `walk1_subject2`。复制项目后若旧 CMake 缓存仍记录原绝对路径，`make.sh` 会调用 `scripts/cmake_cache.sh`，将失效的构建目录备份为 `build/build_arm64.relocated.*` 后重新配置；无需删除模型、配置或整个 build 目录。已确认安装的 runner/probe 为 AArch64 ELF，并核验安装包的模型、动作文件 SHA256。此项为交叉编译验证，不是 ARM 板或真机运行验证。

真实模型已通过完整 163.92 秒 C++ ONNX 闭环、从静止起步的 1 kHz 显式限矩 PD 闭环，以及仅使用合成话题的 ROS 接口验证。显式 PD 闭环中的 Python/C++ 观测与目标最大差为 0，actor 最大差 `1.34e-5`。报告见主项目 `outputs/casbot02_hardware/proprio_280000_walk4_{onnx_full,effort_from_rest}.json`。

该历史模型长时路径仍明显漂移（显式 PD 的平均根部位置误差约 9.89 m）；闭环完成不是路径精度合格，也不是实机测试。当时使用 10 秒预览、dry-run、模式 7 门控、prepare/start 分步启动。当前播放与切换方式以上文 2026-10-28 说明为准。旧 `casbot02_lafan1_proprio_codex` 的 NaN 权重仍不使用。

在主项目目录启动当前包：

```bash
bash rl_controller/scripts/build_supermimic.sh
bash rl_controller/scripts/run_supermimic.sh
```

下文保留原 **649 维、有根部定位**模型的历史说明与结果。复查该路径必须显式选择旧配置：

```bash
SUPERMIMIC_CONFIG="$PWD/rl_controller/config/supermimic_walk4.yaml" \
  bash rl_controller/scripts/run_supermimic.sh
```

下文省略配置变量的启动示例应按上述方式补齐。无定位模型不能沿用下文的旧模型成功成绩。

已添加独立的 `supermimic_runner`，复用当前 SDK 的关节命名与 ROS 消息，替换 AMP 的观测、策略、参考和动作控制路径。原 `rl_controller_runner` 仍是 AMP；真机上两者不能同时下发。

## 历史 649 维模型与验证

- 权重：`casbot02_lafan1/ckpt_latest.pt`，iteration 42150，SHA256 `81656ac54547d1afbe2cccf6fecda943119625f3c6b2e824d7df85af39b32e3b`。
- 单动作：`walk4_subject1`，8196 帧，50 FPS，163.92 秒。ONNX、参考数据及电机参数已放在 `config/model/supermimic_walk4/`。
- C++ ONNX Runtime 1.19 整段闭环：无高度/倾斜失败，平均根部位置误差 0.0717 m，关节 RMSE 0.1650 rad。
- 从零初速度起步，采用显式 PD、1 kHz 物理步长、逐电机力矩限幅的整段闭环：无失败，平均根部位置误差 0.0765 m。
- 抽查 C++/PyTorch actor 输出最大绝对差 `3.82e-5`；8196 步观测及目标构造与 Python 一致。测试包括非零世界 XY/航向对齐。
- 上述为本机 CPU MuJoCo 仿真；ROS 用隔离话题和合成传感器测试，未驱动真机，未测机器人计算板的调度与推理性能。

原始报告在主项目 `outputs/casbot02_hardware/`。选择这个动作依据已有完整仿真表现，不代表已经适合无人保护的真机行走。

## 真机接线要求

配置：`config/supermimic_walk4.yaml`。话题及 frame 均需与实际一致。

| 输入 | 默认话题 | 消息/约定 |
|---|---|---|
| 关节 | `/motion/debug/joint_state` | `crb_ros_msg/msg/JointStateData`；必须一次提供完整 27 关节位置、速度，rad/rad/s |
| 陀螺 | `/motion/imu` | `sensor_msgs/msg/Imu`；角速度经 `imu_to_base_quat_wxyz` 转到 base_link |
| **根部位姿** | `/supermimic/base_pose` | `geometry_msgs/msg/PoseStamped`；base_link 相对固定、地面 z=0、z 向上世界坐标系的位置和姿态；默认 `header.frame_id=odom` |
| 模式 | `/motion/current_mode` | `std_msgs/msg/Int32`；只在模式 7 发布命令 |

根部位姿是**新增必需输入**。当前 649 维权重有 anchor_obs，不能用恒零位置、参考位置、IMU 积分的替代值冒充实测位置。已有 odometry/动捕若测的是其他身体位置，先用外参变换到 base_link。相机/IMU 装在运动躯干上时，单个固定外参不够。

所有传感器必须有有效、同一时间域的时间戳。默认最大消息年龄 100 ms，跨传感器时间差 40 ms。base_pose 的初始 yaw 可任意，启动时只对齐一次 XY/yaw；世界 z 必须保留相对地面的真实高度。

输出使用 `/motion/debug/joint_cmd`，只发送被训练模型控制的 27 个关节，顺序显式映射为腿 12、腰 1、双臂各 7。头和手指由已有控制器负责。

**本实现选择 effort-only 接口**：每 1 ms 计算并限制 `tau=Kp*(q_des-q)-Kd*dq`，消息 `effort=tau`、`kp=kd=0`、`position=实测位置`、`velocity=0`。这样不会把很大的策略位置目标直接下发给位置控制器，也不会重复叠加 PD。

需核实机器人上 `RLAgent::dispatchWholeBodyJointCommand` 及更下层驱动确实透传 effort、接受零 kp/kd，并且对局部 27 关节命令按名称更新。当前可见 SDK 头文件和消息能证明接口存在，不能证明真机整条链路的行为。若实际只允许位置 PD，需要按底层接口另接并验证力矩限制，不能直接把本配置用于该接口。底层命令超时、急停机制也应保持可用；进程退出后本程序无法继续发送阻尼。

当前训练模型为无吉他负载的 27 自由度机器人；旧 AMP 吉他固定手臂逻辑不再使用。真机负载、手臂自由度和零位必须对应这个模型。

## 构建与启动

在当前电脑原生编译：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash rl_controller/scripts/build_supermimic.sh
```

已有 Docker 的 ARM 交叉编译流程也会构建新 target：`cd rl_controller && ./build_in_docker.sh arm64`。当前回合验证了本机 amd64 编译，未验证 ARM 交叉构建。

默认启动是 dry-run，**不创建电机命令发布者**：

```bash
bash rl_controller/scripts/run_supermimic.sh
```

查看状态：

```bash
ros2 topic echo /supermimic_controller/status
ros2 topic echo /supermimic_controller/dry_run_command
```

没有完整根部位姿时会停留在等待状态，prepare/start 返回具体原因。

传感器及 effort 接口确认后，在机器人已具备支撑保护、且旧 AMP 进程退出的条件下，显式启用命令输出：

```bash
bash rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false
```

进入机器人原有的模式 7，然后分两步启动：

```bash
# 6 秒平滑到参考首帧关节姿态，参考时间不前进。
ros2 service call /supermimic_controller/prepare std_srvs/srv/Trigger '{}'

# 状态为 READY 且实际姿态/高度已满足门槛后，开始选中动作。
ros2 service call /supermimic_controller/start std_srvs/srv/Trigger '{}'

# 主动平滑结束：3 秒回到准备前实测姿态，随后 HOLD。
ros2 service call /supermimic_controller/finish std_srvs/srv/Trigger '{}'

# 停止；在模式 7 内切换为有力矩上限的阻尼，不自动重启。
ros2 service call /supermimic_controller/stop std_srvs/srv/Trigger '{}'
```

准备和回位终点的关节目标沿用 sim2sim 的软限位裁剪；策略运行中的 q_des 不做 AMP 裁剪。运行每 20 ms 取一帧实测状态，内部构造 10 帧历史及 10 帧未来目标；上一动作保存原始 actor 输出。入口 0.5 秒的位置目标在 1 kHz 侧由准备目标与策略目标混合，此后策略位置目标零阶保持；PD 始终在 1 kHz 侧重新计算。

首个策略目标沿用权重的 f+1 时序，参考时钟从 start 开始正常前进。默认预演到 10 秒进入 RETURN，3 秒回位后 HOLD；stop 则立即进入 IDLE/阻尼。传感器失效、推理超过 20 ms、策略目标超过 60 ms 未更新、明显倾斜或启用定位时的根部跟踪误差超限会锁存 FAULT 并中断过渡。RETURN/HOLD 期间仍检查传感器和模式。退出模式 7 停止发布；重新进入不会自动开始动作，需要 prepare/start。静态 PD 保持和阻尼都不能保证机器人站立。

## 复查命令

在主项目目录重新导出：

```bash
env -u PYTHONPATH uv run python scripts/export_hardware.py
```

复查 C++ 整段闭环（先用 CMake 编译 probe，或指定 ROS build 的 probe）：

```bash
env -u PYTHONPATH uv run python scripts/validate_hardware.py \
  --probe rl_controller/build/supermimic_ros/supermimic_probe \
  --from-rest --explicit-pd \
  --report outputs/casbot02_hardware/effort_walk4_from_rest.json
```

隔离 ROS 测试用兼容消息包，所有合成数据和输出话题都位于 `/supermimic_offline_test/*`，默认 dry-run 测试及非 dry-run 的模拟消息路由都不接真实 `/motion/*`：

```bash
source /opt/ros/humble/setup.bash
source rl_controller/build/supermimic_msgs/share/crb_ros_msg/local_setup.bash
ROS_LOG_DIR=/tmp/supermimic_ros_logs /usr/bin/python3 \
  rl_controller/tests/test_supermimic_ros.py \
  --runner rl_controller/build/supermimic_ros/supermimic_runner \
  --config rl_controller/config/supermimic_walk4.yaml
```

### 2026-10-27：接收缓存与发布阻塞隔离

传感器回调现在只持有独立的 `sensor_mutex_`，控制器取一次完整快照后释放该锁。
命令发送继续与 prepare/start/stop/finish 串行，保证 stop 确认后不会恢复旧的位置控制；
发布阻塞不会再占用传感器缓存锁。状态发布和日志在两个锁之外执行。
推理结果提交前检查模式和从采样到提交的总年龄，过期结果不会被刷新为新目标。
`max_sensor_age=0.10`、`max_target_age=0.06` 和模式 7 限制保持生效。

状态新增进程启动以来的最大耗时（毫秒）：
- `max_publish_ms`：命令/干运行诊断发布耗时。
- `max_report_io_ms`：状态发布和日志耗时，下一条报告反映本次耗时。
- `max_control_gap_ms`、`max_control_lock_ms`：控制回调间隔、等待控制锁耗时。
- `max_policy_commit_lock_ms`：推理完成后等待提交锁耗时。
- `max_debug_gap_ms`、`max_standard_gap_ms`、`max_imu_gap_ms`、`max_mode_gap_ms`：各路回调最大间隔。
- `*_gaps_over100`：各路回调间隔超过 100 ms 的次数；主动停发也会计入。
- `max_debug_lock_ms`、`max_imu_lock_ms`：回调等待传感器缓存锁的最大耗时。

这些改动不保证 DDS 发布或系统调度永远不会停顿。若命令发布实际阻塞，RUN/HANDOFF
仍应触发目标超时保护；指标用于区分发布阻塞、锁等待和接收间隔，不能用来忽略故障。

隔离回归测试使用 `tests/block_publish.c` 的测试专用 LD_PRELOAD 库，分别注入 350 ms
命令发布和状态发布阻塞。该库不编入、不安装到部署包，测试只使用隔离 ROS 域与测试话题。
