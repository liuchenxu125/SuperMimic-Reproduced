# Casbot02 PICO 实时遥操作

先接实时重定向，复用现有 motion actor。`upper-body` 用头显和双手柄控制双臂参考；`full-body` 用 Tracker 解算的 24 关节骨架。新增的 **GMR 只接全身流程**，不从三点推测骨架；原有简化 IK 保留。PICO 无线头手位姿和 A/B 通信已验证；用户已确认简化三点接口在仿真中双臂跟随。全身 GMR 目前完成离线合成输入验证，待购入 Tracker 后实测。独立 human encoder 训练入口见 [HUMAN_TRAINING.md](HUMAN_TRAINING.md)；此处实时发送器仍使用原 motion actor。

现有训练入口、观测、奖励、电机参数、训练环境 `.venv` 和普通部署 YAML 均不修改。PICO SDK 安装在 `.venv-pico`。模型默认跟随当前安装的部署权重，本次验证为 **cpu7 / ckpt_0096000.pt**，SHA256 `8ed49952531d5364daef6ca00c77b1d3f326803c3e5edf931ae10479cb4f1316`。

## 数据路径与维度

PICO / XRoboToolkit 输入 → 校准 → 简化 DLS IK 或全身 GMR → 50 Hz 机器人参考 → 10 帧真实预览缓冲 → 原策略。

三点模式不需要 Tracker：头显位置用于消除手部的共同平移，两只手柄分别映射两只腕部。转头不会直接转动机器人；首版不映射机器人的头部电机。只求解 14 个手臂关节，骨盆、腰和双腿使用固定站立参考。**固定的是参考，实际 27 个关节仍全部由策略闭环控制**。肘部姿态由 IK 和默认姿态先验决定，不对应人体肘部测量。

- 本体输入仍是 930：10 帧 × 93，包含每帧 6D 相对参考骨盆姿态。该反馈来自机器人 IMU 与当前执行的参考姿态，不能用最新 PICO 姿态替代已缓冲的参考姿态。
- 目标仍是 690：10 帧 × 69。每帧为关节位置 27、关节速度 27、参考航向系骨盆位移 3、参考航向系骨盆朝向 6、骨盆局部线速度 3、骨盆局部角速度 3。
- 策略输出仍是 27 个动作，目标角 `q0 + action_scale × action`，沿用所选模型的 Kp、Kd、q0 和 action_scale。
- 前端缓冲 9 个控制间隔，增加约 **180 ms** 延迟，再加设备、网络与计算耗时。没有伪造未来帧，也不需要真机根部位置测量。
- 默认 `--retargeter simple` 使用原有中立姿态差分 DLS；`--retargeter gmr --input-mode full-body` 调用实际 GMR 的两阶段 Mink/DAQP 求解器，加载下述 Casbot02 JSON。两者使用相同的参考协议和策略，不改训练配置。

## GMR 全身接口（Tracker 到位后使用）

完整链路是 **PICO 4 Ultra＋双手柄＋两枚脚踝 Tracker → XRoboToolkit Full Body Tracking → 24×7 骨架 → GMR → 50 Hz 参考 → 现有策略**。先在头显完成全身校准、开启全身数据发送，并用 `check_pico.py` 确认 `body_available=true`、`body_updates` 持续增长。仅头显和手柄时，全身发送器等待数据，不会生成替代骨架。

本机 GMR 源码为 `/home/casbot/Desktop/GMR`，依赖已安装在 `.venv-pico`。需要重建依赖时运行：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/teleop/setup_gmr.sh
```

其他机器没有该 GMR 目录时，安装脚本会在项目 `outputs/teleop_dependencies` 内克隆固定版本，发送器会自动使用它；也可用 `--gmr-root` 显式指定。

先运行仿真接收端：

```bash
bash scripts/teleop/run_sim2sim.sh
```

另一个终端运行全身 GMR 发送端：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/teleop/run_pico_gmr.sh \
  --record outputs/pico_sessions/gmr_full_body01.npz
```

站直并自然放松双臂，按右手柄 A 校准并开始，B 暂停，A 恢复。校准会从策略 q0 预热求解 30 次，再以实际脚底网格对齐地面；首次校准耗时不计作已建立数据流的中断。首次发送仍有 180 ms 预览缓冲。重新标定需重启两端。

配置文件：`src/supermimic/teleop/ik_configs/xrobot_to_casbot_casbot02.json`。可显式指定：

```bash
bash scripts/teleop/run_pico_gmr.sh \
  --gmr-root /home/casbot/Desktop/GMR \
  --gmr-config src/supermimic/teleop/ik_configs/xrobot_to_casbot_casbot02.json
```

它以 [GMR xrobot_to_g1.json](https://github.com/YanjieZe/GMR/blob/master/general_motion_retargeting/ik_configs/xrobot_to_g1.json) 为基础，参考 `hl_robot_retargeter/config/robot/g1.yaml → casbot02_primitives.yaml` 的轴映射：

- 旋转组合为 `R_new = R_gmr × inverse(A_g1) × A_casbot`；左右腕部均包含旧项目中的额外轴变换，目标落在腕部，不添加手掌。
- 保留 GMR 原本的髋/肩 yaw 段来约束完整肢段姿态，膝、肘、踝等映射到对应 Casbot02 body。G1 的虚拟 toe 目标转换到踝部原点，不能直接把脚尖偏移原样贴到踝部。
- 保留 upstream 的两阶段权重、`human_scale_table` 与身高缩放规则；默认 scale 为躯干/腿 0.9、手臂 0.8。可用 `--human-height` 按 GMR 的规则调整，单位米；不可与简化 IK 的 `--scale` 混用。
- 旧批处理的逐骨长缩放、接触修正、自碰撞优化和导出后处理并不等同于 GMR 的 JSON 配置，当前不宣称复现了那些步骤，也不保证参考无自碰撞。
- 输出按当前 actor 导出模型校验 27 个关节顺序和限位，参考速度默认限为 6 rad/s；该参考平滑限制不修改执行器参数。
- `max_position_error_m` 统计第二阶段有位置权重的目标，`max_orientation_error_rad` 统计有朝向权重的目标；手臂沿用原 GMR 的朝向权重，不能把这两个数当成实际机器人跟踪误差。

配置带有源文件 SHA256，发送端启动也会打印配置与实际 GMR 求解源码 SHA256。加载器只导入 upstream 的 IK 模块，避免把其可视化/Torch 依赖装入 PICO 环境；局部兼容层修正 Mink 1.3 的 `limits` 参数位置，不修改 GMR 仓库或 Mink 全局函数。训练 `.venv` 不安装 GMR 依赖。

如需从那两份本地 HL 配置重新生成 JSON：

```bash
env -u PYTHONPATH .venv-pico/bin/python scripts/teleop/build_gmr_config.py
```

原简化接口保持可用：

```bash
# 原三点简化 IK：目前只有头显和手柄时使用
bash scripts/teleop/run_pico.sh --retargeter simple --input-mode upper-body
# 原全身简化 IK：有 Tracker 后可与 GMR 比较
bash scripts/teleop/run_pico.sh --retargeter simple --input-mode full-body
```

这次只新增重定向前端与配置，不更换部署权重、不修改训练任务，也不执行真机命令。得到真实全身录制后，应先用同一份录制比较两种重定向和策略跟踪，再接入真机。

## 1. PICO 软件准备

独立 Python 环境已在本机安装。其他机器上执行：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/teleop/setup_pico_env.sh
```

前端使用 [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit) 的 PC Service 与 PICO 客户端。只有头显和两个手柄时，使用 `--input-mode upper-body` 并开启 Head、Controller、Send，Tracker 模式设为 None。有两个脚踝 Tracker 时才使用 `--input-mode full-body`，开启 Full Body Tracking，使 `is_body_data_available()` 为真并输出 24×7 骨骼位姿。

SDK 绑定固定为 [PC-Service-Pybind / 8941e1d](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind/tree/8941e1dbbf4313fe83846f82984eaa8c992690bf)，与参考的 [MorphMimic 安装入口](https://github.com/lzyang2000/morphmimic/blob/main/deploy/install_pico.sh)一致。安装脚本只安装 SDK 绑定，不会自动安装或启动 PC Service/PICO 客户端。

### 第一次连接 PICO 4 Ultra

USB 能识别到头显，不代表 Python SDK 已经收到位姿。完整链路还需要电脑 PC Service、头显 XRoboToolkit 客户端和客户端的 Send 开关。

本机已下载官方 PICO 客户端 v1.1.1、Ubuntu 22.04 amd64 PC Service v1.0.0，以及 Android platform-tools，保存在 `outputs/pico_setup`。PC Service 已解包到项目内，无须系统级安装；客户端包名为 `com.xrobotoolkit.client`。

1. 保持头显开机。在设置的“通用/关于”中连续点击“软件版本”，开启开发者菜单中的 USB 调试，并允许此电脑。用下面命令检查，状态需为 `device`：

   ```bash
   bash scripts/teleop/adb_pico.sh devices -l
   ```

2. 电脑服务单独运行并保持该终端打开。本次配置时已在后台工具会话启动，不要重复启动；可先检查端口：

   ```bash
   ss -lntp | rg ':60061|:63901'
   # 仅在服务未运行时执行：
   bash scripts/teleop/run_pico_service.sh
   ```

3. 客户端已安装。其他头显需要安装时可执行：

   ```bash
   bash scripts/teleop/adb_pico.sh install outputs/pico_setup/XRoboToolkit-PICO-1.1.1.apk
   ```

   如果系统显示“本地缓存中没有权限信息，请确保联网后再启动应用”，先在头显连接能上网的 Wi-Fi，再重新打开 XRoboToolkit。该提示属于 PICO 应用校验，不是策略或 IK 报错。

4. 两种数据连接方式任选其一：

   - **USB**：官方客户端 v1.1.1 的数据 TCP 端口是 63901。建立以下映射后，在头显客户端 **Network 区域点击 `connect`**，打开标有 **`Enter PC Service's IP`** 的地址输入框，填写 **`127.0.0.1`** 并点击 Connect。Data & Control 中的 Send 是发送开关，地址输入不在该分区。USB 模式不依赖局域网自动发现；重新拔插线后需重建映射。

     ```bash
     bash scripts/teleop/adb_pico.sh reverse tcp:63901 tcp:63901
     ```

   - **局域网**：让头显与电脑网络互通，在客户端连接电脑 IP。无线接通时电脑 Wi-Fi IP 为 `10.10.7.43`，以后应重新用 `hostname -I` 核对；不要填 Docker 的 `172.17.0.1`。无线接通后不需要 USB 数据线，PC Service 仍需运行。

5. 在头显中开启 **Head、Controller、Send**；关闭 **Switch w/ A Button**，避免右手柄 A 同时切断数据发送与触发策略校准。先用只读诊断确认数据和按键：

   ```bash
   env -u PYTHONPATH .venv-pico/bin/python scripts/teleop/check_pico.py --seconds 15
   ```

   `tracking_updates` 应增长，按 A 后 `A_presses` 应增加。`tracking_timestamp_ns=0` 表示还没收到头显数据；能看到 A/B 而 `body_available=false`，表示头手链路已通，但没有全身骨架。

**当前用户设备只有头显和两个手柄，没有 PICO Tracker。** 使用下面的 `upper-body` 仿真指令，不需要等待 `body_available=true`。全身模式保持原接口，不用零骨架代替真实全身输入。

## 2. 先运行仿真

终端一，启动实时仿真接收器：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/teleop/run_sim2sim.sh
```

终端二，连接 PICO：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/teleop/run_pico.sh --input-mode upper-body \
  --record outputs/pico_sessions/upper_body01.npz
```

先退出旧的实时发送器/仿真接收器，避免旧校准 session 或端口占用。身体放松站直，双手自然放在身体两侧、手肘略弯、手柄分开，按 **PICO 右手柄 A** 校准并开始。这个姿态映射为机器人默认站姿；之后先做小幅抬手、伸手、转腕。仿真在收到第一个完整预览窗口后运行。

按 **PICO B** 冻结当前执行的参考，策略继续闭环；恢复前回到暂停时的手柄姿态，再按 A。改变 VR 原点或需要重新校准时，重启发送器与接收器。单帧手柄相对头部位移跳变超过 20 cm 或姿态跳变超过约 43° 时，三点前端暂停，避免把跟踪突变直接送进参考。

三点模式的 IK 参考速度默认 2 rad/s，腕部相对初始机器人位置的参考位移半径默认 0.35 m，超出时饱和并显示 `workspace_limited=true`。可用 `--max-joint-speed` 和 `--max-hand-displacement` 显式调整；初次测试保持默认。该范围限制不能保证无自碰撞。

发送端现在每约 2 秒输出一次状态：`WAIT_CALIBRATION` 表示确实尚未校准，`POSE` 表示已校准并发送活动参考，`HOLD` 表示暂停，`STREAM_GAP` 表示校准后超过 300 ms 未收到有效输入。`sent_windows` 是发送计数，不能单独证明接收端策略已正常跟踪；接收端应出现 `Live policy active` 且仿真有响应。`max_position_error_m` 是 IK 残差，不是策略实际跟踪误差；参考速度限制也会产生瞬态残差。

退出时按一次 Ctrl+C，等待 `Recording saved: ...`。录制会先写入临时文件并提交到目标 NPZ，然后再关闭 SDK，避免关闭期间再次 Ctrl+C 导致录制丢失。旧版在关闭 SDK 前尚未保存录制，退出时被打断可能没有产生 NPZ；这种情况需要重录。

若需要单独核对三点数据：

```bash
env -u PYTHONPATH .venv-pico/bin/python scripts/teleop/check_pico.py \
  --input-mode upper-body --seconds 15
```

有全身 Tracker 的原入口仍为：

```bash
bash scripts/teleop/run_pico.sh --input-mode full-body \
  --record outputs/pico_sessions/full_body01.npz
```

没有 PICO 时，可先验证完整传输链路：终端一相同，终端二改为：

```bash
bash scripts/teleop/run_pico.sh --source bundle --clip 踏步
```

也可回放之前录制的原始 PICO 骨架，重新经过 IK：

```bash
bash scripts/teleop/run_pico.sh \
  --input-mode upper-body --source recording \
  --recording outputs/pico_sessions/upper_body01.npz
```

关闭发送器或超过 300 ms 无新参考，接收器锁定最后执行的静止参考，策略继续反馈。即使网络恢复也不自动解除；在仿真窗口按 **R** 显式恢复。静止参考保持不等于独立站立控制器，任意姿态下均能保持平衡尚未验证。

全身模式可调整人体缩放 `--scale`，默认按站姿骨盆到脚踝高度估算，参考速度默认 6 rad/s；三点模式缩放默认为 1。它们都不修改训练执行器的速度或力矩参数。全身模式根部突跳超过 15 cm/帧或姿态突跳超过约 20°/帧时，发送端暂停参考并打印原因。

## 3. 独立真机入口

先完成真实 PICO 的仿真验证，三点模式目前只完成了合成数据的策略仿真。下面是真机接口说明，当前不要据此直接开始真机三点遥操作。普通按键选动作仍使用原来的启动脚本；遥操作使用独立的 `supermimic_pico_teleop.yaml`。两种控制器不能同时运行。

本地构建并按现有流程上传：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced/rl_controller
bash build_in_docker.sh arm64
./scripts/install_remote.sh
```

Orin 终端一，启动仅发布参考的 UDP→ROS 桥：

```bash
source /opt/ros/humble/setup.bash
source /workspace/rl_controller/setup.bash
/usr/bin/python3 /workspace/rl_controller/scripts/teleop_reference_bridge.py \
  --bind 0.0.0.0
```

该 UDP 接口不带网络认证，应仅在可信机器人局域网开放。端口默认 5556。PC 上启动 PICO 发送器，填写 Orin IP：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/teleop/run_pico.sh --host <Orin_IP>
```

Orin 终端二，首次检查先保持默认 dry_run：

```bash
source /workspace/rl_controller/setup.bash
bash /workspace/rl_controller/scripts/run_supermimic_teleop.sh
```

此时没有电机命令发布者。按 PICO A 完成校准，机器人按原 SDK 流程进入调试模式 7。检查状态中的 `sensors=OK`、`reference_source=live`、`live_rx` 持续增长、`live_bad=0`；遥操作入口不自动启动，也不响应原离线动作菜单。

```bash
ros2 topic echo /supermimic_controller/status \
  std_msgs/msg/String --once --full-length
ros2 service call /supermimic_controller/play std_srvs/srv/Trigger '{}'
```

dry_run 只测试计算和状态机；完成检查后退出该控制器。实际控制的启动命令为：

```bash
bash /workspace/rl_controller/scripts/run_supermimic_teleop.sh \
  --ros-args -p dry_run:=false
```

再次确认 SDK 模式、传感器和实时参考后，手动调用同一个 `play` 服务。接管使用 0.8 秒混合，并暂停起始参考；之后约 0.8 秒将参考接入实时流，避免直接跳到已经移动的人体姿态。底层通信、电机映射和 PD 反馈验证沿用当前 SDK 2.2.37 部署实现。

参考超时后状态会显示 `live_paused=true`。同一发送会话恢复且确认姿态正确后：

```bash
ros2 service call /supermimic_controller/resume_live std_srvs/srv/Trigger '{}'
```

恢复使用 0.8 秒参考混合。发送器重启产生新 session 时禁止直接续接，需停止并重新启动控制器。`finish` / `stop` 等既有服务仍沿用原实现。

## 验证范围

已验证 Python 与 C++ 的实时入口对相同参考产生与离线入口一致的历史观测、目标和动作；UDP 已知踏步参考连续仿真 10 秒通过；合成中立 PICO 骨架经 IK、预览和原策略仿真 10 秒通过。

10 项 Python 检查、5 项 C++ 检查、独立实时 ROS dry-run 及原离线模式启动回归均通过。独立 `.venv-pico` 的骨架录制回放 → IK → UDP → 仿真也完成 5 秒验证，接收 250 个参考窗口，未拒收数据或触发保持。ARM64 安装包已构建在 `rl_controller/build/install_arm64`；没有上传或启动真机控制器。本次逐文件核对 152 个既有训练文件，内容未变化。

三点模式另通过 15 项 Python 检查及约 20 秒合成上肢动作策略仿真：无高度/倾斜失败，腿腰参考变化为零，腕部 IK 最大位置误差约 1.3 cm，基座最大水平漂移约 6.8 cm。具体记录见 `outputs/casbot02_teleop/upper_body_sim_validation.json`。

上述结果不能替代真实 PICO 跟踪质量、足端接触、复杂动作、自碰撞和真机平衡测试。当前 IK 未实现碰撞约束，也未接入人体 encoder。三点 SDK 只有整体包时间戳，不能从 SDK 缓存独立确认每只手柄的新鲜程度，因此应保持 Head 和 Controller 同时开启，先用原地、小幅动作与录制回放做仿真验证。三点模式不跟踪操作者的行走和下蹲。
