# 当前 Casbot02 SuperMimic 权重的真机接口

用户现已确认只有 IMU 与关节状态。现有权重所需根部位置不可用；下一步的纯本体观测训练与部署路线见 [CASBOT02_PROPRIO.md](CASBOT02_PROPRIO.md)。

本文针对 `logs/casbot02_lafan1/ckpt_latest.pt`，不是 ProtoMotions 的 ONNX 模型。依据实际 checkpoint、训练配置和当前 mjlab 的观测/动作代码核对。

本机已实现 C++ ONNX/ROS 部署入口及 `walk4_subject1` 单动作包。`rl_controller` 是独立的内网部署工程，不随本公开仓库分发；其构建和 dry-run 说明位于本地 `rl_controller/SUPERMIMIC_DEPLOYMENT.md`。当前权重仍需要真实的 base_link 根部位姿，离线闭环通过不代表已完成真机测试。

## 推理接口

```text
obs_hist: float32 [1, 870]
goal:     float32 [1, 649]
       ↓ motion encoder + shared decoder
action:   float32 [1, 27]
```

部署只运行 actor。世界模型、critic、回放缓冲区和域随机化参数 xi 都不需要在线运行。当前权重跟踪参考动作，没有 vx/vy/偏航角速度的直接指令接口。

### 本体观测

单帧 87 维：`[q - q0 (27), qdot (27), omega_base (3), gravity_base (3), previous_action (27)]`。

使用最近 10 个 50 Hz 控制采样，按**观测项优先**拼接，而不是把 10 个完整的 87 维向量直接展开：

| 870 维区间，左闭右开 | 内容 |
|---|---|
| `[0:270]` | 10 帧关节相对位置 |
| `[270:540]` | 10 帧关节速度 |
| `[540:570]` | 10 帧 base_link 坐标系角速度 |
| `[570:600]` | 10 帧 base_link 坐标系单位重力方向 |
| `[600:870]` | 10 帧上一周期的归一化策略动作 |

每一项内部按由旧到新排列。初始化时用第一帧实测状态填满历史，上一动作初始化为 0。角度用 rad，角速度用 rad/s。若 R 是 base_link 到世界坐标系的旋转矩阵，重力项为 `R.T @ [0,0,-1]`，直立时为 `[0,0,-1]`。

IMU 不在 base_link 坐标系时，要先处理安装外参；若装在运动的躯干上，还需处理其与 base_link 之间的关节运动，不能直接把 IMU 数据当作基座数据。项目内部四元数为 wxyz；接 SDK 时须显式转换其约定。

Policy 内部已经包含固定缩放：关节速度除以 5、基座角速度除以 2 等。调用完整 Policy 时提供上述原始单位数据，不要重复归一化。

### 参考目标

前 640 维来自参考文件：从当前相位帧起，取 `t,t+0.02,…,t+0.18 s` 共 10 帧。每帧 64 维：

```text
参考关节位置 - q0                 27
参考关节速度                      27
参考根部坐标系单位重力方向          3
参考根部坐标系角速度                3
参考航向坐标系线速度                3
参考 base_link 高度                 1
```

这些是目标动作数据，不是额外真机传感器测量。参考动作可以离线预计算为 50 Hz 的特征表。

后 9 维是当前权重训练时显式开启的 `anchor_obs`：

```text
goal[640:643] = R_robot.T @ (p_reference - p_robot)
goal[643:649] = (R_robot.T @ R_reference)[:, :2].reshape(6)
```

其中 robot/reference 都指 base_link。6D 旋转的展开顺序为 `R00,R01,R10,R11,R20,R21`。这 9 维针对当前参考帧计算，不是 10 帧历史。

因此，当前权重除了编码器和姿态/角速度数据，还需要**base_link 的三维位置和姿态估计**，并让它与参考处于同一坐标系。可用已有根部状态估计器，或外部位姿测量；不要求测量所有身体刚体。单靠 IMU 的姿态/角速度不能直接得到可靠的位置输入。

动作开始时统一参考与实机的起始 XY 和航向，保持地面高度、重力方向一致。参考的位移、姿态和速度必须一起做一致的坐标变换；不要每一步把参考位置重新钉到机器人当前位置，否则位置误差被人为消去。

若硬件没有根部位置估计，需另训 `anchor_obs=False` 的 640 维目标策略，再单独验证精度与漂移。当前 649 维权重不能通过删除或将后 9 维恒置零，直接变成等价策略。

## 输出和电机控制

输出为 27 个无量纲动作，**不是关节角，也不是力矩**。只有中间 32 维 latent 使用 tanh，27 维最终动作没有 tanh 的 [-1,1] 保证。

对第 i 个关节：

```text
scale_i = 0.25 * effort_limit_i / stiffness_i
q_des_i = q0_i + scale_i * action_i
```

位置目标送给底层 PD。若硬件使用相应的力矩模式，其基准控制律为 `tau = Kp*(q_des-q) - Kd*qdot`，期望关节速度为 0，再遵守实际硬件力矩限制。当前仿真使用隐式位置 PD；真实电机的延迟、闭环响应和数值实现需要在 sim2sim/硬件验证中检查，不能认为控制公式相同就已完成迁移。

策略以 **50 Hz（每 20 ms）** 更新，底层电机控制按照硬件 SDK 所要求的频率运行，两次策略更新之间保持目标。200 Hz 是当前仿真的物理步频，不是从仓库推断出的真机驱动频率。

增益、默认站姿、动作缩放和力矩限值都沿用已复制的 ProtoMotions 参数。比如 `leg_l1_joint`：q0=-0.245 rad，Kp=286，Kd=8，力矩上限 150 Nm，scale≈0.131119 rad/action。硬件 ID、方向和零位还须与真机编码器校准对应，不能用软件中的名称推断总线 ID。

策略输入关节和输出动作均为：

```text
0–5    leg_l1_joint … leg_l6_joint
6–11   leg_r1_joint … leg_r6_joint
12     waist_yaw_joint
13–19  upper_left_1_joint … upper_left_7_joint
20–26  upper_right_1_joint … upper_right_7_joint
```

注意：原始 MuJoCo actuator 枚举中腰部执行器位于最后，mjlab 的 JointPositionAction 实际按自然关节顺序选择目标，策略动作的腰部索引仍为 12。不能直接拿 actuator 枚举或 motor 参数字典顺序替代策略关节顺序。

完整逐关节表及 I/O 定义见 [io_contract.json](../outputs/casbot02_deployment/io_contract.json)。

## 部署顺序与现有代码边界

1. 本 checkpoint 的 actor 已导出为 TorchScript CPU，保留两个输入及内部固定缩放；导出前后输出已对拍。文件为 `outputs/casbot02_sim2sim/actor.pt`。
2. 独立观测历史、参考时钟、空间对齐和 anchor 构造已在原生 CPU MuJoCo 中闭环测试，并与训练环境输入逐项核对；命令和结果见 [sim2sim 说明](CASBOT02_SIM2SIM.md)。
3. 接入现有 SDK：读取 q、qdot、基座 IMU/位姿估计，经名称映射按策略顺序构造输入；推理后按逐关节 scale/q0 生成 q_des，再映射到总线顺序发送。通信超时、限位/限矩和异常状态保护由现有硬件控制层承担。
4. 先以支撑/吊装条件检查关节方向、零位和站姿衔接，再验证短时低幅动作；启动参考必须与实机姿态、速度接近，不能像仿真重置那样瞬间置入动作第一帧。

ProtoMotions 的 `deployment/tracker_inputs.py` 和 `test_casbot02_tracker_mujoco.py` 已有参考对齐和 sim2sim 组织逻辑，可参考复用。但那套导出模型输入为 ProtoMotions 的语义张量，不能直接读取这份 SuperMimic checkpoint，也不能照搬其 ONNX 输入协议。

当前 SuperMimic 仓库没有该 checkpoint 的完整硬件部署程序，硬件 SDK 和根部状态估计接口也没有在此次检查中验证。现有 Casbot02 仿真报告为 20/40 个完整片段通过，且有首帧速度异常及参考关节超限问题；本说明不是策略已达到真机部署质量的证明。应先处理数据与起始状态问题，并选择仿真稳定的参考段。
