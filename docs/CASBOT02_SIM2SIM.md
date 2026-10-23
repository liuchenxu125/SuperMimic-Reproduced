# Casbot02 SuperMimic 独立 MuJoCo sim2sim

使用 `logs/casbot02_lafan1/ckpt_latest.pt` 的 actor，在原生 CPU MuJoCo 中闭环运行。参考了 ProtoMotions 的 Casbot02 配置和独立 MuJoCo 部署结构；机器人网格、PD 增益、力矩限位、armature 和默认站姿复用本地已复制的同一份资产。

运行时只依赖 `numpy`、`torch`、`mujoco`。不需要加载 mjlab、ProtoMotions、世界模型、critic 或 CUDA。导出时使用训练项目构造场景，运行时读取自包含的 XML、STL、参考轨迹与 actor。

## 已生成的运行包

`outputs/casbot02_sim2sim/` 包含：

- `actor.pt`：CPU TorchScript actor，固定输入缩放已保留，接口为 `870 + 649 → 27`。
- `manifest.json`：checkpoint 哈希、关节顺序、q0、动作缩放、PD/力矩参数、时序和导出误差。
- `scene/scene.xml`、`scene/assets/`：独立机器人与地面场景，不引用 ProtoMotions 的绝对路径。
- `motions/`：默认三个片段的根部状态、关节状态及预计算 64 维参考特征。
- `sim2sim.py`：可单独运行的程序，可连同整个包复制到另一目录。

本机没有安装 ONNX Runtime，因此先使用已验证的 TorchScript CPU 路径；本包不是 ProtoMotions 的 `unified_pipeline.onnx`，两者输入协议不同。

## 直接播放

在 SuperMimic 项目根目录：

```bash
env -u PYTHONPATH MUJOCO_GL=glfw uv run python scripts/sim2sim.py \
  --bundle outputs/casbot02_sim2sim \
  --clip walk4_subject1 --render
```

也可以直接调用包内独立程序：

```bash
env -u PYTHONPATH MUJOCO_GL=glfw uv run python \
  outputs/casbot02_sim2sim/sim2sim.py --clip walk3_subject5 --render
```

蓝色半透明机器人是参考，实体机器人由策略控制。空格暂停；R 重置当前片段；N/B 切换下一个/上一个；1–9 选择片段。**切换会重置机器人状态，不是跨动作平滑过渡。** 默认播放一次，到片段末尾或触发失败阈值后退出。

默认包包含 `walk4_subject1`、`walk3_subject5`、`dance1_subject1`。列出片段：

```bash
env -u PYTHONPATH uv run python scripts/sim2sim.py --list-clips
```

无窗口快速测试并保存报告：

```bash
env -u PYTHONPATH uv run python scripts/sim2sim.py \
  --clip walk4_subject1 --seconds 10 \
  --report outputs/casbot02_sim2sim/check_10s.json
```

不加 `--seconds` 就测试整个片段；无窗口时默认尽快执行，有窗口时默认按实际时间播放。`--no-realtime` 可让窗口也尽快执行。

`--start-frame 5` 显式跳过前 0.1 秒，用于原始数据首帧异常的诊断，不会修改数据，也不能把短段结果当作完整片段成绩。`--start-xy 2 -1 --start-yaw 0.7` 可测试起始位置/航向变化，机器人和整条参考只在开始时一起对齐，之后不会逐步跟随机器人重定位参考。

## 与训练保持一致的边界

- 物理步频 200 Hz、策略 50 Hz、4 子步；采用训练的 implicitfast、Newton 等数值配置，而非源 XML 的 1 ms 步长。
- 本体输入由 qpos/qvel 推导，历史按项拼接，IMU 角速度使用自由基座的局部角速度，四元数为 wxyz。
- 当前 checkpoint 开启 anchor_obs，额外 9 维相对位姿独立计算，位置来自 MuJoCo 真值。真机仍需接入状态估计器。
- 关节顺序为左腿6、右腿6、腰1、左臂7、右臂7；MuJoCo actuator 顺序中腰在最后，发送 ctrl 时按名称/关节地址映射。
- 关节初始化按训练的 0.9 软限位裁剪，初始速度保留参考值，不隐藏修复异常首帧。
- 当前训练命令在 reset 的 command compute 中会前进一帧：机器人从 f 帧初始化，首次目标为 f+1 帧。运行包显式记录 `reset_goal_advance=1` 以匹配现有权重，未修改训练代码。
- 未加入 ProtoMotions 的 EMA、加速度平滑或 tanh，因为当前 SuperMimic 训练没有这些后处理。执行器自身力矩限位保留。原始输出不保证 [-1,1]，报告记录最大动作幅值。
- 推理关闭人工动作/观测噪声及域随机化；足部滑动摩擦固定为 1.0。

CPU MuJoCo 与 MuJoCo Warp 的接触轨迹不保证逐位相同。测试分别验证输入/动作接口一致和原生 CPU 闭环表现，不能将其中一项当成另一项的证明。

## 当前权重的验证结果

| 原生 CPU MuJoCo 测试 | 执行长度 | 高度/倾斜终止 |
|---|---:|---|
| walk4_subject1，从第0帧 | 完整 163.92 秒 | 未触发 |
| walk3_subject5，从第0帧 | 完整 246.62 秒 | 未触发 |
| dance1_subject1，从第5帧 | 10 秒短段 | 未触发 |

完整行走片段的平均逐帧关节 RMSE 分别约 0.1651 / 0.2000 rad，平均根部位置误差约 0.0728 / 0.1881 m。完成片段不表示精确重合参考，也不代表真机稳定性。这些是已有训练动作的测试，不是泛化评估。JSON 报告在运行包根目录。

导出 actor 与原始 CPU checkpoint 在多个输入批次上的最大绝对误差为 0。另用 GPU 训练环境产生 32 个连续真实仿真状态，比较独立输入构造与训练环境：

| 项目 | 最大绝对误差 |
|---|---:|
| 870 维本体历史 | 4.77e-7 |
| 649 维目标 | 1.83e-7 |
| 原 actor 与导出 actor | 0 |
| 关节位置目标 | 7.30e-8 |

结果见 `outputs/casbot02_sim2sim/input_parity.json`。复查此项需要可访问的 NVIDIA GPU：

```bash
env -u PYTHONPATH MUJOCO_GL=egl uv run python scripts/validate_sim2sim.py
```

CPU 专项测试覆盖观测/历史、参考特征、起始 XY/yaw 变换、导出模型、多执行器顺序、非法数据拒绝：

```bash
env -u PYTHONPATH uv run pytest tests/test_sim2sim.py -q
```

本次专项测试为 **5 passed**，原生 GLFW 窗口和参考 overlay 也已运行 3 秒并正常退出。预览图为 `outputs/casbot02_sim2sim/preview.png`。

## 导出其他权重或片段

```bash
env -u PYTHONPATH uv run python scripts/export_sim2sim.py \
  --ckpt logs/casbot02_lafan1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1_casbot02 \
  --clips walk4_subject1 walk3_subject5 dance1_subject1 \
  --output outputs/casbot02_sim2sim_new
```

导出拒绝覆盖已有完整包，给新的权重使用新的目录，保留 checkpoint 和资产对应关系。当前 exporter 只接受 Casbot02 checkpoint，不把 G1 权重静默套用到 Casbot02。
