# Casbot02：训练检查点到部署包

训练的 `ckpt_*.pt` 包含策略、世界模型、critic 和优化器状态。真机 C++ 控制器只运行策略 actor，不能直接读取这个训练检查点。

处理流程：固定检查点副本 → 导出 actor → 打包参考动作与控制参数 → 本地仿真验证 → 备份并更新部署目录 → ARM64 打包 → 上传 Orin。

部署包包含：

- `actor.onnx`：策略推理，当前为 `obs_hist[1,870] + goal[1,690] → action[1,27]`。
- `motion.bin`：SMREF002 原始参考轨迹，存根部位姿、关节位置/速度和世界系线/角速度；C++ 以当前参考航向重建 10×69 目标窗口。
- `manifest.json`：关节映射、默认位置、Kp/Kd、action_scale、限位、控制频率和检查点哈希。
- `supermimic_motion_bank/`：所有配置动作共用一份 ONNX，分别保存轨迹和 manifest。

世界模型和 critic 不参与真机推理。策略仍为 50 Hz，SDK 的 1 kHz 命令发布和 PD 适配由现有 C++ runner 执行。

当前本地部署（2026-11-04）为 `casbot02_bones_seed_proprio_cpu4/ckpt_0092000.pt`，
默认踏步。真机 X 循环以 `rl_controller/config/supermimic_walk1_subject2_proprio.yaml` 为准：踏步、TongYongDongTaiQingTian_casbot02-bass1_6R7R2R1R_hand01_whole_50hz_v2、dance_phew_001__A464、SEED_扭胯、SEED_武术、SEED_迪斯科、X1_1000HZ、后退、原地左转、原地右转。`CASBOT02_YKS_04_1000HZ` 仍在动作库，不在该循环中。
三个原地上肢动作的映射见 `outputs/casbot02_seed_small_motions_20261102/deployment_update.json`。三条原地小舞为 `SEED_捂脸`＝`dance_hiphop_dab_R_fast_001__A317`、`SEED_扭胯`＝`dance_party_hips_003__A467`、`SEED_武术`＝`dance_wutang_is_forever_003__A466`，映射见 `outputs/casbot02_seed_dances_20261102/selection.json`。混合导出数据入口 `export_npz/` 通过符号链接汇总参考，训练数据目录保持原样。
输入总维度 1560，不增加实测根部定位或相对朝向反馈。新目标顺序与训练一致：
关节位置偏差 27、关节速度 27、参考航向位移 3、参考航向 rot6d 6、未来骨盆系线速度 3、角速度 3。
本地通过起步 10 秒 C++/PyTorch 一致性和当前 X 循环的 10 组相邻显式 PD 短切换；不是全部动作整段评测或真机验证。
产物和旧包备份：`outputs/casbot02_bones_seed_cpu4_092000_deployment_20261104/`。
上一包 `casbot02_bones_seed_proprio_cpu2/ckpt_0096000.pt` 的记录仍在 `outputs/casbot02_bones_seed_096000_deployment_20261102/`。

以下为历史 2026-10-30 的 80000 次模型记录，不能作为新权重验证结果：
`casbot02_lafan1_proprio_paper_kpkd_primitive_data1/ckpt_0080000.pt`，
默认“踏步”，保留用户修改后的 15 条动作菜单顺序。全部 48 条 GPU Warp 整段评测为
41/48，平均回报 0.501050，按帧数加权 0.433132；SDK 动作 8/8，菜单 14/15。
踏步完整 597 帧、默认 10 秒 C++ 起步及一致性检查通过。
`walk3_subject1` 首帧参考速度初始化失败；13/15 相邻切换短测通过，但 walk2→walk3 和保留历史连续链失败，
零速度起步通过不能替代切换验证。**walk3 切换不要用于真机**。本次按用户要求更新本地权重包，保留菜单和故障阈值，未标为全菜单验证通过。
新训练沿用已对齐的关节范围和新参考，电机参数与接口不变。
ARM64 包已构建，尚未上传 Orin 或真机验证；记录与旧包备份位于
`outputs/casbot02_data1_080000_deployment_20261030/deployment/`。
后文 68000 命令与产物目录保留为历史示例；当前启动日志应为
`checkpoint_iteration=80000`，动作为“踏步”。

## 一条命令处理

在本地项目目录执行，后续换模型只需修改 `--ckpt`：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
env -u PYTHONPATH uv run python scripts/update_casbot02_deployment.py \
  --ckpt logs/casbot02_bones_seed_proprio_cpu2/ckpt_0096000.pt \
  --npz-dir outputs/casbot02_seed_small_motions_20261102/export_npz \
  --initial-motion 踏步 \
  --motion-order 踏步 BS_XiaoDongTai_TaiJiaoPanDing_100HZ X1_1000HZ SEED_鼓掌 SEED_看时间 SEED_思考 SEED_捂脸 SEED_扭胯 SEED_武术 SEED_摆胯 SEED_迪斯科 后退 原地左转 原地右转 \
  --apply --build-arm64
```

默认沿用部署 YAML 中的默认动作、完整动作库和手柄顺序；NPZ 目录从检查点的 `data.npz_dir` 读取。需要指定其他同源数据目录时加 `--npz-dir PATH`。

从旧 640 维包第一次迁移到新 690 维包时，额外加 `--accept-goal-update`；只允许已实现的这一次编码迁移，其他接口/电机变化仍拒绝。
BONES 检查点的数据目录不含这六条 SDK 参考，因此本次明确指定已有动作的 NPZ 目录；这些参考没有被加入该检查点的训练。
`--motion-order` 精确替换 X 循环，其他 `motions` 中的动作仍保留。默认拒绝电机、接口或关节范围发生变化。已核验新训练 XML 的范围变化时，可显式加
`--accept-joint-limit-update`，变更前后范围会记入记录；此选项不放开电机参数或维度变化。
本次隔离切换预检因 walk3 相关两项失败而停止自动替换；后续按用户明确要求安装本地固定导出，
失败报告、补充连续链失败及其适用限制均写入记录，未修改菜单或放宽运行保护。记录见上述本次部署目录。

增加动作并插到菜单最前面时，加 `--prepend-motions`（按参数顺序排列，原有动作保留）；改变首次自动启动的动作时，加 `--initial-motion`。本次 68000 次权重更新使用了：

```bash
--prepend-motions 踏步 直行1步态 后退 直行2步态 原地左转 原地右转 X1_Xiao360_100HZ X1_1000HZ \
--initial-motion 踏步
```

默认动作由主包 `manifest.json` 中的 `clip` 决定。沿用 `supermimic_walk1_subject2_proprio.yaml` 和原主包目录作为入口，不代表仍从 walk1 启动。

- 不加 `--apply`：仅在新的 `outputs/casbot02_deployment/.../` 目录导出和验证，不替换当前部署。
- 加 `--apply`：验证通过后，备份旧主包、动作库和 YAML，再替换本地两个模型目录。更新权重来源注释，并在指定 `--prepend-motions` 时更新对应的动作库和菜单配置；其他控制参数保留。
- 加 `--build-arm64`：要求同时加 `--apply`；使用现有 Docker 构建器重新生成 `rl_controller/build/install_arm64/`，并核对安装包中的模型文件。
- `--output PATH` 可指定一个全新的输出目录；已有目录会被拒绝，防止覆盖旧验证结果和备份。

脚本不上传 Orin、不启动 ROS 控制器、不发送电机命令。首次启动、X 的 0.8 秒短切换、B 回位、播放时长与故障阈值沿用 YAML。

## 导出时做了什么检查

1. 检查策略参数 NaN/Inf、870＋690→27 接口，复制固定检查点并校验哈希。
2. 导出原生 MuJoCo/TorchScript 包、ONNX 主包和全部动作参考。
3. 核对当前机器人模型导出的默认位置、Kp/Kd、scale、armature、关节顺序和限位与现有部署契约一致；不一致时停止自动替换。
4. 编译本地 C++；验证整个动作库可加载，保留历史/上一动作的短切换时序正确。
5. 默认动作从静止起步运行 10 秒 C++ ONNX/MuJoCo 闭环，并检查 C++/PyTorch 观测与推理一致性。
6. 按 YAML 菜单验证相邻动作切换：前 5 秒＋配置的桥接时长＋后最多 10 秒（短动作播放至片段末帧），使用整数增益、1 kHz 显式限矩 PD 和 0/5/10 ms 延迟采样。任一组合触发高度/倾斜失败，就不替换部署。

这些属于固定切换时刻、标称仿真参数的短测，不是全部动作完整完成率，也不等同于真机验证。旧检查点没有内嵌完整电机参数，因此还需要保留训练时的 XML/电机文件；不能仅靠 `.pt` 自动恢复历史机器人参数。

输出目录内的 `deployment.json` 记录检查点与 ONNX 哈希、动作清单、默认动作、手柄顺序、旧包备份位置和是否构建 ARM64。各步骤日志位于 `logs/`，默认动作闭环结果为 `cpp_default_10s.json`（历史版本名为 `cpp_walk1_10s.json`），切换结果为 `x_transitions.json`。

## 本次 68000 次模型的产物

```text
outputs/casbot02_primitive_data_068000_15motions/
  ckpt_0068000.pt        固定快照
  sim2sim/              原生 MuJoCo 可视化/验证包
  hardware/             导出的默认动作主包
  motion_bank/          15 条动作，共用一个 actor
  previous_deployment/  更新前的部署包与配置备份
  deployment.json       导出和验证记录
```

查看仿真：

```bash
env -u PYTHONPATH MUJOCO_GL=glfw uv run python \
  outputs/casbot02_primitive_data_068000_15motions/sim2sim/sim2sim.py \
  --clip 踏步 --render
```

## 上传和启动

先停止 Orin 上旧的 SuperMimic 控制器，再在本地上传构建好的安装包：

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced/rl_controller
./scripts/install_remote.sh
```

在 Orin 上按原流程启动底层 SDK，然后启动新控制器：

```bash
source /workspace/rl_controller/setup.bash
bash /workspace/rl_controller/scripts/run_supermimic.sh --ros-args -p dry_run:=false
```

当前启动日志应显示 `checkpoint_iteration=96000`、`goal=690`、`total_input=1560`，默认动作为 `踏步`。这是固定导出的权重，训练继续更新不会自动改变正在部署的策略。

## 需要逐步执行时

脚本依次调用现有工具，可用 `--help` 查看每一步参数：

```bash
env -u PYTHONPATH uv run python scripts/export_sim2sim.py --help
env -u PYTHONPATH uv run python scripts/export_hardware.py --help
env -u PYTHONPATH uv run python scripts/prepare_deployment_motions.py --help
env -u PYTHONPATH uv run python scripts/validate_hardware.py --help
env -u PYTHONPATH uv run python scripts/validate_motion_transition.py --help
```
