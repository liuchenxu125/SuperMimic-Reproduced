# Casbot02 motion＋velocity 联合训练

独立入口 `scripts/velocity/train_joint.py`，只新增 `src/supermimic/velocity/`；不修改原 `sm-train`、motion＋human任务、环境/奖励、机器人XML、电机参数或部署包。新任务只联合motion和velocity两路，不包含human。训练会同时改变两路encoder及共享decoder的权重，因此motion能力需要随checkpoint重新评测。

## 原作者实现与本地适配

参考 [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic/tree/abc124c1bd5976341034668af3d12c087f2f1f55)，主要是 `tasks/task_spec.py::make_velocity_task`、`models.py::MultiEncoderAE`、`envs/multi_task_world_model_env.py`。

- velocity输入为根部机体系 `[vx, vy, wz]`，单位m/s、m/s、rad/s，不是世界系速度。
- 指令来自动作参考的根部世界线/角速度，先做25帧居中均值，再用实际仿真/想象根部姿态旋转到机体系。每步加独立均匀噪声半宽 `[0.5, 0.5, 0.785]`，最后裁剪到x `[-1.5,1.5]`、y `[-0.5,0.5]`、z角速度 `[-3.14,3.14]`。
- 居中均值只用于已知的离线训练动作，不代表部署速度命令需要未来帧。与上游的一个明确差异：本地均值在每条clip内夹住，避免把拼接后的其他动作平均进来。
- 两路分别编码到32维tanh latent，共享现有decoder。默认仿真环境和策略想象batch各按50%分配；`--batch-size`是两路合计，不是每路各这么大。
- 两路都使用现有motion tracking奖励、终止、critic及BPTT目标。没有额外加入vx误差、高度/站姿、足端或能耗奖励。critic保留原完整motion目标和状态作为训练特权信息，部署不使用critic。
- velocity额外有权重0.1的latent BC：向同一状态/参考上的motion latent对齐。teacher停止梯度；BC不更新motion encoder或decoder，也不通过command反向修改状态。正常BPTT则同时更新两路encoder和共享decoder。
- motion保留 **930维历史＋690维目标→27动作**，不换成官方的观测或目标。原10帧term-major排列、尺度、每帧6D参考姿态均保留。
- velocity使用 **870维本体历史＋3维速度指令→27动作**。它没有动作参考，因此不能读取motion的60维历史参考姿态。内部给共享930维decoder补固定identity rot6d；不修改motion的那部分输入。已验证更换motion目标/6D参考姿态不会改变velocity输出。
- velocity command有独立的运行均值/方差，只在真实velocity采集槽更新；motion仍用原固定尺度。世界模型在混合采集的replay上按原目标训练。
- 继承本地TF32 float32、UTD=0.4、32步想象、250k学习率衰减、默认320k外层更新。`--compile default`沿用本地WM和想象单步编译，不代表整套Python更新都被编译。

## 从零训练

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/velocity/train_joint.py \
  --from-scratch \
  --run-name casbot02_seed_motion_velocity \
  --npz-dir data/npz/bones_seed_casbot02_3mm \
  --manifest data/npz/bones_seed_casbot02_3mm/manifest_filtered.csv \
  --num-envs 2048 --batch-size 2048 \
  --wm-batch 2048 --wm-batches 10 \
  --library-device cpu --library-transfer pinned \
  --velocity-fraction 0.5 --compile default \
  --iterations 320000
```

此命令motion/velocity各1024个采集环境、每次想象各1024条，共享一次actor/critic更新。不是human入口的两路独立batch。默认无验证集；所有manifest保留的机器人动作参与训练，不需要人体骨架配对。默认no-anchor-position＋每帧anchor-orientation、primitive碰撞、episode执行器延迟、单CPU计算线程；未修改原任务的默认值。

可用 `--check-only` 只检查选中的数据/接口；`--data-limit 2` 用于短测，不应放到正式全量指令。另可设置 `--command-window`、`--command-noise X Y Z`、`--lin-x-range LO HI`、`--lin-y-range LO HI`、`--ang-z-range LO HI` 和 `--latent-bc-weight`。

4096环境＋总batch4096时，新增velocity/latent BC分支会提高BPTT显存峰值。独立velocity入口在每次actor/critic更新返回并同步后，调用一次 `torch.cuda.empty_cache()`，把未使用的PyTorch缓存交还给CUDA，供下一轮Warp物理图使用。不是每个物理步都释放；模型、优化器、replay等活动张量不释放，batch、损失和采样设置不变。代价是后续更新可能重新申请显存。这不能消除活动张量本身超出容量的OOM；原motion与human入口没有改动。启动日志和 `velocity_config.json` 中的 `cuda_cache_handoff` 记录此行为。

之前失败的 `logs/casbot02_seed_motion_velocity` 已存在配置但没有checkpoint，重新从零训练请换run-name（例如 `casbot02_seed_motion_velocity_v2`），保留原记录。需要4096时，把上述命令的 `--num-envs` 与 `--batch-size` 都设为4096，其余不变。

若希望从旧motion权重开始，把 `--from-scratch` 换为 `--init-checkpoint logs/casbot02_bones_seed_proprio_cpu7/ckpt_0096000.pt`，并用新run-name。只加载旧WM/motion/decoder/critic权重，velocity encoder新建，优化器和计数重新开始；不是resume。要求同样的motion观测和goal维度。旧motion checkpoint没有可靠的全套资产哈希，使用者需要选与当前机器人资产相匹配的权重；新联合checkpoint会保存当前电机及编译模型身份并在resume核对。

## 保存、恢复与日志

每4000轮保存snapshot及latest；正常结束、Ctrl+C保存latest。异常退出不把失败权重覆盖到latest。原有 `train_ret`仍是存活步平均训练统计，不能当整段完成率。

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/velocity/train_joint.py \
  --resume logs/casbot02_seed_motion_velocity/ckpt_latest.pt \
  --iterations 320000
```

恢复WM、双encoder、共享decoder、critic、三套优化器、velocity归一化统计、采样器、计数和RNG；replay与物理状态不保存，恢复后按原流程重新填充40步。严格检查速度配置、文件元数据/manifest身份及机器人参数，不能用旧 `sm-train --resume` 恢复该任务。可显式调整batch等资源设置，但会影响采样统计。

TensorBoard保留原标签；新增 `velocity_metrics.jsonl` 记录 `motion_imag_return`、`velocity_imag_return`、`velocity_imag_survival`、`velocity_latent_bc`和两类采集计数。它们是世界模型里的想象结果，不是纯速度控制的真仿真误差或真实动作完成率。

## 导出双接口

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/velocity/export_joint.py \
  --checkpoint logs/casbot02_seed_motion_velocity/ckpt_latest.pt \
  --output-dir outputs/casbot02_motion_velocity_export
```

输出 `motion_actor.pt`（930＋690）、`velocity_actor.pt`（870＋3）和manifest。速度指令直接传入，无需motion文件、未来参考或根部位置；输入排列/单位按manifest。导出不替换现有部署权重。

同时输出 `motion_eval_checkpoint.pt`，过滤velocity encoder并去掉优化器，可用于原 `scripts/eval_benchmark.py` 评测motion分支；它只用于推理/评测，不能作为resume文件。原完整joint checkpoint直接交给原motion-only加载器会因额外encoder键而失败，这是预期行为。

## 当前验证范围

已通过速度平滑边界/机体系/噪声、motion不变性、velocity无参考泄漏、BC梯度隔离、两路BPTT/critic更新及多batch导出检查。GPU已做32环境/32步的3轮更新并恢复到5轮，以及2048环境/总batch2048/32步/default编译的2轮更新（2条动作、每轮1个WM batch）。这些是功能短测，不证明全量训练收敛或速度跟踪已达部署水平；全量SEED的额外CPU速度表约每帧6个float32，需约1.1GB内存。

2026-11-16显存交接修复后，10项velocity测试通过。正式入口使用完整manifest中的126608条动作、4096环境、总batch4096、WM batch2048×10、H32、default编译、CPU库＋pinned搬运，连续完成10轮并正常保存checkpoint；修复前同一故障点可复现Warp graph launch OOM。验证记录见 `outputs/velocity_memory_fix_validation_20261116/validation.json`。这是启动/更新阶段回归，不是320k轮长训练的显存或收敛保证。

本次没有启动正式长训练、修改机器人部署或加入PICO/human第三接口。
