# CPU 参考库运行配置（2026-11-07 对齐 cpu4）

`--env.library-device cpu` 只把参考数据存放在内存；物理仿真、世界模型、策略和反向传播仍在 GPU。

## 已确认的差异

- `casbot02_bones_seed_proprio_cpu4`：PyTorch 2.14.1+cu130，CPU 参考库，intra-op 1、inter-op 24，OMP/OPENBLAS/MKL 均为 1；完成约 92350 次更新。
- `cpu5` / `cpu6`：同一 PyTorch 版本和 CPU 库查询实现，intra-op 变成 24、inter-op 24，没有上述环境线程限制；分别最后记录到 2650 / 400 次更新。
- cpu4 是 870 维历史，cpu5/6 是 930 维历史。姿态历史在每次策略更新开始时额外查询 `10 × batch_size` 帧参考四元数；batch 4096 对应 40960 帧。
- GPU 常驻参考库的 LAFAN1 姿态任务也使用 24 个 CPU 线程，已训练到 88000 次。因此不能认定 6 维姿态或 24 线程单独就是错误原因。
- 本机 torch、Warp、MuJoCo 的安装时间均为 2026-10-22；现有记录不支持“昨天依赖刚升级”的结论。cpu4 的 runtime.json 没有记录历史驱动版本，无法据此证明驱动从未变化。
- 两次最新失败在内核中记录为 `libcuda.so.595.99.02` 段错误；之前还出现过内部设备号 -1。它们不是 OOM 日志，也不是正常的维度校验或非有限梯度异常。

## 当前改动

- CPU 库任务默认使用 1 个 PyTorch intra-op 线程。可用 `--cpu-threads N` 显式覆盖；非 CPU 库任务保留原线程行为。
- `sm-train` / `python -m supermimic.scripts.train` 在 `--env.library-device cpu` 时，先于 Torch/NumPy/BLAS 导入，将 OMP/OPENBLAS/MKL 三个线程环境变量全部设为 1，覆盖终端已有值。GPU 常驻参考库任务保持原有环境。
- 按用户要求，CPU 库默认恢复 `--env.library-transfer pinned`：每批选中参考行调用 `pin_memory()`，再以 `non_blocking=True` 搬到 GPU，与 cpu4 的旧路径一致。
- `--env.library-transfer sync` 仍可显式选择，作为此前同步搬运配置的对照。
- runtime.json 记录实际线程数、选择来源及搬运模式。
- 参考仍用 FP16 存储、FP32 查询；数据、930/690 输入、随机化、奖励、网络、学习率、TF32 与 torch.compile 均不变。

2026-11-06 曾将默认搬运改为 sync 并通过短测；2026-11-07 此次按用户要求恢复 cpu4 的 pinned 路径，同时补齐提前生效的三项线程环境变量。它们不代表已经证明某个 CUDA/Warp/驱动 bug 为唯一根因。

这些设置从下一次启动生效，不会改变已经运行的 cpu7 进程。直接在其他程序里构造 Trainer 时，仍应在导入 Torch/NumPy 前显式设置三项环境变量。

## 建议的新训练命令

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1 \
  uv run sm-train \
    --robot casbot02 --run-name casbot02_bones_seed_ori_cpu4_runtime \
    --data.npz-dir data/npz/bones_seed_casbot02 \
    --data.manifest data/npz/bones_seed_casbot02/manifest_filtered.csv \
    --env.compact-library --env.library-device cpu --env.library-transfer pinned \
    --cpu-threads 1 --env.no-anchor-obs --env.anchor-ori-obs \
    --env.actuator-delay --env.actuator-delay-mode episode \
    --env.collision-model primitive --env.num-envs 4096 \
    --policy-trainer.batch-size 4096 --schedule.checkpoint-interval 1000 \
    2>&1 | tee outputs/seed_cpu4_runtime_train.log
```

如果 run-name 已存在，换新名称或按原接口从其 checkpoint 续训。每 1000 次保存用于减少原生崩溃时的进度损失，不改变训练更新预算。

## 验证记录

- 2026-11-07 本次对齐：6 项环境/真实 CLI 入口测试通过（含导入 Torch 前覆盖已有 24 线程环境、GPU 参考任务不改环境）；5 项 CPU/CUDA 查询、目标、FK 和梯度测试通过（含 pinned/sync 与 FP16/FP32）。未启动或重启实际训练。
- CPU/GPU 参考状态、目标、FK 和策略梯度对照：两种搬运模式及 FP16/FP32 存储均通过。
- 独立旧 pinned/24 线程/编译 backward 压测 3000 次通过，说明简单复制压力不足以稳定复现完整训练的故障。
- 2026-11-06 历史 sync 配置：全量 126759 条 SEED、4096 环境 / 4096 batch / 930 维历史 / 编译开启的诊断正常完成 1000 次更新（2540 个仿真控制步），训练循环用时约 905 秒。世界模型、策略和 critic 的 checkpoint 张量均为有限值，200/400/600/800/1000 次 checkpoint 及 latest 保存正常。详见本地 `outputs/cpu_library_stability_20261106/analysis.json` 和 `full_seed_sync_1000.log`。这不是本次 pinned 默认设置的长期训练验证。
- 本次已超过 cpu6 约 400 次更新的失败位置，但没有超过 cpu5 约 2650 次的失败位置；长期稳定性和唯一根因仍未验证。
- 机器人碰撞资产的原有哈希测试存在与当前本地改动不一致的问题；此次未改动机器人资产，也未改写该哈希清单来绕过测试。

若仍崩溃，保留 console.log、runtime.json、checkpoint 和内核的 sm-train/libcuda 记录，再对编译路径或驱动做单变量对照。不要把“短测通过”解释为长期稳定保证。
