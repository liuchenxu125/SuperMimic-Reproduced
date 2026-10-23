# 本地补齐的数据与回放模块

公开仓库的 `.gitignore` 原先使用 `data/`，同时忽略了根目录数据、`src/supermimic/data/`
和 `scripts/data/`。本地改为 `/data/`，并依据现有测试、调用方与固定版本 mjlab 实现缺失模块。
这些是重新实现的代码，不代表恢复了作者原实现，也不代表已经复现论文的训练结果。

## 参考仓库检查

检查了 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9)。
它使用 Isaac Lab 3，而当前项目使用 mjlab；其 `supermimic/buffer.py` 返回字典、批次在前，
并通过 mask 屏蔽跨回合的窗口后缀。当前项目需要 `WindowBatch` / `StartBatch`、时间在前，
且世界模型损失没有这个 mask 接口。因此没有直接复制该文件。

## 实现约定

- `motion_convert.py`：30 Hz 的 G1 CSV 转换为 50 Hz NPZ；CPU MuJoCo FK 生成全部 30 个刚体的
  位姿和原点速度。CSV 四元数 xyzw，输出 wxyz；长度单位米，角度单位弧度。
- `ReplayBuffer(num_envs, capacity, device)`：每个环境独立的 `[env, time]` 环形存储，
  `capacity` 包含正常状态行及终态行。单调序号防止把环形覆盖后的数据当作连续轨迹。
- `add_step(...)`：存动作执行前的状态、干净动作和实际执行的带噪动作、该回合仿真参数、
  奖励、物理终止标签、参考帧与回合起点。历史观测使用上一步的干净动作。
- `add_terminal(env_ids, s_next)`：在自动重置前保存真实终态；终态不能作为窗口或想象起点。
  时间上限和 clip 结束会截断回合，但不作为物理终止头的正样本。
- `sample_windows(n, k)`：返回 `(k+1,n,71)` 状态、`(k,n,29)` 动作、`(n,33)` 仿真参数、
  `(k,n)` 终止标签与参考帧。窗口严格不跨回合；终态可以作为最后一个状态目标。
- `sample_starts(n, hist)`：历史由旧到新；在回合起点或已覆盖的数据边界重复最早可用状态，
  与 mjlab 的历史填充约定一致。终态不参与想象起点采样。
- `ReplayRecorder`：重置后缓存初态，`record_pre_reset` 抢在状态/动作清零前写入最后一步及终态；
  `record_post_step` 仅写入尚未被 pre-reset 记录的环境。显式手动重置模式也保留终态。

## 验证

```bash
env -u PYTHONPATH uv run sm-train --help
env -u PYTHONPATH uv run pytest tests/test_replay_buffer.py -q
env -u PYTHONPATH uv run pytest tests/test_env_smoke.py -q
```

`test_replay_buffer.py` 使用可控序列验证终态、超时、环形覆盖、回合隔离、历史填充和动作拷贝；
环境测试需要可访问的 NVIDIA GPU 和 `data/npz/lafan1/`，否则会跳过。
长期训练的数值与最终回报仍需用实际训练评估。

本机 RTX 4090 D 验证：GPU 完整测试集 30 项通过（242 秒），包含 60 步短训练、
checkpoint 保存/恢复、世界模型多步误差下降。随后将记录器测试扩展为自动/手动重置
与物理终止/超时的四种组合，扩展后的回放专项测试 9 项全部通过。
