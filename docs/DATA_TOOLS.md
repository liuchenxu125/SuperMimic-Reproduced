# CSV 转换、BONES-SEED 过滤与分片评测

上游参考：CuboiLeo/SuperMimic-Reproduced `ba61053`（2026-10-27）。
本地工具保留 Casbot02 适配和整段回报口径；不修改现有训练集或自动启用过滤。

## CSV 转换

`scripts/data/csv_to_npz_batch.py` 自动区分：

- LAFAN1：无表头，根位置为米、四元数 xyzw、关节弧度；G1 29 关节，Casbot02 27 关节。
- BONES-SEED：`Frame,root_translateX,...` 表头，位置厘米、xyz 欧拉角度数、关节度数。按关节列名核对顺序，再转换为统一单位。

输出为 50 Hz、wxyz 四元数、世界系身体速度。源帧率必须按数据实际值设置，不能从输出 50 Hz 推断源帧率。
混合帧率可提供 `--fps-map source_fps.csv`（列为 `name,fps`，name 不含扩展名）。
默认不覆盖已有 NPZ；下面的输入路径仅作示例，替换为实际原始 CSV 目录。

```bash
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --robot g1 --input-dir data/raw/bones_seed/g1 \
  --output-dir data/npz/bones_seed --input-fps 30 --workers 4
```

SDK 的 61 列 `.data` 保留独立入口 `scripts/data/casbot02_data_to_npz.py`；不要用本 CSV 入口处理。

## G1 BONES-SEED 过滤

```bash
env -u PYTHONPATH uv run python scripts/data/filter_bones_seed.py \
  --npz-dir data/npz/bones_seed --workers 4
```

生成 `manifest_filtered.csv`，包含 `name,frames,keep,reason`；不删除或改写 NPZ。
关键词包含上游恢复的 cartwheel、handstand、safety_roll 等；另检查持续悬空、穿地及静止漂浮脚。
可行性阈值、G1 脚胶囊和 50 Hz 要求与上游一致，这些阈值仅适用于 G1，不直接用于 Casbot02。
训练/评测必须显式加 `--data.manifest ...` / `--manifest ...` 才会使用过滤结果。
例如 G1 训练使用：

```bash
env -u PYTHONPATH uv run sm-train --run-name g1_bones_filtered \
  --data.npz-dir data/npz/bones_seed \
  --data.manifest data/npz/bones_seed/manifest_filtered.csv
```

## 分片评测及合并

以当前 Casbot02 权重为例，单卡依次运行两片；不会同时占用两个评测进程。
这仍会使用 GPU，先按需要安排训练/评测资源。`--ckpt` 应使用固定编号的检查点，避免分片期间 latest 变化。

```bash
for shard in 0 1; do
  env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
    --ckpt logs/casbot02_lafan1_proprio_paper_kpkd_primitive_data/ckpt_0068000.pt \
    --npz-dir data/npz/lafan1_casbot02_codex \
    --num-shards 2 --shard-index "$shard" --num-envs 256 \
    --out "outputs/benchmark_068000_shard_${shard}.json"
done

env -u PYTHONPATH uv run python scripts/merge_benchmark_shards.py \
  --shards outputs/benchmark_068000_shard_0.json outputs/benchmark_068000_shard_1.json \
  --out outputs/benchmark_068000_merged.json
```

新评测文件记录检查点 SHA256、协议参数、分片编号和完整动作名列表的摘要。
合并拒绝重复动作、不完整分片、不同权重/协议及矛盾的回报口径；主分数始终按完整参考长度归一化。
动作名摘要不等同于 NPZ 内容摘要，分片期间应保持数据不变。
旧本地文件没有分片编号/权重摘要时仍可合并，但会明确提示无法证明完整性/权重一致性。
旧上游存活步数计分 JSON 必须显式加 `--legacy-survived`，按 `旧分数 × steps / frames` 转为整段分数。
不明确旧分数含义时不要使用该选项。支持普通 JSON 和 `.json.gz` 输入。

## 断点续训预热

两边都不把 replay 存入检查点，所以恢复后都必须重新采集。
上游用 `iteration - start_iteration` 判断预热，预热期间仍增加仿真步 iteration。
本地在更新循环前显式补采 `warmup_iterations` 步，单独记录 `replay_refill_env_steps`，
不增加训练更新 iteration，也不推进更新学习率；恢复时还保存采集比例的小数累积器。
因此保留本地实现。`tests/test_paper_alignment.py` 已有恢复前后更新数、采集比例、学习率和优化器回归检查。
