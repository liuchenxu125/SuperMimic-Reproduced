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
  --robot g1 --input-dir data/seed/g1/csv --pattern '**/*.csv' \
  --output-dir data/npz/bones_seed --input-fps 120 --output-fps 50 \
  --bones-name-filter --workers 8 --progress-every 1000
```

官方 BONES-SEED G1 CSV 为 120 FPS，见 [数据集说明](https://huggingface.co/datasets/bones-studio/seed/blob/main/README.md)。
`--pattern '**/*.csv'` 递归读取日期子目录；输出使用动作名，发现重名时会停止。
`--bones-name-filter` 在耗时的转换前应用同一套名称排除规则，并写入
`manifest_name_filter.csv`（覆盖全部原始动作、原路径和排除原因）。源 CSV 不变。
之后仍需执行下面的 NPZ 可行性检查，并在训练时传入最终 manifest。
转换默认跳过已有输出，可用相同命令续跑；只有源数据和转换参数未变时才能这样续跑。

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

## Casbot02 BONES-SEED 转换与过滤

重定向后的无表头 Casbot02 CSV 使用米、xyzw 四元数和 27 关节弧度。
`hl_robot_retargeter/scripts/batch_seed_to_casbot02.py` 导出的 SEED 保留 120 FPS。
下面的入口先按名称排除不适合平地的动作，再以训练模型做 FK 和可行性检查，
仅把通过的动作写成 50 Hz NPZ；原 CSV 不变。

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/data/filter_bones_seed.py \
  --robot casbot02 \
  --csv-dir ~/Desktop/hl_robot_retargeter/output_data/robot_motion/seed \
  --output-dir data/npz/bones_seed_casbot02 \
  --input-fps 120 --tracked-bodies-only --workers 16 --progress-every 5000
```

`--tracked-bodies-only` 保留训练所需的 14 个身体，以及全部 27 关节位置/速度，
所有运动数组仍为 float32。完整身体的可行性检查发生在裁减前；训练的 full/compact
MotionLibrary 都能读取输出。省掉未参与跟踪的冗余身体，不通过降低精度节省磁盘。

Casbot02 过滤沿用名称规则和悬空/穿地/净空阈值，但用它自己的 14 个脚底胶囊。
静止漂浮脚检查的根高度候选范围按名义站立高度相对 G1 0.78 m 换算，脚底悬空阈值仍为
0.02 m。这是跨机器人适配，不声称复现了论文完全相同的保留动作清单。
不额外截帧、裁剪关节角度或修改动作姿态。

输出 `manifest_filtered.csv` 记录全部输入动作及排除原因，`summary.json` 记录统计。
`preparation.json/jsonl` 保存输入清单、参数和已完成动作；相同命令可以断点继续，
输入、模型或参数变化时要求使用新输出目录。训练时指定：

```bash
--robot casbot02 \
--data.npz-dir data/npz/bones_seed_casbot02 \
--data.manifest data/npz/bones_seed_casbot02/manifest_filtered.csv
```

大动作库可使用 `--env.compact-library --env.library-device cpu`：参考状态保存在
CPU 内存，按需将当前帧和目标窗口传到 GPU；FK、模型、经验回放和仿真仍在 GPU。
CPU 存储沿用 compact 的 FP16 参考缓存，查询返回 FP32，不修改源 NPZ 或训练目标。
每次启动仍须加载全部 NPZ，首次加载十几万文件需要数分钟；这不是磁盘缓存。

已转换 NPZ 也能单独复核：

```bash
uv run python scripts/data/filter_bones_seed.py --robot casbot02 \
  --npz-dir data/npz/bones_seed_casbot02 --workers 8 \
  --out outputs/casbot02_seed_recheck.csv
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

## Casbot02 BONES-SEED 时间跳变清洗

2026-11-01 已处理 `data/npz/bones_seed_casbot02` 中 1,959 条明显跳变的动作。
规则为相邻 50 Hz 帧的任一关节变化 >1 rad、根位移 >0.2 m、根旋转 >1 rad；
四元数正负号等价处理。这是本次数据诊断规则，不是 SuperMimic 论文过滤规则。

仅前 0.1 秒存在跳变的 280 条裁去 5 帧，仅前 0.5 秒存在跳变的另 183 条裁去 25 帧；
其余 1,496 条沿全部异常边切开，保留跨度至少 1 秒（51 帧）的连续片段。
不连接切口两侧，不夹紧关节、不插值修补姿态。4 条无可用片段退出清单。
清洗后的数据为 128,325 条、46,382,547 帧，源 CSV 保留。

`scripts/data/clean_temporal_jumps.py` 先生成暂存文件及 `plan.json`，
逐片段检查有限值、关节不变、导数不跨切口及原生 MuJoCo FK/身体速度；
再用 `--apply` 安装。安装前校验原 NPZ/清单 SHA256、源 CSV 大小与时间戳，
备份全部受影响原件，使用原子替换，拒绝重复应用已完成计划。
支持原来的全身体 NPZ 和本次 14 身体紧凑存储；本次全量 `--verify` 针对 Casbot02 27 关节/14 身体。

原件备份在 `outputs/bones_temporal_cleaning_20261101/backup/`，
使用硬链接节省磁盘；不要原地修改备份文件。
逐条原帧范围（左闭右开）及 SHA256 在同目录 `plan.json`，
每个新 NPZ 也记录原名、原 SHA256 和帧范围，派生片段使用 `__clean_partNN` 后缀。
全量复核结果在该目录 `validation.json`。

训练推荐显式使用 `data/npz/bones_seed_casbot02/manifest_temporal_cleaned.csv`；
`manifest_filtered.csv` 也已同步更新。原转换的 `summary.json`/`preparation.jsonl`
保持历史记录，当前统计见 `temporal_cleaning.json`。切勿在清洗后目录直接重新执行旧 CSV 转换任务。
训练命令及复核命令见 [训练流程](../训练流程.md)。动作分段改变采样分箱，使用新训练运行名。
