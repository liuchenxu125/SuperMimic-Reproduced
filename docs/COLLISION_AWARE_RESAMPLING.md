# 避免重采样新增自碰撞

原始 30 Hz CSV 的离散帧不相交，不保证关节线性插值经过的姿态不相交。换成三次样条、增加输出帧率或仅对位置做低通滤波，也没有几何避碰保证。

本项目新增的处理流程是：常规重采样 → 全段外观几何筛查 → 局部带碰撞约束的轨迹优化 → 更新关节速度及全部身体 FK/速度 → 保存到新 NPZ。使用 SciPy SLSQP，在接近原目标、修正量平滑、关节限位与几何间距之间求局部可行解；不保证所有输入都可自动修复。距离与约束算法参考 [MuJoCo 凸网格碰撞说明](https://mujoco.readthedocs.io/en/latest/computation/)及 [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。

## 使用方法

在项目根目录运行。输出必须是新文件，脚本拒绝覆盖输入或已经存在的输出。

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/data/repair_resample_collisions.py \
  --input outputs/casbot02_deployed_npz/dance1_subject2.npz \
  --output outputs/dance1_subject2_collision_repair_20261106/dance1_subject2.npz \
  --collision-xml outputs/casbot02_deployment/casbot02_bones_seed_proprio_cpu4_ckpt_0092000_20261106_152440/sim2sim/scene/scene.xml \
  --margin-mm 2 --substeps 4 --clamp-joints
```

示例输出若已存在，请改为另一新路径。脚本读取 `robot`、`fps`、`input_fps` 元数据，要求关节/身体顺序匹配，并核验碰撞 XML 与当前转换模型的运动学及关节范围一致。

- `--substeps 4`：50 Hz 输出节点及帧间 200 Hz 采样点一起约束；输出仍为 50 Hz。
- `--margin-mm 2`：待修正部位目标间隔 2 mm；优化可行性容差为 0.1 mm。其他未进入局部约束的部位仍要通过全段复扫。
- `--max-change-rad 0.15`：每个关节相对输入的最大修正预算，不是必须改这么多。
- `--clamp-joints`：显式允许修正输入已有的关节超限。没有此参数，明显超限会报错。此次样本原有 28 个关节值超限，最大约 0.00198 rad。
- 保留时长和根部位置/姿态轨迹；尽量保留原采样点与输出网格重合的关节姿态。30→50 Hz 每 0.1 秒重合一次；若这些帧自身超限且显式允许限位修复，则会有相应限位修正。
- 正则项约束修正轨迹的一阶/二阶差分。它是平滑代价，不是对执行器速度、加速度或落脚位置的硬约束。

## 检查范围

使用外观三角网格的凸包距离作保守约束，不直接把简化胶囊的正常重叠当作目标错误。同一/焊接 body、XML 显式排除项，以及运动链距离不超过 2 的机械连接界面不参与检查。当前部署 XML 共检查 412 对外观网格。

修复后必须对保存为 float32 的结果再次检查。200 Hz 或更密采样仍是离散验证，不能视为连续轨迹无碰撞的数学证明，也不能证明策略执行、地面接触或动力学可行性。参考修复后，仍应重新做 sim2sim 验证策略实际跟踪。

脚本只生成新 NPZ 和同名 `.report.json`，不会替换训练集、部署动作包或上传机器人。结果中的关节速度、身体位置/姿态、线速度和角速度都基于修复后的关节轨迹重新生成。
