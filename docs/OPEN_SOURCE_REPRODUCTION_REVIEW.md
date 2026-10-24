# 两个 SuperMimic 开源复现的借鉴分析

核查日期：2026-10-24。仅检查公开源码、作者公开结果和本地接口，并执行一个 CPU 旋转梯度回归检查；没有复跑第三方训练，没有修改或暂停当前训练。

固定版本：

- `panhainan2020/SuperMimic-MuJoCo`：[`db045afa9c52458f213e2c66037387e9c109a03a`](https://github.com/panhainan2020/SuperMimic-MuJoCo/tree/db045afa9c52458f213e2c66037387e9c109a03a)。
- `YanYuan98/Supermimic_mod`：[`6932f1cd33532b4e84ff173df926d8195a58c45a`](https://github.com/YanYuan98/Supermimic_mod/tree/6932f1cd33532b4e84ff173df926d8195a58c45a)。

按“只优先补齐原论文明确项”的进一步核查见 [PAPER_FIDELITY_GAPS.md](PAPER_FIDELITY_GAPS.md)。下文包含可选工程优化和论文未明确的实现选择，并非全部都属于复现必做项。

## 结论

继续以本地项目承载 casbot02 的机器人、电机、数据、数值修复和部署接口。优先借鉴第二个项目的评测修正、漂移诊断、训练计数和批量想象头；借鉴第一个项目的 CPU/GPU 共用环境语义、模型契约及公开评测流程。相对朝向应作为单独实验选项，不替换现有无相对观测基线。

## 1. 两者实际做了什么

| 项目 | 源码核查 | 作者公开结果，未独立复现 | 与本地的主要区别 |
|---|---|---|---|
| SuperMimic-MuJoCo | 有 WM、actor、critic 更新；原生 MuJoCo CPU 和 MJLab GPU 后端；基于固定版 supermimic_vibed 移植 | G1 98k 更新：LAFAN 40/40、回报 0.8713；100STYLE 3569/3573、回报 0.8045 | 高分权重使用 `legacy99`：每帧本体观测 93 + 相对参考朝向 6，并保留 10 帧历史；目标 73×10。不是其可选的 `paper93` 实验配置 |
| Supermimic_mod | 延续本地相近的模块结构；有完整训练闭环；none / ori / pos_ori 观测消融和多种子记录 | 3 h：v2-none 0.564、5/40 失败；v2-ori 0.803、1/40 失败。v4 默认三个种子 0.843/0.843/0.848，均 0/40 失败 | 默认相对朝向进入 goal；默认 traj 目标、running normalization、ELU、解析终止、bf16，并关闭自碰撞 |

证据：[第一个项目的训练循环](https://github.com/panhainan2020/SuperMimic-MuJoCo/blob/db045afa9c52458f213e2c66037387e9c109a03a/supermimic/train.py)、[MDP/观测](https://github.com/panhainan2020/SuperMimic-MuJoCo/blob/db045afa9c52458f213e2c66037387e9c109a03a/supermimic/mdp.py)、[公开报告](https://github.com/panhainan2020/SuperMimic-MuJoCo/blob/db045afa9c52458f213e2c66037387e9c109a03a/results/098000/REPORT.md)；[第二个项目的配置](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/config.py)、[实验记录](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/docs/EXPERIMENTS.md)。

## 2. 优先借鉴：评测分母与漂移诊断

本地 `src/supermimic/eval/benchmark.py:107` 仍按已执行步数归一化。提前失败时，这描述的是存活部分的平均奖励，而不是整段动作的完成质量。原论文 §4 明确要求提前终止使得分降低。

例：1000 步动作只完成 100 步，每步回报 0.08、理论上限 0.1。存活口径是 0.8，整段口径是 0.08。第二个项目保留两者，但主分数除以整段长度，汇总也按原始长度加权；第一个项目同样按整段长度归一化。

优先同时输出：整段归一化回报、存活段平均回报、完整完成率、记录覆盖率、全局 XY/L2 误差、yaw 误差、关节误差、去除 XY/yaw 后的局部身体误差。不要以局部对齐误差替代全局路径误差。

第二个项目的 `scripts/diag_drift.py` 对比世界坐标速度误差与校正 yaw 后的误差，可以区分航向漂移和局部动作错误。其 `exp(-mean(error))` 诊断量不等于 `mean(exp(-error))` 的真实奖励；借鉴时应把诊断量和实际奖励分别命名。

证据：[修正后的评测](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/eval/benchmark.py#L103)、[漂移诊断](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/scripts/diag_drift.py)、[轨迹误差定义](https://github.com/panhainan2020/SuperMimic-MuJoCo/blob/db045afa9c52458f213e2c66037387e9c109a03a/tools/trajectory_mae.py)。

此前本地 20.8 万步 CPU sim2sim 报告中的 36/40 完成率与该奖励分母无关，仍成立；0.53 是训练期的回合平均回报，不能直接与对方的 0.85 整段确定性评测回报比较。

## 3. 优先借鉴：把训练计数分清

本地每采集一个向量化仿真步，`iteration += 1`；按 UTD=0.4 决定是否做外层 WM/策略更新。两个项目均采用一个外层训练更新为一次 iteration，每次收集平均 2.5 个向量化仿真步。这与原论文 Algorithm A.1 的每迭代更新 WM 和策略的描述更一致。

本地固定检查点 `ckpt_0208000.pt` 的元数据是 `iteration=208000`、`num_updates=83184`。不能把这个 20.8 万与第一个项目的 9.8 万直接比较。按本地默认预算，32 万仿真步约有 12.8 万外层更新；学习率在 25 万仿真步、约 10 万更新时已经衰减完，而另一解释是衰减跨 25 万外层更新。

后续应分别记录 env_steps、WM 外层更新数、actor 更新数、处理的窗口数、墙钟时间，给检查点标明计数单位。修改计数时必须显式处理旧 checkpoint 的恢复与学习率，不应直接给运行中的任务换定义。

此外，本地 actor loss 为 `-J/H`，第二个项目默认 `-J`，与论文 Eq. 2 的写法一致。缩放会改变梯度裁剪及 Adam epsilon 的相对作用，不能简单解释为“学习慢 32 倍”；建议作为独立对照项。

证据：[第二个项目的训练循环](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/training/loop.py#L219)、[actor loss](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/imagination/policy_trainer.py#L86)。原论文核对自本地 `~/Documents/SuperMimic.pdf` 附录 A/D。

## 4. 优先借鉴：计算等价的训练提速

第二个项目的 `ImagStep` 只保留下一步动作必须用到的策略、动力学、观测与历史更新；`ImagHeads` 在整个 rollout 后，对堆叠状态批量计算全身 FK、奖励、终止和 critic 特征。由于动作不依赖当步奖励，计算图仍可保留完整 BPTT。作者报告想象阶段 2.5×、整体训练约 1.8× 提速；这些是其硬件上的结果，不是对本地 4090 D 的保证。

这比先改策略观测更适合作为首个训练性能优化。移植时须验证相同输入与噪声下状态、奖励、存活权重、critic 自举和 actor 梯度的等价性。批量全身 FK 也可能改变显存峰值，本地仅余约 0.8 GB 显存时不可直接假定可容纳。

可一并检查 fused Adam、减少逐标量 GPU→CPU 同步和整体编译世界模型窗口。bf16 只作为单独性能对照，保持状态积分、FK、奖励为 fp32，并保留所有有限值保护。

证据：[批量想象头](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/imagination/compiled.py)、[作者吞吐记录](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/docs/EXPERIMENTS.md)。

## 5. 值得单独消融：相对朝向与未来轨迹

两个项目的高分配置都提供了参考相对机器人朝向。第一个项目把 6 维旋转表示并入每帧历史观测；第二个项目把当前 6 维相对锚点朝向加在 goal 末尾，原本体观测仍为 93 维。

对于 casbot02，可借鉴三模式：none（当前 870+640 接口）、ori（保留当前每帧参考特征时为 870+646）、pos_ori（870+649，需要实际位置估计）。这是新增实验接口，现有 640 维 goal 权重不能直接变为 646 维继续使用。

ori 不要求根部位置估计，但需要可用的姿态四元数、正确的 IMU→base 变换和启动航向对齐。仅有重力方向不能恢复 yaw；若 IMU 航向随时间漂移，相对朝向也会受影响。

需要修正“Table A.1 不列相对朝向，因此论文 g_t 一定没有它”的推断：A.1 只列本体观测 o_t，论文没有逐项列全 g_t。第二个项目也明确承认 ori 是对论文的解释而非官方确认。应保留 none 对照，不能仅凭这两个仓库断言论文用了哪一种。

traj 目标还加入未来参考根相对当前参考帧的位移/旋转。这不引入实际机器人位置输入，但增加模型输入维数；作者现有对照未显示稳定独立收益，优先级低于诊断与朝向消融。朝向反馈也不能直接保证消除 XY 漂移。

证据：[三模式定义](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/goal_spec.py)、[实验及解释限制](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/docs/IMPROVEMENT_PLAN.md)。

## 6. 可以借鉴，但证据不宜夸大

- 策略输入 running normalization + ELU：作者的组合消融显示主要改善前期学习，并非已证明单独归一化有效；导出时必须保存并冻结统计量。
- WM running normalization、LayerNorm、解析终止：作者单种子消融中没有可测的稳定收益，不宜把它们当作当前漂移的主要解决方案。
- WM-only 预热 500 次更新、想象起点离 clip 尾部至少 H 帧：可以单独评估，改变训练过程或采样分布，需在复现说明中列明。
- 碰撞/约束 overflow 日志：第二个项目记录实际溢出比例，比静默降低 nconmax/njmax 更有参考价值；它的 G1 缓冲大小不能直接照搬到 casbot02。
- 第一个项目的机器人/控制配置契约、checkpoint SHA 与模型兼容校验、原子保存、CPU/GPU 共用奖励与终止语义，适合完善部署前一致性检查。本地已有部分 manifest 与哈希，可补差而不是重写。
- 默认关闭自碰撞不能当作论文要求：删除自碰撞惩罚与关闭物理自碰撞是不同操作。casbot02 的模型和电机参数继续使用已有适配。

## 7. 必须保留本地数值修复

第二个项目的 `_sq_quat_err` 仍调用 `mjlab ... quat_box_minus`。在本地现有依赖中加载其固定版本源码，对相同单位四元数进行 CPU 反向传播，前向为 0，梯度四个分量均为 NaN。本地修复版本同一输入前向为 0、梯度全 0 且有限。这只是指定函数和本地依赖的边界回归，不证明对方公开训练在其环境中失败。

第二个项目 actor/critic 更新没有本地统一的 loss/gradient 有限值保护；第一个项目对 actor 梯度范数非有限时跳过 step，但 WM/critic 没有同等保护。两者都不能替代本地已验证的稳定旋转梯度与三套优化器保护。

证据：[第二个项目的四元数奖励](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/envs/mdp/rewards.py#L69)、[第一个项目的优化器更新](https://github.com/panhainan2020/SuperMimic-MuJoCo/blob/db045afa9c52458f213e2c66037387e9c109a03a/supermimic/train.py#L260)。

## 8. 结果可比性与建议顺序

不能把对方 G1 原始数据结果直接当作 casbot02 清洗数据上的预期效果。也不能混比 MAE/RMSE、XYZ 分量 MAE/L2 距离、动作等权/逐帧加权、训练回报/整段评测回报。第一个项目公开的 27.29 cm 为 XYZ 分量 MAE，其同报告的根位置平均 L2 为 64.31 cm；均不是本地 4.64 m 的同机器人同协议对照。

建议顺序：

1. 保留当前 none 训练及 20.8 万步评测，修正未来主评测口径，加入 yaw 与局部/全局误差诊断。
2. 明确并统一 env-step / update / 学习率 / checkpoint 的计数，给后续实验使用；显式处理旧 checkpoint。
3. 移植批量 heads 等性能优化，先证明输出和梯度等价，再做相同硬件吞吐与显存对照。
4. 单开 ori 实验，其他配置和机器人数据不变，与 none 对照。不要一次混入观测、归一化、终止、碰撞等全部变化。
5. 验证多个 checkpoint 和多个种子后，再决定是否切换部署策略接口。
