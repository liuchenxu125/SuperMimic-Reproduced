# Casbot02 motion＋human 双接口联合训练

本任务参考 [MorphMimic 的训练循环](https://github.com/lzyang2000/morphmimic/blob/main/src/morphmimic/train/trainer.py) 中 `--kp-interface --kp-student copy` 的当前推荐路线：同一轮采集真实仿真数据、拟合世界模型、更新 motion encoder/共享 decoder/critic，再更新 human encoder。human 更新步骤暂时冻结其余网络；整轮训练并没有冻结世界模型或 motion 策略。

独立入口是 `scripts/human/train_joint.py`。原 `sm-train`、配置默认值、环境/奖励源码、机器人资产和部署包均保持原样。此版本支持单卡、单 Casbot02、`copy`；未实现 `return` human critic、direct-human 或多机器人共享训练。原来的冻结 teacher 后训练见 [HUMAN_TRAINING.md](HUMAN_TRAINING.md)。

## Casbot02 适配契约

- 27 个原生关节，保留关节顺序、q0、Kp/Kd、armature、力矩和 action_scale，不映射成 G1 的 29 关节。
- 本体历史 **930=10×93**：27 关节位置、27 速度、角速度3、重力3、上次动作27、参考骨盆相对朝向6；仍为原 term-major 顺序与固定尺度，无根位置观测。
- G1 的历史也恰好是930，但其每帧是29＋29＋3＋3＋29，不含你这项额外6D骨盆反馈。数值相同不代表语义相同；此任务使用Casbot02自己的布局、尺度和decoder，不能直接套G1权重。
- motion 目标仍为 **690=10×69**，保留当前参考航向系中的骨盆位移/rot6d和骨盆系速度，不采用 G1 的 670。
- 上游G1每个目标帧67维是29关节位置＋29关节速度＋anchor位置3＋rot6d6；本地69维是27＋27＋位置3＋rot6d6＋根线速度3＋根角速度3。这里保留你的两个根速度参考项。
- human 目标 **2254**，24 个 SMPL slot、10 帧、32 维 latent，接同一个 decoder 输出27动作。姿态为 SOMA Uniform/SMPL rest，米、Z-up、wxyz。
- human 朝向反馈以 Spine3（五点时为 Pelvis）相对 `waist_yaw_link` 的朝向构造；torso 姿态由 root quaternion 和真实腰部关节轴计算，已与 MuJoCo FK 比较。
- 世界模型保留本地67维广义状态、66维增量、31维随机化上下文；身体目标/奖励由原FK重建。上游G1用240维全身体状态和33维上下文，这里移植联合更新机制，不替换本地世界模型，也不能直接载入G1的WM权重。
- 原观测中的6D参考骨盆项仍保留，和human目标自己的60维朝向块是独立的两项；训练没有加入机器人实际根位置输入。
- 50 Hz控制、10帧预览、默认32步想象、mixed full/five各约一半、完整预览、jitter/drift关闭。采用现有网络宽度和TF32 float32，未切换bf16。
- 继承初始化 checkpoint 的奖励、随机化、噪声、延迟配置。cpu7/96000对应：地板摩擦0.3、摩擦/编码器偏置startup采样、0/5/10 ms每回合固定延迟、collection动作噪声0.1、primitive碰撞。
- human/robot必须严格配对。新任务默认 `--validation-fraction 0`，全部有效配对均参与训练，不留出SEED验证动作；目前为126576条。显式验证过的起始偏移/length会生成独立裁剪副本；源NPZ不改。

## 准备更多数据：逐条转换，不整包解压

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
env -u PYTHONPATH .venv/bin/python scripts/human/prepare_seed_joint.py \
  --archive /media/casbot/Linux_Drive/supermimic_ws/bones_seed_human/soma_uniform.tar.gz \
  --robot-dir data/npz/bones_seed_casbot02_3mm \
  --manifest data/npz/bones_seed_casbot02_3mm/manifest_filtered.csv \
  --output-dir data/human/seed_uniform_joint_all --limit 0 --workers 16
```

处理全部keep-list中的SEED动作，逐条转换临时BVH后立即删除，只留下human50 NPZ、pairs.csv和审计记录。不是按回报挑选成功动作。帧率、帧数、关节顺序或有限值检查不通过会拒绝，不自动截短。可重跑同一指令继续；已有配对不重复转换。全部归档扫描完成后，真正缺少的人体BVH也会记录为rejected。全量训练要求 preparation_report.json 的 pending=0，避免误用仍在生成的部分pairs.csv。需要小规模排查时可换新目录并用 --limit 1000。

## 正式训练（全量配对完成后）

以下4096/4096配置在本机RTX 4090 D上已出现第一轮联合更新后Warp显存不足，不能视为已验证可用的配置。较低显存方案保持4096环境，改用 `--motion-batch 2048 --human-batch 512`；此前全量联合任务已用该批次组合运行。两种设置都使用全部有效配对，batch大小不改变训练集范围。

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/human/train_joint.py \
  --pairs data/human/seed_uniform_joint_all/pairs.csv \
  --run-name casbot02_seed_motion_human_all_b4096_full \
  --from-scratch \
  --num-envs 4096 --motion-batch 4096 --human-batch 4096 \
  --wm-batch 2048 --wm-batches 10 --iterations 60000 \
  --human-fraction 0.5 --masking mixed --validation-fraction 0 \
  --library-device cpu --library-transfer sync --cache-clips 0 \
  --cpu-threads 1 --compile default
```

上述命令按当前要求从零训练，不加载旧权重。若需要用cpu7初始化，可把 `--from-scratch` 替换为 `--init-checkpoint logs/casbot02_bones_seed_proprio_cpu7/ckpt_0096000.pt`：这时只读取旧motion/WM/critic权重，并验证导出bundle的关节、电机和q0契约；新任务仍从0计数，使用新优化器和采样器。默认bundle跟随当前已验证的导出，可用 `--init-bundle` 显式给匹配路径。电机不匹配会拒绝。

可用 `--init-human logs/human_casbot02_seed20/human_latest.pt` 初始化已有human编码器和归一化，但要求它的teacher SHA完全对应本次motion初始化；默认重新初始化human，避免把20条的小样本过拟合作为强制起点。

默认各2048个环境分别由motion和human驱动；噪声注入、奖励、reset和recorder共用原环境。世界模型学习两类状态。motion步仍采用本地原BPTT目标和critic；human步从真实replay的state、xi、10帧历史、历史参考帧开始，在自己的想象状态上复制当前motion teacher动作/latent，两个损失分别作运行尺度归一化，权重均为1。目标在片段末尾夹住，并屏蔽结束后的padding损失。episode mask在重置时重采样。

已分别验证 --compile none 与 --compile default 的4096环境联合更新。formal命令使用default编译原WM/运动想象步，human copy仍为eager；首次编译会增加时间/峰值显存，未宣称吞吐与上游相同。人类分支批次与motion批次可独立调节。脚本只将本进程的OMP/BLAS/MKL/Torch线程设为1，并默认限制Inductor编译线程为1（可由显式环境变量覆盖）。

默认 `--cache-clips 0` 将训练人体轨迹按原生全局帧号预载为连续CPU张量，仅选中的窗口按 `--library-transfer` 搬到GPU；不重复缓存整库机器人状态。`sync` 使用普通CPU内存并等待拷贝完成；`pinned` 先将选中批次放入锁页内存，再提交非阻塞GPU拷贝。这里没有独立预取stream，且每批锁页也有开销，不能保证pinned一定更快；两者数据、dtype及训练目标相同。1000条约190MB人体轨迹，全量约数十GB，需要相应内存。正数启用有界按片段缓存，但4096环境同时覆盖许多片段时会反复读盘，只适合内存受限时的取舍。

`--from-scratch` 可用本地默认网络从零开始，但仍采用本地原训练预热机制（40次仅采集后开始更新），没有移植上游额外500次WM-only阶段；当前首轮验证采用已经训练好的cpu7世界模型。

## 日志、保存和恢复

除原TensorBoard外新增 `joint_metrics.jsonl` 与 `[JOINT]` 行：`action_mse`、`latent_mse`、`collected.motion/human`、`modules_updated.wm/motion/human`。验证误差是teacher匹配误差；`val_imagined_survival`是模型预测，均不是整段真仿真完成率。原 `train_ret` 仍为存活步平均的训练统计，不能当整段评测分数。

每4000次保存 `ckpt_0004000.pt` 等及 `ckpt_latest.pt`，正常结束或Ctrl+C保存latest。数值异常不覆盖上一个正常latest。

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/human/train_joint.py \
  --resume logs/casbot02_seed_motion_human_all/ckpt_latest.pt --iterations 60000
```

恢复motion/WM/critic/human、四套优化器、human归一化/损失尺度、采样器、计数器和RNG；严格核对数据身份/分组、机器人资产和human接口配置。旧checkpoint续训仍保留其已保存的验证比例，不会因新默认值而改变数据分组；旧10%留出任务改为全数据应新建任务。replay和物理环境不保存，恢复后按原流程重新采集40步，因此不是逐位复现中断前物理状态。请用此独立入口恢复联合训练；原 `sm-train --resume` 不会更新额外human分支。

允许显式调整续训batch，保留权重和优化器状态并在joint_runtime/checkpoint记录变化。把motion/human都设为4096、参考数据继续放CPU：

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/human/train_joint.py \
  --resume logs/casbot02_seed_motion_human_all/ckpt_latest.pt \
  --num-envs 4096 --motion-batch 4096 --human-batch 4096 \
  --wm-batch 2048 --library-device cpu --library-transfer sync \
  --cache-clips 0 --cpu-threads 1 --compile default --iterations 60000
```

这里使用cpu7 runtime.json记录的CPU库＋sync搬运；此前联合任务用的是CPU库＋pinned搬运。WM batch仍为2048，与上游默认一致；若也要改WM，可显式用 --wm-batch 4096。CPU存储指原始参考库，采样批次、模型、激活与replay仍按原训练方式在GPU。增大batch仍会增加GPU峰值；4096/4096的pinned配置已在本机出现Warp OOM，sync配置也未通过该批次组合的显存验证。

## 双接口导出

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/human/export_joint.py \
  --checkpoint logs/casbot02_seed_motion_human_all/ckpt_latest.pt \
  --output-dir outputs/casbot02_seed_motion_human_all_export
```

输出共享同一decoder权重的 `motion_actor.pt`、`human_actor.pt` 和机器人契约/输入manifest；比较full/five及多个batch下TorchScript/PyTorch输出。motion权重保留原checkpoint结构，可用原motion评测/导出工具验证。联合训练会改变motion能力，应同时做原motion回归与human留出数据仿真评测。

此步骤没有替换真机部署，也没有把PICO发送器自动切成human。实时human接收、人体格式/时钟对齐以及原930维观测中的6D参考姿态构造仍需单独接入和仿真验证。
